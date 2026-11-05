from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field, model_validator

from retrieval.structured_store import DATASETS, get_structured_store


logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
PLANNER_MODEL = "qwen2.5:1.5b"

MAX_QUESTION_LENGTH = 2000
DEFAULT_RESULT_LIMIT = 100
MAX_FILTERS = 8


class PlanningError(RuntimeError):
    """Raised when a safe and complete retrieval plan cannot be produced."""


class StrictModel(BaseModel):
    """Reject unexpected fields and normalize surrounding whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class RetrievalNeeds(StrictModel):
    """Independent source requirements; Python derives the final route."""

    needs_structured: bool = Field(
        description="Whether exact inventory or support-ticket records are required."
    )

    needs_semantic: bool = Field(
        description="Whether NexaTech document content is required."
    )


class FilterCondition(StrictModel):
    """Canonical exact-match filter passed to the structured store."""

    field: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=500)


class StructuredQuery(StrictModel):
    """Safe application-level query plan. Never contains SQL."""

    dataset: Literal["inventory", "tickets"]

    operation: Literal["records", "count", "group_by"]

    filters: list[FilterCondition] = Field(
        default_factory=list,
        max_length=MAX_FILTERS,
    )

    group_by: str | None = Field(
        default=None,
        max_length=64,
    )

    limit: int = Field(
        default=DEFAULT_RESULT_LIMIT,
        ge=1,
        le=1000,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_operation(self) -> "StructuredQuery":

        if self.operation == "group_by" and not self.group_by:
            raise ValueError(
                "group_by operation requires a grouping column."
            )

        if self.operation != "group_by" and self.group_by is not None:
            raise ValueError(
                "group_by must only be set for group_by operations."
            )

        fields = [item.field for item in self.filters]

        if len(fields) != len(set(fields)):
            raise ValueError(
                "A filter column may only appear once."
            )

        return self


class QueryPlan(StrictModel):
    """Final validated route and retrieval instructions."""

    route: Literal[
        "structured",
        "semantic",
        "hybrid",
        "out_of_scope",
    ]

    structured_query: StructuredQuery | None = None

    semantic_query: str | None = Field(
        default=None,
        max_length=MAX_QUESTION_LENGTH,
    )

    route_reason: str = Field(
        min_length=1,
        max_length=250,
    )

    @model_validator(mode="after")
    def validate_route_consistency(self) -> "QueryPlan":

        has_structured = self.structured_query is not None

        has_semantic = bool(
            self.semantic_query and self.semantic_query.strip()
        )

        expected = {
            "structured": (True, False),
            "semantic": (False, True),
            "hybrid": (True, True),
            "out_of_scope": (False, False),
        }[self.route]

        if (has_structured, has_semantic) != expected:
            raise ValueError(
                f"Route {self.route!r} has inconsistent retrieval instructions."
            )

        return self


class FilterProposal(StrictModel):
    """Model-proposed filter with evidence from the original question."""

    field: str = Field(min_length=1, max_length=64)

    value: str = Field(min_length=1, max_length=500)

    evidence: str = Field(min_length=1, max_length=300)


class StructuredExtraction(StrictModel):
    """Intermediate model output, not executable until validated."""

    dataset: Literal["inventory", "tickets"]

    operation: Literal["records", "count", "group_by"]

    filters: list[FilterProposal] = Field(
        default_factory=list,
        max_length=MAX_FILTERS,
    )

    group_by: str | None = Field(
        default=None,
        max_length=64,
    )

    group_by_evidence: str | None = Field(
        default=None,
        max_length=300,
    )

    limit: int = Field(
        default=DEFAULT_RESULT_LIMIT,
        ge=1,
        le=1000,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_operation(self) -> "StructuredExtraction":

        if self.operation == "group_by" and not self.group_by:
            raise ValueError(
                "group_by operation requires a grouping column."
            )

        if self.operation != "group_by" and self.group_by is not None:
            raise ValueError(
                "group_by is only valid for group_by operations."
            )

        if self.group_by is None and self.group_by_evidence is not None:
            raise ValueError(
                "Grouping evidence requires a grouping column."
            )

        fields = [item.field for item in self.filters]

        if len(fields) != len(set(fields)):
            raise ValueError(
                "A filter column may only appear once."
            )

        return self


CLASSIFIER_PROMPT = """
You classify which NexaTech knowledge sources are needed to answer a question.
You do not answer the question. Evaluate both sources independently.

STRUCTURED SOURCE:
- inventory: asset_tag, asset_type, location, assigned_team, status
- tickets: ticket_id, created_at, priority, product, issue, resolution, status

Use it for exact asset/ticket lookups, record filters, counts and groupings.

DOCUMENT SOURCE:
NexaTech PDF, DOCX and TXT materials such as the IT Security Policy,
API Integration Guide, Cloud Operations Runbook, Product Atlas Manual,
Employee Handbook, Finance Expense Policy and onboarding documents.

Use it for policies, explanations, procedures, recommendations and guidance.

Set needs_structured=true only when structured records are needed.
Set needs_semantic=true only when document contents are needed.
Set both true when the question requires both sources.
Set both false when neither source is relevant.

Examples:
- "How many inventory assets are under maintenance?" -> true, false
- "Which support tickets have P1 priority?" -> true, false
- "What does the IT Security Policy say about passwords?" -> false, true
- "Which API tickets exist, and what does the API Integration Guide recommend?" -> true, true
- "What is the weather in Nairobi?" -> false, false
- "How many API tickets are open?" -> true, false
- "What does the API Integration Guide recommend about rate limiting?" -> false, true

Return only the two boolean fields in the supplied schema.
"""


EXTRACTION_PROMPT = """
You convert a question (or its structured-data portion) into a NexaTech query plan.
You do not answer the question and you never write SQL.

DATASETS AND COLUMNS

inventory:
asset_tag, asset_type, location, assigned_team, status

tickets:
ticket_id, created_at, priority, product, issue, resolution, status

OPERATIONS

- "how many", "count", "number of" -> count
- "grouped by", "group by", "breakdown by", "distribution by" -> group_by
- otherwise, "which", "list", "show" -> records

FILTER RULES

Every filter must have an evidence phrase copied verbatim from the user's question.
The evidence must explicitly support the chosen field and value.

Never infer an unmentioned value.
Never add defaults such as status=Open.

- "under maintenance" can filter status to the canonical database value Maintenance.
- "P1 priority" can filter priority to P1.
- "API tickets" or "API product tickets" can filter product to API.
- "open tickets" can filter status to Open only when the word Open is in the question.

GROUPING RULES

"grouped by status" means group_by=status, NOT status=Open or status=Closed.

The grouping phrase must be captured in group_by_evidence.
A grouping column is not automatically a filter.

HYBRID QUESTIONS

Extract only the structured-data portion.

For example:

"Which API tickets exist, and what does the API Integration Guide recommend?"

means dataset=tickets, operation=records, filter product=API with evidence "API tickets".

Do not invent filters from the document portion.

Return a complete result matching the supplied schema.
Use only registered datasets and columns.
If a proposed filter is not explicitly supported, omit it.
"""


# Only categorical columns are considered for deterministic fallback.
# Free-text issue and resolution fields are deliberately excluded.

FILTERABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "inventory": (
        "asset_type",
        "location",
        "assigned_team",
        "status",
    ),
    "tickets": (
        "priority",
        "product",
        "status",
    ),
}


GROUPING_ALIASES: dict[str, tuple[str, ...]] = {
    "asset_tag": ("asset tag", "asset_tag", "tag"),
    "asset_type": ("asset type", "asset types", "type", "types"),
    "location": ("location", "locations", "office", "region"),
    "assigned_team": ("assigned team", "team", "teams"),
    "status": ("status", "statuses"),
    "ticket_id": ("ticket id", "ticket_id", "ticket identifier"),
    "created_at": ("created date", "creation date", "date", "created_at"),
    "priority": ("priority", "priorities"),
    "product": ("product", "products"),
}


class QueryPlanner:
    """
    Two-stage LangChain planner with evidence and live-value validation.

    Stage 1 classifies source requirements using a compact Pydantic schema.
    Python derives the route deterministically.

    Stage 2 extracts a structured query. Its output is never trusted until
    evidence, dataset columns and actual SQLite values have been verified.

    A deterministic, database-backed fallback handles common explicit queries
    when the small local model proposes unsupported filters or grouping.
    """

    def __init__(
        self,
        *,
        model_name: str = PLANNER_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        llm: Any | None = None,
        structured_store: Any | None = None,
    ) -> None:

        self.llm = llm or ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            num_ctx=4096,
            num_predict=300,
        )

        self.structured_store = structured_store

        self.classifier = self.llm.with_structured_output(
            RetrievalNeeds,
            method="json_schema",
            include_raw=True,
        )

        self.extractor = self.llm.with_structured_output(
            StructuredExtraction,
            method="json_schema",
            include_raw=True,
        )

    @staticmethod
    def derive_route(needs: RetrievalNeeds) -> str:

        if needs.needs_structured and needs.needs_semantic:
            return "hybrid"

        if needs.needs_structured:
            return "structured"

        if needs.needs_semantic:
            return "semantic"

        return "out_of_scope"

    @staticmethod
    def validate_plan(plan: QueryPlan) -> QueryPlan:
        """Validate final query columns against the registered datasets."""

        structured = plan.structured_query

        if structured is None:
            return plan

        dataset = DATASETS.get(structured.dataset)

        if dataset is None:
            raise PlanningError(
                f"Unsupported dataset: {structured.dataset!r}"
            )

        allowed = set(dataset["columns"])

        for condition in structured.filters:

            if condition.field not in allowed:
                raise PlanningError(
                    f"Unsupported filter column {condition.field!r} "
                    f"for dataset {structured.dataset!r}."
                )

        if (
            structured.group_by is not None
            and structured.group_by not in allowed
        ):
            raise PlanningError(
                f"Unsupported grouping column {structured.group_by!r} "
                f"for dataset {structured.dataset!r}."
            )

        return plan

    @staticmethod
    def _parse_response(
        response: Any,
        expected_type: type[BaseModel],
    ) -> BaseModel:

        if not isinstance(response, dict):
            raise PlanningError(
                "Structured model returned an unexpected response."
            )

        parsing_error = response.get("parsing_error")

        if parsing_error is not None:
            raise PlanningError(
                "Model output did not match the required schema."
            ) from parsing_error

        parsed = response.get("parsed")

        if parsed is None:
            raise PlanningError(
                "Model returned no parsed structured output."
            )

        if isinstance(parsed, expected_type):
            return parsed

        return expected_type.model_validate(parsed)

    @staticmethod
    def _normalize(text: str) -> str:

        text = text.casefold().replace("_", " ")
        text = re.sub(r"[^a-z0-9]+", " ", text)

        return " ".join(text.split())

    @classmethod
    def _contains_phrase(cls, phrase: str, text: str) -> bool:

        normalized_phrase = cls._normalize(phrase)
        normalized_text = cls._normalize(text)

        if not normalized_phrase:
            return False

        if f" {normalized_phrase} " in f" {normalized_text} ":
            return True

        # Handle simple plural forms.
        words = normalized_phrase.split()

        if words:
            last = words[-1]
            plural_forms = {last + "s", last + "es"}

            for plural in plural_forms:
                plural_phrase = " ".join([*words[:-1], plural])

                if f" {plural_phrase} " in f" {normalized_text} ":
                    return True

        return False

    def _get_store(self):

        if self.structured_store is None:
            self.structured_store = get_structured_store()

        return self.structured_store

    def _get_column_values(
        self,
        dataset: str,
        field: str,
    ) -> list[str]:

        if field not in DATASETS[dataset]["columns"]:
            raise PlanningError(
                f"Unknown column {field!r} for {dataset!r}."
            )

        try:
            rows = self._get_store().query(
                dataset,
                group_by=field,
                limit=1000,
            )

        except Exception as error:
            raise PlanningError(
                f"Could not validate values for {dataset}.{field}."
            ) from error

        return [
            str(row["value"])
            for row in rows
            if row.get("value") is not None
        ]

    def _canonical_value(
        self,
        dataset: str,
        field: str,
        proposed_value: str,
        evidence: str,
    ) -> str:
        """Resolve a supported proposal to an exact stored database value."""

        proposed_norm = self._normalize(proposed_value)

        # First check the exact proposed value using a parameterized query.
        try:
            exact_match = self._get_store().query(
                dataset,
                filters={field: proposed_value},
                count=True,
            )

            if (
                exact_match
                and int(exact_match[0].get("total", 0)) > 0
            ):
                if self._contains_phrase(proposed_value, evidence):
                    return proposed_value

        except Exception as error:
            raise PlanningError(
                f"Could not verify proposed value for {dataset}.{field}."
            ) from error

        values = self._get_column_values(dataset, field)

        for actual in values:

            if self._normalize(actual) == proposed_norm:

                if self._contains_phrase(actual, evidence):
                    return actual

        # Map natural language such as "under maintenance"
        # to the canonical database value "Maintenance".

        evidence_matches = [
            actual
            for actual in values
            if self._contains_phrase(actual, evidence)
        ]

        if len(evidence_matches) == 1:
            return evidence_matches[0]

        raise PlanningError(
            f"Could not map proposed {dataset}.{field} value "
            f"{proposed_value!r} to one unambiguous stored value "
            f"supported by evidence {evidence!r}."
        )

    def _validate_extraction(
        self,
        question: str,
        extraction: StructuredExtraction,
    ) -> StructuredQuery:

        dataset = DATASETS.get(extraction.dataset)

        if dataset is None:
            raise PlanningError(
                f"Unsupported dataset: {extraction.dataset!r}"
            )

        allowed = set(dataset["columns"])

        expected_group = self._infer_group_by(
            extraction.dataset,
            question,
        )

        normalized_question = self._normalize(question)

        has_count_intent = bool(
            re.search(
                r"\b(how many|count|number of|total number)\b",
                normalized_question,
            )
        )

        expected_operation = (
            "group_by"
            if expected_group is not None
            else "count"
            if has_count_intent
            else "records"
        )

        if extraction.operation != expected_operation:
            raise PlanningError(
                f"The model selected operation {extraction.operation!r}, "
                f"but the question indicates {expected_operation!r}."
            )

        if (
            expected_group is not None
            and extraction.group_by != expected_group
        ):
            raise PlanningError(
                f"The question asks to group by {expected_group!r}, "
                f"but the model selected {extraction.group_by!r}."
            )

        if expected_group is None and extraction.group_by is not None:
            raise PlanningError(
                "The model added a grouping column without an "
                "explicit grouping request."
            )

        filters: list[FilterCondition] = []
        seen_fields: set[str] = set()

        for proposal in extraction.filters:

            if proposal.field not in allowed:
                raise PlanningError(
                    f"Unsupported filter field {proposal.field!r} "
                    f"for {extraction.dataset!r}."
                )

            if proposal.field in seen_fields:
                raise PlanningError(
                    f"Duplicate filter field {proposal.field!r}."
                )

            if not self._contains_phrase(proposal.evidence, question):
                raise PlanningError(
                    f"Filter evidence is not present in the question: "
                    f"{proposal.evidence!r}."
                )

            canonical = self._canonical_value(
                extraction.dataset,
                proposal.field,
                proposal.value,
                proposal.evidence,
            )

            filters.append(
                FilterCondition(
                    field=proposal.field,
                    value=canonical,
                )
            )

            seen_fields.add(proposal.field)

        group_by = extraction.group_by

        if group_by is not None:

            if group_by not in allowed:
                raise PlanningError(
                    f"Unsupported grouping column {group_by!r} "
                    f"for {extraction.dataset!r}."
                )

            evidence = extraction.group_by_evidence

            if (
                not evidence
                or not self._contains_phrase(evidence, question)
            ):
                raise PlanningError(
                    "Grouping evidence is absent from the question."
                )

            if not self._contains_phrase(group_by, evidence):
                raise PlanningError(
                    f"Grouping evidence {evidence!r} "
                    f"does not support {group_by!r}."
                )

        query = StructuredQuery(
            dataset=extraction.dataset,
            operation=extraction.operation,
            filters=filters,
            group_by=group_by,
            limit=extraction.limit,
        )

        # Independently derive the expected query from explicit terms.
        # This catches fabricated filters and omitted explicit filters.

        baseline = self._deterministic_extraction(
            question,
            extraction.dataset,
        )

        if (
            baseline.operation != query.operation
            or baseline.group_by != query.group_by
        ):
            raise PlanningError(
                "The model's operation or grouping does not match "
                "the explicit wording of the question."
            )

        proposed_pairs = {
            (item.field, item.value)
            for item in query.filters
        }

        for expected_filter in baseline.filters:

            if (
                expected_filter.field,
                expected_filter.value,
            ) not in proposed_pairs:
                raise PlanningError(
                    f"The model omitted an explicit filter: "
                    f"{expected_filter.field}={expected_filter.value}."
                )

        return query

    def _infer_dataset(
        self,
        question: str,
        hint: str | None = None,
    ) -> str:

        text = self._normalize(question)

        inventory_terms = (
            "inventory",
            "asset",
            "assets",
            "laptop",
            "laptops",
            "server",
            "servers",
            "monitor",
            "monitors",
            "network switch",
            "network switches",
        )

        ticket_terms = (
            "ticket",
            "tickets",
            "support",
            "priority",
            "p1",
            "p2",
            "p3",
            "incident",
            "incidents",
        )

        inventory_hit = any(
            self._contains_phrase(term, text)
            for term in inventory_terms
        )

        ticket_hit = any(
            self._contains_phrase(term, text)
            for term in ticket_terms
        )

        if inventory_hit and not ticket_hit:
            return "inventory"

        if ticket_hit and not inventory_hit:
            return "tickets"

        if hint in DATASETS:
            return hint

        if inventory_hit and ticket_hit:
            raise PlanningError(
                "The question refers to both inventory and tickets; "
                "the structured dataset is ambiguous."
            )

        raise PlanningError(
            "Could not confidently identify a structured dataset."
        )

    def _infer_group_by(
        self,
        dataset: str,
        question: str,
    ) -> str | None:

        normalized = self._normalize(question)

        markers = (
            "grouped by",
            "group by",
            "breakdown by",
            "distribution by",
            "by each",
            "per",
        )

        marker_positions = [
            normalized.find(marker)
            for marker in markers
            if marker in normalized
        ]

        if not marker_positions:
            return None

        suffix = normalized[min(marker_positions):]

        candidates: list[tuple[int, str]] = []

        for field in DATASETS[dataset]["columns"]:

            for alias in GROUPING_ALIASES.get(field, (field,)):

                if self._contains_phrase(alias, suffix):
                    candidates.append(
                        (len(self._normalize(alias)), field)
                    )

        if not candidates:
            raise PlanningError(
                "A grouping instruction was detected, but its "
                "grouping column could not be identified."
            )

        candidates.sort(reverse=True)

        return candidates[0][1]

    def _deterministic_extraction(
        self,
        question: str,
        dataset_hint: str | None = None,
    ) -> StructuredQuery:
        """Conservative fallback using registered columns and live DB values."""

        dataset = self._infer_dataset(
            question,
            dataset_hint,
        )

        group_by = self._infer_group_by(
            dataset,
            question,
        )

        normalized = self._normalize(question)

        if group_by is not None:

            operation: Literal[
                "records",
                "count",
                "group_by",
            ] = "group_by"

        elif re.search(
            r"\b(how many|count|number of|total number)\b",
            normalized,
        ):

            operation = "count"

        else:
            operation = "records"

        if operation == "group_by" and group_by is None:
            raise PlanningError(
                "Could not determine the requested grouping column."
            )

        # For a grouped query, values before the grouping clause may
        # still be real filters.

        prefix = normalized

        for marker in (
            "grouped by",
            "group by",
            "breakdown by",
            "distribution by",
        ):

            position = normalized.find(marker)

            if position >= 0:
                prefix = normalized[:position]
                break

        filters: list[FilterCondition] = []

        for field in FILTERABLE_COLUMNS[dataset]:

            if group_by == field:
                search_text = prefix
            else:
                search_text = normalized

            matched_values = [
                value
                for value in self._get_column_values(dataset, field)
                if self._contains_phrase(value, search_text)
            ]

            if len(matched_values) > 1:
                raise PlanningError(
                    f"The question matches multiple values "
                    f"for {dataset}.{field}: {matched_values!r}. "
                    "Please clarify."
                )

            if matched_values:
                filters.append(
                    FilterCondition(
                        field=field,
                        value=matched_values[0],
                    )
                )

        return StructuredQuery(
            dataset=dataset,
            operation=operation,
            filters=filters,
            group_by=group_by,
            limit=DEFAULT_RESULT_LIMIT,
        )

    def _build_structured_query(
        self,
        question: str,
    ) -> StructuredQuery:

        extraction_error: Exception | None = None
        dataset_hint: str | None = None

        try:

            response = self.extractor.invoke([
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(content=question),
            ])

            extraction = self._parse_response(
                response,
                StructuredExtraction,
            )

            dataset_hint = extraction.dataset

            return self._validate_extraction(
                question,
                extraction,
            )

        except Exception as error:

            extraction_error = error

            logger.warning(
                "LLM structured extraction was rejected; using conservative "
                "database-backed fallback. Reason: %s",
                error,
            )

        try:

            fallback = self._deterministic_extraction(
                question,
                dataset_hint,
            )

            self.validate_plan(
                QueryPlan(
                    route="structured",
                    structured_query=fallback,
                    route_reason=(
                        "Structured query derived and validated "
                        "from explicit question terms."
                    ),
                )
            )

            return fallback

        except Exception as fallback_error:

            raise PlanningError(
                "Could not construct a safe structured query. "
                "The model proposal failed validation and the deterministic "
                "fallback could not identify an unambiguous query."
            ) from (extraction_error or fallback_error)

    def plan(self, question: str) -> QueryPlan:
        """Classify sources, derive the route, and validate retrieval instructions."""

        if not isinstance(question, str):
            raise TypeError(
                "question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"question cannot exceed {MAX_QUESTION_LENGTH} characters."
            )

        try:

            response = self.classifier.invoke([
                SystemMessage(content=CLASSIFIER_PROMPT),
                HumanMessage(content=question),
            ])

            needs = self._parse_response(
                response,
                RetrievalNeeds,
            )

        except Exception as error:

            logger.exception(
                "Retrieval source classification failed."
            )

            raise PlanningError(
                "Could not determine which knowledge sources are required."
            ) from error

        route = self.derive_route(needs)

        if route == "out_of_scope":

            return QueryPlan(
                route="out_of_scope",
                route_reason=(
                    "Neither NexaTech structured data nor its "
                    "document collection is relevant."
                ),
            )

        structured_query = None

        if needs.needs_structured:
            structured_query = self._build_structured_query(question)

        semantic_query = (
            question
            if needs.needs_semantic
            else None
        )

        reasons = {
            "structured": "Exact structured records are required.",
            "semantic": "NexaTech document evidence is required.",
            "hybrid": (
                "Both structured records and document evidence are required."
            ),
        }

        plan = QueryPlan(
            route=route,
            structured_query=structured_query,
            semantic_query=semantic_query,
            route_reason=reasons[route],
        )

        return self.validate_plan(plan)


def get_query_planner() -> QueryPlanner:
    """Create the configured local Ollama query planner."""

    return QueryPlanner()