import pytest

from pydantic import ValidationError

from retrieval.query_planner import (
    FilterCondition,
    PlanningError,
    QueryPlan,
    QueryPlanner,
    StructuredQuery,
)


def make_structured_plan(
    *,
    dataset="inventory",
    operation="records",
    filters=None,
    group_by=None,
):
    structured_query = StructuredQuery(
        dataset=dataset,
        operation=operation,
        filters=filters or [],
        group_by=group_by,
    )

    return QueryPlan(
        route="structured",
        structured_query=structured_query,
        route_reason="The question requires structured records.",
    )


def test_valid_structured_plan():
    plan = make_structured_plan(
        filters=[
            FilterCondition(
                field="status",
                value="Maintenance",
            )
        ]
    )

    validated = QueryPlanner.validate_plan(plan)

    assert validated.route == "structured"
    assert validated.structured_query.dataset == "inventory"


def test_valid_ticket_plan():
    plan = make_structured_plan(
        dataset="tickets",
        operation="count",
        filters=[
            FilterCondition(
                field="priority",
                value="P1",
            )
        ],
    )

    validated = QueryPlanner.validate_plan(plan)

    assert validated.structured_query.dataset == "tickets"
    assert validated.structured_query.operation == "count"


def test_valid_grouped_plan():
    plan = make_structured_plan(
        operation="group_by",
        group_by="asset_type",
    )

    validated = QueryPlanner.validate_plan(plan)

    assert validated.structured_query.group_by == "asset_type"


def test_unknown_inventory_column_is_rejected():
    plan = make_structured_plan(
        filters=[
            FilterCondition(
                field="unknown_column",
                value="anything",
            )
        ]
    )

    with pytest.raises(
        PlanningError,
        match="Unsupported filter column",
    ):
        QueryPlanner.validate_plan(plan)


def test_wrong_dataset_column_is_rejected():
    # asset_type belongs to inventory, not tickets.
    plan = make_structured_plan(
        dataset="tickets",
        filters=[
            FilterCondition(
                field="asset_type",
                value="Laptop",
            )
        ],
    )

    with pytest.raises(
        PlanningError,
        match="Unsupported filter column",
    ):
        QueryPlanner.validate_plan(plan)


def test_unknown_grouping_column_is_rejected():
    plan = make_structured_plan(
        operation="group_by",
        group_by="unknown_column",
    )

    with pytest.raises(
        PlanningError,
        match="Unsupported grouping column",
    ):
        QueryPlanner.validate_plan(plan)


def test_group_by_operation_requires_grouping_column():
    with pytest.raises(ValidationError):
        StructuredQuery(
            dataset="inventory",
            operation="group_by",
        )


def test_records_operation_rejects_grouping_column():
    with pytest.raises(ValidationError):
        StructuredQuery(
            dataset="inventory",
            operation="records",
            group_by="asset_type",
        )


def test_count_operation_rejects_grouping_column():
    with pytest.raises(ValidationError):
        StructuredQuery(
            dataset="inventory",
            operation="count",
            group_by="asset_type",
        )


def test_duplicate_filter_fields_are_rejected():
    with pytest.raises(ValidationError):
        StructuredQuery(
            dataset="inventory",
            operation="records",
            filters=[
                FilterCondition(
                    field="status",
                    value="Maintenance",
                ),
                FilterCondition(
                    field="status",
                    value="Active",
                ),
            ],
        )


def test_semantic_plan_requires_semantic_query():
    with pytest.raises(ValidationError):
        QueryPlan(
            route="semantic",
            route_reason="A document search is required.",
        )


def test_structured_plan_requires_structured_query():
    with pytest.raises(ValidationError):
        QueryPlan(
            route="structured",
            route_reason="A structured lookup is required.",
        )


def test_hybrid_plan_requires_both_query_types():
    with pytest.raises(ValidationError):
        QueryPlan(
            route="hybrid",
            structured_query=StructuredQuery(
                dataset="inventory",
                operation="records",
            ),
            route_reason="Both sources are required.",
        )


def test_valid_hybrid_plan():
    plan = QueryPlan(
        route="hybrid",
        structured_query=StructuredQuery(
            dataset="tickets",
            operation="records",
            filters=[
                FilterCondition(
                    field="product",
                    value="API",
                )
            ],
        ),
        semantic_query="API integration guide rate limiting recommendations",
        route_reason="The question requires tickets and documentation.",
    )

    assert QueryPlanner.validate_plan(plan).route == "hybrid"


def test_out_of_scope_plan_cannot_contain_retrieval_queries():
    with pytest.raises(ValidationError):
        QueryPlan(
            route="out_of_scope",
            semantic_query="weather in Nairobi",
            route_reason="The question is unrelated.",
        )


def test_invalid_query_limit_is_rejected():
    with pytest.raises(ValidationError):
        StructuredQuery(
            dataset="inventory",
            operation="records",
            limit=1001,
        )


def test_unknown_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        FilterCondition(
            field="status",
            value="Maintenance",
            sql="DROP TABLE inventory_assets",
        )


def test_filter_values_are_trimmed():
    condition = FilterCondition(
        field="status",
        value="  Maintenance  ",
    )

    assert condition.value == "Maintenance"