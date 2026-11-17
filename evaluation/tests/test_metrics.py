
import pytest

from evaluation.metrics import evaluate_ranking
from evaluation.evaluator.retrieval_evaluator import percentile


def ground_truth():
    return [
        {"chunk_id": "A", "relevance": 2},
        {"chunk_id": "B", "relevance": 1},
    ]


def test_perfect_ranking():
    result = evaluate_ranking(
        ["A", "B", "C"],
        ground_truth(),
        ks=(1, 2),
    )

    assert result["1"]["precision"] == 1.0
    assert result["1"]["recall"] == 0.5
    assert result["1"]["mrr"] == 1.0
    assert result["1"]["ndcg"] == 1.0

    assert result["2"]["precision"] == 1.0
    assert result["2"]["recall"] == 1.0
    assert result["2"]["mrr"] == 1.0
    assert result["2"]["ndcg"] == 1.0


def test_relevant_chunk_at_second_position():
    result = evaluate_ranking(
        ["X", "A", "B"],
        ground_truth(),
        ks=(1, 2),
    )

    assert result["1"]["precision"] == 0.0
    assert result["1"]["recall"] == 0.0
    assert result["2"]["precision"] == 0.5
    assert result["2"]["recall"] == 0.5
    assert result["2"]["mrr"] == 0.5


def test_no_relevant_results():
    result = evaluate_ranking(
        ["X", "Y"],
        ground_truth(),
        ks=(1, 3),
    )

    assert result["1"]["recall"] == 0.0
    assert result["3"]["recall"] == 0.0
    assert result["3"]["mrr"] == 0.0
    assert result["3"]["ndcg"] == 0.0
    assert result["3"]["hit_rate"] == 0.0


def test_precision_uses_requested_k():
    result = evaluate_ranking(
        ["A"],
        ground_truth(),
        ks=(3,),
    )

    assert result["3"]["precision"] == pytest.approx(1 / 3)
    assert result["3"]["recall"] == 0.5


def test_duplicate_retrieved_ids_rejected():
    with pytest.raises(ValueError):
        evaluate_ranking(
            ["A", "A"],
            ground_truth(),
            ks=(1,),
        )


def test_empty_ground_truth_rejected():
    with pytest.raises(ValueError):
        evaluate_ranking(
            ["A"],
            [],
            ks=(1,),
        )


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        evaluate_ranking(
            ["A"],
            ground_truth(),
            ks=(0,),
        )


def test_percentiles():
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 25.0
    assert percentile(values, 100) == 40.0
