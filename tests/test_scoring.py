from tokenbench.evals import load_task, score_classification, score_extraction, score_math
from tokenbench.metrics import percentile
from tokenbench.router import choose_tier
from tokenbench.evals import build_prompt


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5.0], 99) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(101)), 90) == 90


def test_score_math():
    assert score_math("7 * 3 = 21, 50 - 21 = 29\nAnswer: 29", 29) == 1
    assert score_math("Answer: **$2.20**", 2.2) == 1
    assert score_math("so the total is 97,200", 97200) == 1
    assert score_math("Answer: 30", 29) == 0
    assert score_math("I don't know", 29) == 0


def test_score_extraction():
    answer = {"order_id": "4471", "customer": "Daniel Okafor", "date": "2024-03-18", "quantity": 3, "product": "X200 headset"}
    perfect = '```json\n{"order_id": "#4471", "customer": "Daniel Okafor", "date": "2024-03-18", "quantity": 3, "product": "X200 headsets"}\n```'
    assert score_extraction(perfect, answer) == 1
    assert score_extraction('{"order_id": "4471", "customer": "Dan", "quantity": "3"}', answer) == 0.4
    assert score_extraction("not json", answer) == 0


def test_score_classification():
    assert score_classification("billing", "billing") == 1
    assert score_classification("Bug report", "bug_report") == 1
    assert score_classification("shipping", "billing") == 0


def test_eval_data_is_well_formed():
    for task in ("math", "extraction", "classification"):
        items = load_task(task)
        assert len(items) >= 10
        assert len({i["id"] for i in items}) == len(items)


def test_router_tiers():
    assert choose_tier(build_prompt("math", load_task("math")[0]))[0] == "large"
    assert choose_tier(build_prompt("extraction", load_task("extraction")[0]))[0] == "small"
    assert choose_tier(build_prompt("classification", load_task("classification")[0]))[0] == "small"
    assert choose_tier("Write a haiku about rain.")[0] == "small"
