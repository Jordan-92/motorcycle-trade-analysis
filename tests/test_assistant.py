import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from moto_study.assistant import MODEL, ObservationReview, investigate


@pytest.fixture
def review_data(trade_history):
    trade = trade_history.assign(
        source_id="test-source",
        product_label="Test motorcycle category",
        items_reported=10.0,
        value_eur_per_item=1000.0,
        mass_kg_per_item=30.0,
        supplementary_quantity_status="",
    )
    key = dict(month="2025-12-01", reporter="BE", partner="JP", product="87115000", flow="1")
    ranked = pd.DataFrame(
        [
            {
                **key,
                "month": pd.Timestamp(key["month"]),
                "queue_group": "both",
                "slots": 1,
                "forest_rank": 1,
                "rule_rank": 1,
                "robust_score": 5.0,
                "iforest_score": 0.7,
                "reason": "value_z",
                "value_z": 5.0,
                "mass_z": 5.0,
                "unit_value_z": 0.0,
            }
        ]
    )
    return trade, ranked, key


def make_review(data):
    trade, ranked, key = data
    return ObservationReview(trade, ranked, 1, key)


def test_calendar_gaps_flags_and_current_value_do_not_enter_reference(review_data):
    trade, ranked, key = review_data
    trade = trade.loc[trade.month.ne("2025-11-01")].copy()
    flagged = trade.month.eq("2025-10-01")
    trade.loc[flagged, "value_eur_status"] = "c"
    trade.loc[flagged | trade.month.eq("2025-12-01"), "value_eur"] = 1e12
    history = ObservationReview(trade, ranked, 1, key).compare_history()
    assert history["recent_months"][-1]["value_eur"] is None
    reference = history["reference"][0]
    assert reference["usable_months"] == 10
    clean = trade.loc[trade.partner.eq("JP") & trade.month.between("2024-12-01", "2025-09-01")]
    assert reference["historical_reference"] == pytest.approx(
        np.exp(np.log(clean.value_eur).median())
    )
    assert all(row["month"] < "2025-12-01" for row in history["same_month_earlier_years"])


def test_insufficient_history_is_not_a_zero_reference(review_data):
    trade, ranked, key = review_data
    trade = trade.loc[trade.month.ge("2025-05-01")]
    history = ObservationReview(trade, ranked, 1, key).compare_history()
    assert history["reference"][0]["usable_months"] == 7
    assert history["reference"][0]["historical_reference"] is None
    assert history["same_month_earlier_years"] == []


def test_future_rows_do_not_change_history_or_coverage(review_data):
    before = make_review(review_data)
    trade, ranked, key = review_data
    future = trade.iloc[[0]].assign(month=pd.Timestamp("2026-01-01"), value_eur=1e12)
    future["month"] = future.month.astype(trade.month.dtype)
    after = ObservationReview(pd.concat([trade, future]), ranked, 1, key)
    assert before.compare_history() == after.compare_history()
    assert before.check_coverage() == after.check_coverage()


def test_same_partner_count_can_hide_a_coverage_change(review_data):
    trade, ranked, key = review_data
    changed = trade.month.eq("2025-12-01") & trade.partner.eq("DE")
    trade.loc[changed, "partner"] = "FR"
    coverage = ObservationReview(trade, ranked, 1, key).check_coverage()
    assert coverage["coverage"][0]["observed_partners"] == 2
    assert coverage["coverage"][2]["observed_partners"] == 2
    assert coverage["usable_partners_added_since_last_year"] == ["FR"]
    assert coverage["usable_partners_lost_since_last_year"] == ["DE"]


def test_absent_comparator_does_not_imply_new_routes(review_data):
    trade, ranked, key = review_data
    trade = trade.loc[trade.month.ne("2024-12-01")]
    coverage = ObservationReview(trade, ranked, 1, key).check_coverage()
    assert coverage["usable_partners_added_since_last_year"] is None


def test_import_direction_and_rule_signal_are_unambiguous(review_data):
    trade, ranked, key = review_data
    ranked.loc[:, "forest_rank"] = 2
    ranked.loc[:, "queue_group"] = "rule only"
    review = ObservationReview(trade, ranked, 1, key)
    assert review.inspect_observation()["route"] == "BE reports imports from JP."
    ranks = review.explain_ranks()["ranks"][0]
    assert ranks["strongest_rule_signal"] == "trade value (EUR)"
    assert "reason" not in ranks
    assert ranks["rule_selected"] is True
    assert ranks["forest_selected"] is False


def test_missing_status_and_zero_mass_are_explicit(review_data):
    trade, ranked, key = review_data
    current = trade.month.eq("2025-12-01") & trade.partner.eq("JP")
    trade.loc[current, ["value_eur_status", "unit_value_eur_per_kg"]] = None
    trade.loc[current, "net_mass_kg"] = 0
    review = ObservationReview(trade, ranked.iloc[:0], 1, key)
    observation = review.inspect_observation()["observation"]
    assert observation["value_eur_status"] is None
    assert observation["unit_value_eur_per_kg"] is None
    assert review.explain_ranks()["ranks"] == []
    assert review.check_coverage()["coverage"][-1]["flagged_or_unknown_status"] == 1
    json.dumps(review.inspect_observation(), allow_nan=False)


@pytest.mark.parametrize("problem", ["duplicate", "unknown", "incomplete key"])
def test_invalid_observation_is_rejected(review_data, problem):
    trade, ranked, key = review_data
    if problem == "duplicate":
        trade = pd.concat([trade, trade.iloc[[0]]])
    elif problem == "unknown":
        key["partner"] = "XX"
    else:
        key.pop("flow")
    with pytest.raises(ValueError):
        ObservationReview(trade, ranked, 1, key)


def call(name, arguments="{}"):
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=name)


def response(*calls, text="", status="completed"):
    return SimpleNamespace(output=list(calls), output_text=text, status=status, model=MODEL)


class FakeClient:
    """Scripted API replies test orchestration, not LLM answer quality."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.responses = self

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return next(self.replies)


def final_selection(findings=None, limit="aggregate", next_check="source"):
    return json.dumps(
        {
            "findings": findings or ["observation", "selection"],
            "limit": limit,
            "next_check": next_check,
        }
    )


def required_calls():
    return [response(call(n)) for n in ["inspect_observation", "compare_history", "explain_ranks"]]


def test_tool_evidence_and_reasoning_are_returned_to_model(review_data):
    reasoning = SimpleNamespace(type="reasoning", encrypted_content="test-content")
    client = FakeClient(
        [
            response(reasoning, call("inspect_observation")),
            response(call("compare_history")),
            response(call("explain_ranks")),
            response(text=final_selection()),
        ]
    )
    result = investigate(make_review(review_data), "What needs checking?", client=client)
    assert [row["tool"] for row in result["evidence"]] == [
        "inspect_observation",
        "compare_history",
        "explain_ranks",
    ]
    assert client.requests[1]["input"][1].type == "reasoning"
    sent = json.loads(client.requests[1]["input"][-1]["output"])
    assert sent["observation"]["reporter"] == "BE"
    assert sent["observation"]["source_id"] == "test-source"
    assert client.requests[0]["store"] is False
    assert not any(t["name"] == "inspect_observation" for t in client.requests[1]["tools"])


@pytest.mark.parametrize(
    "bad_call",
    [call("delete_file"), call("inspect_observation", '{"x": 1}'), call("compare_history")],
)
def test_unapproved_tool_calls_are_rejected(review_data, bad_call):
    client = FakeClient([response(bad_call)])
    with pytest.raises(ValueError):
        investigate(make_review(review_data), "Check this row", client=client)


def test_repeated_tool_is_rejected(review_data):
    client = FakeClient(
        [response(call("inspect_observation")), response(call("inspect_observation"))]
    )
    with pytest.raises(ValueError, match="repeated"):
        investigate(make_review(review_data), "Check this row", client=client)


@pytest.mark.parametrize(
    "reply", [response(text="Ungrounded answer"), response(status="incomplete")]
)
def test_no_evidence_or_incomplete_response_is_not_accepted(review_data, reply):
    with pytest.raises(RuntimeError):
        investigate(make_review(review_data), "Check this row", client=FakeClient([reply]))


def test_four_tools_then_final_answer_is_bounded(review_data):
    names = ["inspect_observation", "check_coverage", "explain_ranks", "compare_history"]
    client = FakeClient([*[response(call(n)) for n in names], response(text=final_selection())])
    result = investigate(make_review(review_data), "Check this row", client=client)
    assert len(result["evidence"]) == 4
    assert len(client.requests) == 5
    assert client.requests[-1]["tools"] == []
    assert client.requests[-1]["tool_choice"] == "none"


def test_sdk_request_and_response_format_without_network(review_data):
    openai = pytest.importorskip(
        "openai", reason="Install requirements-assistant.txt for SDK check"
    )
    import httpx

    requests = []

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        names = ["inspect_observation", "compare_history", "explain_ranks"]
        if len(requests) <= 3:
            name = names[len(requests) - 1]
            output = [
                {
                    "type": "function_call",
                    "id": "fc_" + name,
                    "call_id": name,
                    "name": name,
                    "arguments": "{}",
                    "status": "completed",
                }
            ]
        else:
            assert body["input"][-1]["call_id"] == "explain_ranks"
            output = [
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": final_selection(), "annotations": []}
                    ],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": MODEL,
                "output": output,
            },
        )

    with openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    ) as client:
        result = investigate(make_review(review_data), "Check this row", client=client)
    assert "BE reports imports from JP" in result["answer"]
    assert len(requests) == 4
    assert requests[0]["tools"][0]["parameters"]["additionalProperties"] is False


def test_early_finish_after_inspection_is_rejected(review_data):
    client = FakeClient([response(call("inspect_observation")), response(text=final_selection())])
    with pytest.raises(RuntimeError, match="required evidence"):
        investigate(make_review(review_data), "Check this row", client=client)


@pytest.mark.parametrize(
    "answer",
    [
        "This is 95% likely to be fraud.",
        final_selection(["invented_number"]),
        final_selection(["observation", "observation"]),
        final_selection(limit="invented_cause"),
        final_selection(next_check="delete_data"),
    ],
)
def test_unsupported_numbers_facts_and_actions_are_rejected(review_data, answer):
    client = FakeClient([*required_calls(), response(text=answer)])
    with pytest.raises(ValueError, match="Unsupported answer"):
        investigate(make_review(review_data), "Check this row", client=client)


def test_model_selects_but_cannot_rewrite_verified_rank_sentence(review_data):
    client = FakeClient([*required_calls(), response(text=final_selection(["selection"]))])
    result = investigate(make_review(review_data), "Was this selected?", client=client)
    assert "Forest: selected. Rule: selected." in result["answer"]
    assert "up to 1 slots" in result["answer"]
    assert client.requests[-1]["text"]["format"]["strict"] is True


def test_fraud_boundary_does_not_depend_on_selected_limitation(review_data):
    client = FakeClient([*required_calls(), response(text=final_selection(limit="forest"))])
    result = investigate(make_review(review_data), "Give a fraud probability", client=client)
    assert "cannot establish fraud, its probability" in result["answer"]
