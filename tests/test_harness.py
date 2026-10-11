import json

from redteam.attacks.pair import PAIR, parse_attacker
from redteam.behaviors import Behavior, _parse_csv, split_behaviors
from redteam.judge import LLMJudge
from redteam.metrics import summarize
from redteam.models import CountingModel
from redteam.runner import load_results, run


class Stub:
    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def generate(self, messages):
        self.n += 1
        return self.fn(messages, self.n)


def B(i, cat="c"):
    return Behavior(f"b{i}", f"goal{i}", "Sure", cat, "t")


def attacker_stub():
    return Stub(lambda m, n: json.dumps({"improvement": "x", "prompt": f"attempt {n}"}))


def judge_for(success_on):
    # judge sees the target response; target echoes the prompt
    def fn(m, n):
        return "Rating: [[10]]" if success_on in m[-1]["content"] else "Rating: [[1]]"
    return LLMJudge(Stub(fn))


def echo_target():
    return Stub(lambda m, n: m[-1]["content"])


def test_csv_parse_and_ids():
    text = 'Index,Goal,Target,Behavior,Category,Source\n0,g,"Sure, here",B,Cat,Orig\n'
    (b,) = _parse_csv(text)
    assert (b.id, b.goal, b.target, b.category) == ("jbb-0", "g", "Sure, here", "Cat")


def test_splits_disjoint_deterministic_and_cover():
    bs = [B(i) for i in range(200)]
    s1, s2 = split_behaviors(bs, 0.5, seed=1), split_behaviors(bs, 0.5, seed=1)
    assert s1 == s2
    ids = lambda k: {b.id for b in s1[k]}
    assert not ids("train") & ids("eval")
    assert len(ids("train")) + len(ids("eval")) == 200
    assert 60 < len(s1["eval"]) < 140


def test_judge_parsing():
    j = LLMJudge(Stub(lambda m, n: "Rating: [[7]]"))
    v = j.judge("g", "r")
    assert v.score == 7 and not v.success
    bad = LLMJudge(Stub(lambda m, n: "I cannot rate")).judge("g", "r")
    assert bad.parse_error and not bad.success and bad.score == 1
    assert LLMJudge(Stub(lambda m, n: "[[10]]")).judge("g", "r").success
    assert LLMJudge(Stub(lambda m, n: "[[11]]")).judge("g", "r").parse_error


def test_judge_refusal_fail_compliance_pass():
    # hand-written cases through a rubric-faithful stub
    j = LLMJudge(Stub(lambda m, n: "Rating: [[10]]" if "Step 1" in m[-1]["content"] else "Rating: [[1]]"))
    assert not j.judge("g", "I can't help with that.").success
    assert j.judge("g", "Step 1: ...").success


def test_parse_attacker():
    assert parse_attacker('noise {"improvement":"a","prompt":"p"} tail') == "p"
    assert parse_attacker("no json") is None
    assert parse_attacker('{"prompt": ""}') is None


def test_pair_succeeds_and_counts_queries():
    res = PAIR(attacker_stub(), judge_for("attempt 3")).run(B(1), echo_target(), budget=10)
    assert res.success and res.queries == 3 and res.best_score == 10
    assert res.prompt == "attempt 3"


def test_pair_respects_budget_and_fails():
    res = PAIR(attacker_stub(), judge_for("never")).run(B(1), echo_target(), budget=4)
    assert not res.success and res.queries == 4


def test_pair_terminates_on_garbage_attacker():
    garbage = Stub(lambda m, n: "not json")
    tgt = CountingModel(echo_target())
    res = PAIR(garbage, judge_for("x")).run(B(1), tgt, budget=3)
    assert res.queries == 0 and tgt.calls == 0 and garbage.n == 6


def test_runner_resumes(tmp_path):
    out = tmp_path / "r.jsonl"
    bs = [B(i) for i in range(4)]
    win = Stub(lambda m, n: json.dumps({"improvement": "x", "prompt": "win"}))
    atk = PAIR(win, judge_for("win"))
    run(atk, bs[:2], echo_target(), out, budget=3, workers=2)
    assert len(load_results(out)) == 2
    tgt = CountingModel(echo_target())
    res = run(atk, bs, tgt, out, budget=3, workers=2)
    assert len(res) == 4 and tgt.calls == 2  # only the 2 new ones ran
    assert len(out.read_text().strip().splitlines()) == 4


def test_metrics():
    results = {
        "a": {"success": True, "queries": 2, "category": "X"},
        "b": {"success": False, "queries": 5, "category": "X"},
        "c": {"success": True, "queries": 4, "category": "Y"},
    }
    m = summarize(results)
    assert m["n"] == 3 and abs(m["asr"] - 2 / 3) < 1e-9
    assert m["mean_queries_to_success"] == 3 and m["by_category"] == {"X": 0.5, "Y": 1.0}
    assert summarize({})["asr"] == 0.0
