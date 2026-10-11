from __future__ import annotations

from collections import defaultdict


def summarize(results: dict[str, dict]) -> dict:
    recs = list(results.values())
    n = len(recs)
    wins = [r for r in recs if r["success"]]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_cat[r["category"]].append(r)
    return {
        "n": n,
        "asr": len(wins) / n if n else 0.0,
        "mean_queries_to_success": (
            sum(r["queries"] for r in wins) / len(wins) if wins else None
        ),
        "mean_queries_all": sum(r["queries"] for r in recs) / n if n else 0.0,
        "by_category": {
            c: sum(r["success"] for r in rs) / len(rs) for c, rs in sorted(by_cat.items())
        },
    }
