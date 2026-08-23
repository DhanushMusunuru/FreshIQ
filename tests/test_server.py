"""
Tests for the local server's data layer (backend/server.py).

These cover the pure logic - history storage, stats, CSV - without binding a
socket, so they run fast and cannot flake on a busy port. The HTTP routing
itself is thin glue over these functions.
"""

import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from backend.server import History, Handler  # noqa: E402


def make_record(condition="fresh", food="tomato", ts="2026-08-15T10:00:00+00:00",
                fresh=80, spoiled=20, ident="insp_test"):
    return {
        "inspection_id": ident,
        "device_id": "freshiq-unit-01",
        "timestamp": ts,
        "elapsed_ms": 42.0,
        "demo": False,
        "result": {
            "status": "ok",
            "food": food,
            "condition": condition,
            "confidence": {"fresh": fresh, "spoiled": spoiled},
            "evidence": ["dark spots / bruising"] if condition == "spoiled" else [],
            "image_quality": "good",
            "views_used": 4,
            "views_total": 4,
        },
    }


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def test_append_and_read_newest_first(tmp_path):
    h = History(str(tmp_path / "d" / "hist.jsonl"))
    for i in range(3):
        h.append(make_record(ident=f"insp_{i}", ts=f"2026-08-15T10:0{i}:00+00:00"))
    items = h.read(10)
    assert [r["inspection_id"] for r in items] == ["insp_2", "insp_1", "insp_0"]


def test_read_respects_limit(tmp_path):
    h = History(str(tmp_path / "hist.jsonl"))
    for i in range(10):
        h.append(make_record(ident=f"insp_{i}"))
    assert len(h.read(4)) == 4
    assert len(h.all()) == 10


def test_missing_file_reads_empty(tmp_path):
    assert History(str(tmp_path / "nope.jsonl")).read(5) == []


def test_torn_final_line_is_skipped(tmp_path):
    """A power cut mid-write must cost one record, not the whole file."""
    p = tmp_path / "hist.jsonl"
    h = History(str(p))
    h.append(make_record(ident="good_1"))
    h.append(make_record(ident="good_2"))
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"inspection_id": "torn", "resu')      # truncated
    items = h.read(10)
    assert [r["inspection_id"] for r in items] == ["good_2", "good_1"]


def test_concurrent_appends_do_not_interleave(tmp_path):
    """The dashboard can fire overlapping requests; lines must stay whole."""
    h = History(str(tmp_path / "hist.jsonl"))

    def worker(n):
        for i in range(20):
            h.append(make_record(ident=f"t{n}_{i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(tmp_path / "hist.jsonl", encoding="utf-8") as fh:
        lines = [l for l in fh.read().split("\n") if l.strip()]
    assert len(lines) == 80
    for line in lines:
        json.loads(line)          # every line must parse


# --------------------------------------------------------------------------- #
# Stats + CSV  (called on the Handler class, no socket involved)
# --------------------------------------------------------------------------- #
@pytest.fixture
def handler(tmp_path):
    h = Handler.__new__(Handler)          # bypass BaseHTTPRequestHandler.__init__
    h.history = History(str(tmp_path / "hist.jsonl"))
    return h


def test_stats_counts_by_condition(handler):
    handler.history.append(make_record("fresh"))
    handler.history.append(make_record("spoiled"))
    handler.history.append(make_record("spoiled"))
    s = handler._stats()
    assert s["total"] == 3
    assert s["by_condition"] == {"fresh": 1, "spoiled": 2}
    assert s["spoiled_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_stats_empty_history(handler):
    s = handler._stats()
    assert s["total"] == 0 and s["spoiled_rate"] == 0.0


def test_stats_groups_by_food(handler):
    handler.history.append(make_record(food="tomato"))
    handler.history.append(make_record(food="banana"))
    handler.history.append(make_record(food="tomato"))
    assert handler._stats()["by_food"] == {"tomato": 2, "banana": 1}


def test_stats_handles_rejected_inspection(handler):
    """An unusable frame has condition None - it must not crash the counter."""
    rec = make_record()
    rec["result"] = {"status": "invalid_image", "condition": None,
                     "food": "unknown", "reason": "blurred"}
    handler.history.append(rec)
    s = handler._stats()
    assert s["by_condition"] == {"unknown": 1}


def test_csv_has_header_and_one_row_per_record(handler):
    handler.history.append(make_record("fresh", "tomato"))
    handler.history.append(make_record("spoiled", "banana"))
    rows = [r for r in handler._csv().strip().split("\n") if r]
    # "compartment" sits between inspection_id and food: every result now
    # belongs to a tray position, and a CSV without it cannot be matched back
    # to what was physically in the machine.
    assert rows[0].startswith(
        "timestamp,inspection_id,compartment,food,condition")
    assert len(rows) == 3                      # header + 2


def test_csv_compartment_blank_for_pre_tray_records(handler):
    """Single-shot records predate compartments; they must still export."""
    handler.history.append(make_record("fresh", "tomato"))
    rows = handler._csv().strip().split("\n")
    header = rows[0].split(",")
    row = rows[1].split(",")
    assert row[header.index("compartment")] == ""


def test_csv_is_oldest_first(handler):
    handler.history.append(make_record(ident="first", ts="2026-08-15T10:00:00+00:00"))
    handler.history.append(make_record(ident="second", ts="2026-08-15T11:00:00+00:00"))
    body = handler._csv()
    assert body.index("first") < body.index("second")


def test_capture_layout_matches_what_training_expects(tmp_path):
    """
    The capture endpoint writes dataset/<split>/<condition>/<food>/*.jpg .
    training/dataset.py must read that back as a NESTED layout, otherwise
    captured photos train a one-head model and the food head is silently lost.

    This pins the contract between the two halves.
    """
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "training")))
    import dataset as D

    root = tmp_path / "train"
    for cond, food in [("fresh", "tomato"), ("fresh", "banana"), ("spoiled", "tomato")]:
        d = root / cond / food
        d.mkdir(parents=True, exist_ok=True)
        (d / "rig_abc123.jpg").write_bytes(b"x")

    layout, conditions, foods = D.detect_layout(str(root))
    assert layout == "nested"
    assert conditions == ["fresh", "spoiled"]
    assert foods == ["banana", "tomato"]

    total, counts, _ = D.count_images(str(root))
    assert total == 3
    assert counts == {"fresh": 2, "spoiled": 1}


def test_csv_joins_evidence_without_breaking_columns(handler):
    rec = make_record("spoiled")
    rec["result"]["evidence"] = ["visible mould", "dark discoloration"]
    handler.history.append(rec)
    import csv as _csv
    import io
    rows = list(_csv.reader(io.StringIO(handler._csv())))
    assert len(rows[1]) == len(rows[0])         # no stray column from the join
    assert "visible mould; dark discoloration" in rows[1]
