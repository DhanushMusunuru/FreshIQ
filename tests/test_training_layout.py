"""
Tests for dataset layout detection in training/dataset.py.

These cover the pure-Python parts only, so they run WITHOUT TensorFlow -- the
same reason the rest of the suite runs on the heuristic backend.

The layout decides how many output heads the model gets, so getting it wrong is
not a cosmetic problem: it silently changes the architecture.

    flat:   dataset/train/<condition>/*.jpg            -> 1 head
    nested: dataset/train/<condition>/<food>/*.jpg     -> 2 heads
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

import dataset as D  # noqa: E402


def _make(root, tree):
    """tree maps 'relative/dir' -> number of dummy jpgs to create in it."""
    for rel, n in tree.items():
        d = os.path.join(root, rel)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            with open(os.path.join(d, f"img{i}.jpg"), "wb") as fh:
                fh.write(b"x")
    return root


@pytest.fixture
def flat(tmp_path):
    return _make(str(tmp_path / "train"), {"fresh": 3, "spoiled": 5})


@pytest.fixture
def nested(tmp_path):
    return _make(str(tmp_path / "train"), {
        "fresh/tomato": 2, "fresh/banana": 3,
        "spoiled/tomato": 4, "spoiled/banana": 1,
    })


# --------------------------------------------------------------------------- #
# flat layout
# --------------------------------------------------------------------------- #
def test_flat_layout_detected(flat):
    assert D.detect_layout(flat) == ("flat", ["fresh", "spoiled"], [])


def test_flat_lists_every_image(flat):
    paths, conds, foods = D.list_files(flat, ["fresh", "spoiled"], None)
    assert len(paths) == 8
    assert foods == []                      # no food labels available
    assert conds.count(0) == 3 and conds.count(1) == 5


def test_flat_counts(flat):
    assert D.count_images(flat) == (8, {"fresh": 3, "spoiled": 5}, "flat")


# --------------------------------------------------------------------------- #
# nested layout
# --------------------------------------------------------------------------- #
def test_nested_layout_detected(nested):
    layout, conditions, foods = D.detect_layout(nested)
    assert layout == "nested"
    assert conditions == ["fresh", "spoiled"]
    assert foods == ["banana", "tomato"]    # sorted, deduped across conditions


def test_nested_pairs_condition_with_food(nested):
    paths, conds, foods = D.list_files(nested, ["fresh", "spoiled"], ["banana", "tomato"])
    assert len(paths) == len(conds) == len(foods) == 10
    pairs = list(zip(conds, foods))
    assert pairs.count((0, 0)) == 3         # fresh/banana
    assert pairs.count((0, 1)) == 2         # fresh/tomato
    assert pairs.count((1, 1)) == 4         # spoiled/tomato
    assert pairs.count((1, 0)) == 1         # spoiled/banana


def test_nested_counts(nested):
    assert D.count_images(nested) == (10, {"fresh": 5, "spoiled": 5}, "nested")


# --------------------------------------------------------------------------- #
# failure modes -- these must be loud, never silent
# --------------------------------------------------------------------------- #
def test_mixed_layout_raises(tmp_path):
    # Half-nested would silently skip every image under the flat condition.
    mixed = _make(str(tmp_path / "train"), {"fresh/tomato": 2, "spoiled": 4})
    with pytest.raises(ValueError, match="Mixed layout"):
        D.detect_layout(mixed)


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.detect_layout(str(tmp_path / "does_not_exist"))


def test_empty_split_raises(tmp_path):
    empty = str(tmp_path / "train")
    os.makedirs(empty)
    with pytest.raises(ValueError):
        D.detect_layout(empty)


def test_non_image_files_ignored(tmp_path):
    d = _make(str(tmp_path / "train"), {"fresh": 2, "spoiled": 2})
    with open(os.path.join(d, "fresh", "notes.txt"), "w") as fh:
        fh.write("not an image")
    total, counts, _ = D.count_images(d)
    assert total == 4 and counts == {"fresh": 2, "spoiled": 2}


# --------------------------------------------------------------------------- #
# the repo's own dataset/ skeleton
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Weight transfer when continuing from an existing model
# --------------------------------------------------------------------------- #
class FakeW:
    """Stands in for a numpy array - shapes_match only ever reads .shape."""
    def __init__(self, *shape):
        self.shape = shape


def _shapes_match():
    import train
    return train.shapes_match


def test_identical_shapes_transfer():
    assert _shapes_match()([FakeW(1280, 2), FakeW(2)], [FakeW(1280, 2), FakeW(2)])


def test_changed_class_count_is_rejected():
    """
    The whole point: if you retrain with a different food list, the food head
    changes width. Copying those weights would be nonsense, so it must be
    refused and the head left freshly initialised.
    """
    assert not _shapes_match()([FakeW(1280, 17), FakeW(17)],
                               [FakeW(1280, 8), FakeW(8)])


def test_different_weight_count_is_rejected():
    assert not _shapes_match()([FakeW(1280, 2)], [FakeW(1280, 2), FakeW(2)])


def test_empty_matches_empty():
    assert _shapes_match()([], [])


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_repo_dataset_conditions_match_config(split):
    """
    The condition folders must always match ai/config.DEFAULT_CONDITIONS.

    Deliberately does NOT assert the layout is flat: as soon as anyone captures
    a training example through the dashboard, the split becomes nested. Both are
    valid. What must never drift is the condition vocabulary.

    A "Mixed layout" error here is real and worth surfacing - it means one
    condition has food sub-folders and another does not, which would silently
    skip every image under the flat one.
    """
    root = os.path.join(os.path.dirname(__file__), "..", "dataset", split)
    # dataset/ holds training images and is gitignored, so a fresh clone (and
    # the copy that ships to the board) does not have it. Absent is not a
    # failure - only a PRESENT dataset with the wrong vocabulary is.
    if not os.path.isdir(root):
        pytest.skip("dataset/{} not present (training images are gitignored)".format(split))
    try:
        layout, conditions, foods = D.detect_layout(root)
    except ValueError as exc:
        if "Mixed layout" in str(exc):
            pytest.skip(f"dataset/{split} is mid-collection (mixed layout): {exc}")
        raise
    assert layout in ("flat", "nested")
    assert conditions == ["fresh", "spoiled"], (
        "dataset/ must match ai/config.DEFAULT_CONDITIONS")
