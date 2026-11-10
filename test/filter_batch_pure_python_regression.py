#!/usr/bin/env python3
"""CPU-only regression for ScheduleBatch.filter_batch keep-index selection.

Extracts the real `if keep_indices is None:` block from
python/sglang/srt/managers/schedule_batch.py via AST and executes it with
stdlib fake requests. No torch/CUDA import, no server, no network.
"""

import ast
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "managers" / "schedule_batch.py"


def _find_filter_batch(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ScheduleBatch":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "filter_batch":
                    return item
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "filter_batch":
            return node
    raise AssertionError("ScheduleBatch.filter_batch not found")


def _is_keep_indices_assign(stmt):
    if isinstance(stmt, ast.If):
        return any(_is_keep_indices_assign(s) for s in stmt.body) or any(
            _is_keep_indices_assign(s) for s in stmt.orelse
        )
    if isinstance(stmt, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == "keep_indices"
            for target in stmt.targets
        )
    return False


def extract_keep_indices_block():
    text = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    method = _find_filter_batch(tree)
    block = None
    for stmt in method.body:
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "keep_indices"
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Constant)
            and stmt.test.comparators[0].value is None
        ):
            block = stmt
            break
    if block is None:
        raise AssertionError("if keep_indices is None: block not found")
    lines = text.splitlines()
    indent = block.col_offset
    chunk = []
    for line in lines[block.lineno - 1:block.end_lineno]:
        chunk.append(line[indent:] if line[:indent].strip() == "" else line)
    return "\n".join(chunk), block


EXTRACTED_SOURCE, EXTRACTED_BLOCK = extract_keep_indices_block()


def _selection_statement():
    for stmt in EXTRACTED_BLOCK.body:
        if _is_keep_indices_assign(stmt):
            return stmt
    raise AssertionError("keep_indices selection statement not found")


def _has_notin_compare(node):
    return any(
        isinstance(inner, ast.Compare)
        and any(isinstance(op, ast.NotIn) for op in inner.ops)
        for inner in ast.walk(node)
    )


class FakeReq:
    __hash__ = None  # set()/dict() use on requests or exclusions would TypeError

    def __init__(self, name, is_finished=False):
        self.name = name
        self._is_finished = is_finished

    def finished(self):
        return self._is_finished

    def __repr__(self):
        return "FakeReq(%r, finished=%r)" % (self.name, self._is_finished)


class EqReq(FakeReq):
    __hash__ = None

    def __eq__(self, other):
        return isinstance(other, EqReq) and self.name == other.name

    def __repr__(self):
        return "EqReq(%r)" % (self.name,)


class TrapEmptyList(list):
    def __contains__(self, item):
        raise AssertionError("membership test ran against an empty exclusion list")


class RecordingList(list):
    def __init__(self, items):
        super().__init__(items)
        self.contains_calls = 0

    def __contains__(self, item):
        self.contains_calls += 1
        return super().__contains__(item)


def run_extracted(reqs, reqs_to_exclude, keep_indices=None):
    namespace = {
        "Req": FakeReq,
        "self": types.SimpleNamespace(reqs=reqs),
        "reqs_to_exclude": reqs_to_exclude,
        "keep_indices": keep_indices,
    }
    exec(compile(EXTRACTED_SOURCE, "<schedule_batch.filter_batch>", "exec"), namespace)
    return namespace["keep_indices"]


def make_reqs(cls=FakeReq):
    return [
        cls("r0", False),
        cls("r1", True),
        cls("r2", False),
        cls("r3", True),
        cls("r4", False),
    ]


class KeepIndicesSelectionTest(unittest.TestCase):
    def test_none_exclusions_selects_nonfinished(self):
        self.assertEqual(run_extracted(make_reqs(), None), [0, 2, 4])

    def test_empty_exclusions_selects_nonfinished(self):
        self.assertEqual(run_extracted(make_reqs(), []), [0, 2, 4])

    def test_empty_exclusion_list_is_not_probed(self):
        self.assertEqual(run_extracted(make_reqs(), TrapEmptyList()), [0, 2, 4])

    def test_all_finished_selects_nothing(self):
        reqs = [FakeReq("r0", True), FakeReq("r1", True)]
        self.assertEqual(run_extracted(reqs, None), [])
        self.assertEqual(run_extracted(reqs, TrapEmptyList()), [])

    def test_partial_finished_nonempty_exclusions_use_list_membership(self):
        reqs = make_reqs()
        exclusions = RecordingList([reqs[0], reqs[3]])
        self.assertEqual(run_extracted(reqs, exclusions), [2, 4])
        self.assertEqual(exclusions.contains_calls, 3)

    def test_single_req_exclusion(self):
        reqs = make_reqs()
        self.assertEqual(run_extracted(reqs, reqs[2]), [0, 4])

    def test_nonempty_exclusions_with_duplicates(self):
        reqs = make_reqs()
        self.assertEqual(run_extracted(reqs, [reqs[1], reqs[1]]), [0, 2, 4])

    def test_partial_finished_with_overlapping_exclusion(self):
        reqs = make_reqs()
        self.assertEqual(run_extracted(reqs, [reqs[1], reqs[2]]), [0, 4])

    def test_multiple_nonempty_exclusions(self):
        reqs = make_reqs()
        self.assertEqual(run_extracted(reqs, [reqs[0], reqs[2]]), [4])

    def test_equality_based_nonempty_membership_without_hash(self):
        reqs = [EqReq("a", False), EqReq("b", False), EqReq("c", True)]
        self.assertEqual(run_extracted(reqs, [EqReq("b", False)]), [0])

    def test_explicit_keep_indices_untouched(self):
        reqs = make_reqs()
        explicit = [1, 3]
        result = run_extracted(reqs, None, explicit)
        self.assertEqual(result, [1, 3])
        self.assertIs(result, explicit)

    def test_explicit_empty_keep_indices_untouched(self):
        reqs = make_reqs()
        self.assertEqual(run_extracted(reqs, [], []), [])
        self.assertEqual(run_extracted(reqs, None, []), [])

    def test_empty_reqs(self):
        self.assertEqual(run_extracted([], None), [])
        self.assertEqual(run_extracted([], TrapEmptyList()), [])

    def test_all_unfinished_none_exclusions(self):
        reqs = [FakeReq("r0", False), FakeReq("r1", False)]
        self.assertEqual(run_extracted(reqs, None), [0, 1])


class ExtractedBlockStructureTest(unittest.TestCase):
    def test_truthiness_split_and_list_membership_only_when_nonempty(self):
        selection = _selection_statement()
        self.assertIsInstance(selection, ast.If)
        self.assertIsInstance(selection.test, ast.Name)
        self.assertEqual(selection.test.id, "reqs_to_exclude")
        self.assertEqual(len(selection.body), 1)
        self.assertEqual(len(selection.orelse), 1)
        nonempty_assign = selection.body[0]
        empty_assign = selection.orelse[0]
        self.assertIsInstance(nonempty_assign, ast.Assign)
        self.assertIsInstance(empty_assign, ast.Assign)
        nonempty_comp = nonempty_assign.value
        empty_comp = empty_assign.value
        self.assertIsInstance(nonempty_comp, ast.ListComp)
        self.assertIsInstance(empty_comp, ast.ListComp)
        nonempty_ifs = nonempty_comp.generators[0].ifs
        empty_ifs = empty_comp.generators[0].ifs
        self.assertTrue(any(_has_notin_compare(cond) for cond in nonempty_ifs))
        self.assertFalse(any(_has_notin_compare(cond) for cond in empty_ifs))
        self.assertEqual(len(empty_ifs), 1)
        self.assertFalse(
            any(isinstance(node, (ast.Set, ast.SetComp)) for node in ast.walk(selection))
        )

    def test_extracted_source_has_no_set_conversion(self):
        self.assertIn("if reqs_to_exclude:", EXTRACTED_SOURCE)
        self.assertNotIn("set(", EXTRACTED_SOURCE)
        self.assertNotIn("set (", EXTRACTED_SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
