"""Boundary tests for the bounded local reader; no models or old artifacts."""

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from .. import reading


class ReadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="schema-read-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sources = self.root / "sources"
        self.artifacts = self.root / "artifacts"
        self.sources.mkdir()
        self.artifacts.mkdir()
        self.roots = [self.sources, self.artifacts]

    def write(self, name, text, *, artifact=False):
        path = (self.artifacts if artifact else self.sources) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
        return path

    def test_cli_pairs_bounded_pages_and_can_replay_either_page(self):
        path = self.write("long.txt", '行 "\\' + "x" * 250 + "\r\n" + "y" * 8000)
        requests = [{"path": str(path)}]
        command = [sys.executable, "-B", reading.__file__, "--root", str(self.sources), "--root", str(self.artifacts),
                   "--requests", json.dumps(requests), "--max-bytes", "1500"]
        expected = list(self.pages(requests, max_bytes=1500))
        self.assertGreater(len(expected), 2)
        for offset in range(0, len(expected), 2):
            cursor = ["--cursor", expected[offset]["cursor"]] if offset else []
            raw = subprocess.check_output([*command, "--pages", "2", *cursor], text=True)
            self.assertEqual(json.loads(raw), expected[offset:offset + 2])
            self.assertLessEqual(len(json.dumps(raw, ensure_ascii=False).encode()), 3000)
        for page in expected[:2]:
            replay = subprocess.check_output([*command, "--cursor", page["cursor"]], text=True)
            self.assertEqual(json.loads(replay), page)

    def read(self, requests, **kwargs):
        return reading.read_page(requests, roots=self.roots, **kwargs)

    def pages(self, requests, max_bytes=1500):
        cursor, seen = None, set()
        while True:
            page = self.read(requests, cursor=cursor, max_bytes=max_bytes)
            wrapped = json.dumps(reading.serialize_page(page), ensure_ascii=False, separators=(",", ":"))
            self.assertEqual(len(wrapped.encode("utf-8")), reading.output_size(page))
            self.assertLessEqual(reading.output_size(page), max_bytes)
            self.assertLessEqual(len(reading.serialize_page(page).encode("utf-8")), max_bytes)
            yield page
            if page["complete"]:
                self.assertIsNone(page["remaining"])
                self.assertIsNone(page["next_cursor"])
                return
            cursor = page["next_cursor"]
            self.assertNotIn(cursor, seen, "every page must advance")
            seen.add(cursor)
            self.assertLess(len(seen), 1000, "unexpected non-progress")

    def collect(self, requests, *, max_bytes=1500):
        """Check byte bounds and exact original positions on every fragment."""
        originals = [Path(request["path"]).read_bytes().decode("utf-8").splitlines(keepends=True)
                     for request in requests]
        selected = [lines[request.get("start_line", 1) - 1:request.get("end_line")]
                    for request, lines in zip(requests, originals)]
        received = ["" for _ in requests]
        positions = [(request.get("start_line", 1), 1) for request in requests]
        pages = []
        for page in self.pages(requests, max_bytes):
            for source in page["files"]:
                index = source["request_index"]
                self.assertEqual(str(Path(requests[index]["path"]).resolve()), source["path"])
                self.assertEqual(len(originals[index]), source["total_lines"])
                for fragment in source["fragments"]:
                    line, column = positions[index]
                    self.assertEqual((line, column), (fragment["line"], fragment.get("column", 1)))
                    text = fragment["text"]
                    self.assertTrue(text)
                    self.assertEqual(originals[index][line - 1][column - 1:column - 1 + len(text)], text)
                    ends_line = fragment.get("ends_line", True)
                    self.assertEqual(column - 1 + len(text) == len(originals[index][line - 1]), ends_line)
                    self.assertNotEqual(1, fragment.get("column"))
                    self.assertNotEqual(True, fragment.get("ends_line"))
                    received[index] += text
                    positions[index] = (line + 1, 1) if ends_line else (line, column + len(text))
            pages.append(page)
            if page["complete"]:
                continue
            remaining = page["remaining"]
            index = remaining["request_index"]
            self.assertEqual(positions[index], (remaining["line"], remaining.get("column", 1)))
            self.assertNotIn("path", remaining)
            self.assertNotEqual(1, remaining.get("column"))
            self.assertEqual(len(requests) - index - 1, remaining["following_requests"])
        self.assertEqual(["".join(lines) for lines in selected], received)
        return pages

    def test_batch_ranges_and_two_roots_preserve_all_lines(self):
        first = self.write("source.txt", "outside\n" + "中文 source\n" * 90 + "outside\n")
        second = self.write("draft.yaml", "a: 1\nb: 2\nc: 3", artifact=True)
        requests = [{"path": str(first), "start_line": 2, "end_line": 91},
                    {"path": str(second), "end_line": 100}]
        before = json.dumps(requests)
        pages = self.collect(requests)
        self.assertGreater(len(pages), 1)
        self.assertEqual(before, json.dumps(requests))
        self.assertTrue(any(page["remaining"] and page["remaining"]["following_requests"] == 1 for page in pages))

    def test_long_unicode_lines_and_json_escaping(self):
        text = ("汉🙂e\u0301\t\\\"\x00" * 1200) + "\r\n" + "last\\quoted\""
        path = self.write("long-中文.txt", text)
        pages = self.collect([{"path": str(path)}], max_bytes=1600)
        self.assertGreater(len(pages), 10)
        self.assertTrue(any(fragment.get("column", 1) > 1 for page in pages for source in page["files"]
                            for fragment in source["fragments"]))

    def test_all_line_endings_empty_file_and_missing_final_newline(self):
        values = ["", "\n", "alpha\r\nbeta\rgamma\ndelta\u2028epsilon\u0085zeta\veta\f", "last"]
        requests = [{"path": str(self.write(str(index), value))} for index, value in enumerate(values)]
        self.collect(requests)

    def test_crlf_near_page_boundaries_preserves_line_numbers(self):
        path = self.write("crlf", "A" * 5000)
        requests = [{"path": str(path)}]
        capacity = len(self.read(requests, max_bytes=1500)["files"][0]["fragments"][0]["text"])
        # Exercise both sides of the boundary without depending on temp path
        # length or cursor serialization details for an exact capacity.
        next_lines = set()
        for length in range(max(1, capacity - 20), capacity + 81, 5):
            path.write_bytes(("A" * length + "\r\n" + "B" * 3000).encode())
            page = self.read(requests, max_bytes=1500)
            next_lines.add(page["remaining"]["line"])
            self.collect(requests)
        self.assertEqual({1, 2}, next_lines)

    def test_page_and_cursor_replay_are_deterministic(self):
        path = self.write("source.txt", "data\\\" 中文\n" * 100)
        requests = [{"path": str(path)}]
        first = self.read(requests, max_bytes=1500)
        self.assertEqual(first, self.read(requests, max_bytes=1500))
        self.assertEqual(first, self.read(requests, cursor=first["cursor"], max_bytes=1500))
        second = self.read(requests, cursor=first["next_cursor"], max_bytes=1500)
        self.assertEqual(second, self.read(requests, cursor=first["next_cursor"], max_bytes=1500))
        # Page size is a delivery choice, not part of the source binding.
        wider = self.read(requests, cursor=first["next_cursor"], max_bytes=3000)
        self.assertEqual(second["files"][0]["fragments"][0].get("column", 1),
                         wider["files"][0]["fragments"][0].get("column", 1))

    def test_cursor_damage_in_valid_positions_is_distinct_from_changed_inputs(self):
        source = self.write("source", "first line\nsecond line\n" * 100)
        requests = [{"path": str(source)}, {"path": str(source)}]
        cursor = self.read(requests)["cursor"]
        wire = base64.urlsafe_b64decode(cursor)
        layout = struct.Struct(">B32sQQQ")
        version, binding, index, line, column = layout.unpack_from(wire)
        self.assertEqual((2, 0, 0, 0), (version, index, line, column))
        self.assertEqual(92, len(cursor))
        # Each changed coordinate still addresses selected source material.
        # Altering the binding or version must also be diagnosed as damage.
        for fields in [(version, binding, 1, line, column), (version, binding, index, 1, column),
                       (version, binding, index, line, 1), (version, bytes([binding[0] ^ 1]) + binding[1:], index, line, column),
                       (version + 1, binding, index, line, column)]:
            damaged = base64.urlsafe_b64encode(layout.pack(*fields) + wire[layout.size:]).decode()
            with self.subTest(fields=fields), self.assertRaises(reading.ReadingError) as failure:
                self.read(requests, cursor=damaged)
            self.assertEqual("invalid_cursor", failure.exception.code)
            self.assertIn("integrity", str(failure.exception))
        source.write_bytes(b"changed source\n")
        for token, code in [(damaged, "invalid_cursor"), (cursor, "stale_cursor")]:
            with self.subTest(code=code), self.assertRaises(reading.ReadingError) as failure:
                self.read(requests, cursor=token)
            self.assertEqual(code, failure.exception.code)

    def test_cursor_versions_require_a_new_batch_and_positions_remain_checked(self):
        source = self.write("source", "line one\nline two")
        requests = [{"path": str(source)}]
        page = self.read(requests)
        current = base64.urlsafe_b64decode(page["cursor"])
        layout = struct.Struct(">B32sQQQ")
        version, binding, index, line, column = layout.unpack_from(current)
        old = json.dumps([1, binding.hex(), index, line, column], separators=(",", ":")).encode()
        future = layout.pack(version + 1, binding, index, line, column)
        outside = layout.pack(version, binding, index, line, 1000000)
        for raw, code in [(old, "unsupported_cursor_version"),
                          (future + hashlib.sha256(future).digest()[:12], "unsupported_cursor_version"),
                          (outside + hashlib.sha256(outside).digest()[:12], "invalid_cursor")]:
            token = base64.urlsafe_b64encode(raw).decode().rstrip("=")
            with self.subTest(code=code), self.assertRaises(reading.ReadingError) as failure:
                self.read(requests, cursor=token)
            self.assertEqual(code, failure.exception.code)
            self.assertIn("new batch without --cursor", str(failure.exception))

    def test_changing_page_size_at_each_cursor_reconstructs_exact_text(self):
        original = "短行🙂\\\"\r\n" * 400 + "final"
        path = self.write("varying", original)
        requests, budgets = [{"path": str(path)}], (1024, 1700, 3200)
        cursor, text = None, []
        for index in range(100):
            page = self.read(requests, cursor=cursor, max_bytes=budgets[index % len(budgets)])
            if cursor:
                self.assertEqual(cursor, page["cursor"])
            text.extend(fragment["text"] for group in page["files"] for fragment in group["fragments"])
            if page["complete"]:
                break
            cursor = page["next_cursor"]
        else:
            self.fail("varying-size pages did not finish the fixture")
        self.assertGreaterEqual(index, len(budgets))
        self.assertEqual(original, "".join(text))

    def test_same_size_same_mtime_changes_reject_continuation(self):
        path = self.write("source.txt", "A" * 5000)
        requests = [{"path": str(path)}]
        page = self.read(requests, max_bytes=1500)
        old_stat = path.stat()
        path.write_bytes(b"B" * 5000)
        os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        with self.assertRaises(reading.ReadingError) as failure:
            self.read(requests, cursor=page["next_cursor"])
        self.assertEqual("stale_cursor", failure.exception.code)

    def test_changed_unread_file_and_request_order_reject_continuation(self):
        first = self.write("first", "A" * 5000)
        second = self.write("second", "B")
        requests = [{"path": str(first)}, {"path": str(second)}]
        page = self.read(requests, max_bytes=1500)
        with self.assertRaises(reading.ReadingError) as failure:
            self.read(list(reversed(requests)), cursor=page["next_cursor"])
        self.assertEqual("stale_cursor", failure.exception.code)
        second.write_bytes(b"C")
        with self.assertRaises(reading.ReadingError) as failure:
            self.read(requests, cursor=page["next_cursor"])
        self.assertEqual("stale_cursor", failure.exception.code)

    def test_file_change_during_read_is_not_returned_as_a_page(self):
        path = self.write("source", "before")
        real_fstat, calls = os.fstat, []

        def changing_fstat(fd):
            calls.append(fd)
            if len(calls) == 2:
                path.write_bytes(b"after!")
            return real_fstat(fd)

        with patch.object(reading.os, "fstat", side_effect=changing_fstat):
            with self.assertRaises(reading.ReadingError) as failure:
                self.read([{"path": str(path)}])
        self.assertEqual("file_changed", failure.exception.code)

    def test_outside_relative_missing_directory_and_symlink_paths_fail(self):
        outside = self.root / "outside"
        outside.write_bytes(b"unselected")
        symlink = self.sources / "escape"
        symlink.symlink_to(outside)
        loop = self.sources / "loop"
        loop.symlink_to(loop)
        names = [str(outside), "source.txt", str(self.sources / "missing"), str(self.sources), str(symlink), str(loop)]
        for name in names:
            with self.subTest(name=name), self.assertRaises(reading.ReadingError):
                self.read([{"path": name}])

    def test_invalid_input_encoding_range_cursor_and_limits_fail(self):
        path = self.write("source", "first\nsecond")
        requests = [{"path": str(path)}]
        for request in ([], {}, [{"path": str(path), "extra": True}],
                        [{"path": str(path), "start_line": True}],
                        [{"path": str(path), "start_line": 4}],
                        [{"path": str(path), "start_line": 2, "end_line": 1}]):
            with self.subTest(request=request), self.assertRaises(reading.ReadingError):
                self.read(request)
        for value in (0, 1023, 6001, True, "6000"):
            with self.subTest(max_bytes=value), self.assertRaises(reading.ReadingError):
                self.read(requests, max_bytes=value)
        for value in ("!broken", "a" * 257, 123, "", self.read(requests)["cursor"] + "="):
            with self.subTest(cursor=value), self.assertRaises(reading.ReadingError):
                self.read(requests, cursor=value)
        invalid = self.write("bytes", b"\xff")
        with self.assertRaises(reading.ReadingError) as failure:
            self.read([{"path": str(invalid)}])
        self.assertEqual("invalid_encoding", failure.exception.code)

    def test_empty_range_at_eof_is_explicit(self):
        path = self.write("source", "one\ntwo")
        page = self.read([{"path": str(path), "start_line": 3}])
        self.assertTrue(page["complete"])
        self.assertEqual([], page["files"][0]["fragments"])
        self.assertEqual(2, page["files"][0]["total_lines"])

    def test_find_returns_original_lines_for_case_sensitive_literals(self):
        source = self.write("notes", "NaN outside\r\nPRIVATE NaN repeated NaN\r\naaa\n"
                            "literal [a].* here\n小雪与NaN\u2028nan only\nother\nNaN outside")
        request = {"path": str(source), "start_line": 2, "end_line": 7, "find": ["[a].*", "NaN", "雪", "雪"]}
        page = self.read([request])
        group = page["files"][0]
        self.assertTrue(page["complete"])
        self.assertEqual([2, 4, 5], group["matching_lines"])
        self.assertEqual(3, group["total_matches"])
        self.assertNotIn("fragments", group)
        self.assertNotIn("PRIVATE", reading.serialize_page(page))
        # The exact text at a result is available through the same range API.
        evidence = self.read([{"path": str(source), "start_line": 5, "end_line": 5}])
        self.assertEqual([{"line": 5, "text": "小雪与NaN\u2028"}], evidence["files"][0]["fragments"])

    def test_find_pages_mixed_with_text_and_zero_matches(self):
        body = self.write("body", "first\nsecond\nthird")
        search = self.write("search", "".join(f"PRIVATE-{n} " + ("needle 雪\n" if n % 3 else "miss\n")
                                              for n in range(1, 1801)), artifact=True)
        requests = [{"path": str(body), "end_line": 1}, {"path": str(search), "find": ["needle", "雪"]},
                    {"path": str(search), "find": ["absent"]}, {"path": str(body), "start_line": 2}]
        pages = list(self.pages(requests, max_bytes=1500))
        self.assertGreater(len(pages), 2)
        groups = [[group for page in pages for group in page["files"] if group["request_index"] == index]
                  for index in range(len(requests))]
        expected = [line for line in range(1, 1801) if line % 3]
        self.assertEqual(expected, [line for group in groups[1] for line in group["matching_lines"]])
        self.assertTrue(all(group["total_matches"] == len(expected) for group in groups[1]))
        self.assertEqual([[]], [group["matching_lines"] for group in groups[2]])
        self.assertEqual(0, groups[2][0]["total_matches"])
        self.assertTrue(all("fragments" not in group for group in groups[1] + groups[2]))
        self.assertTrue(all("PRIVATE" not in reading.serialize_page(page) for page in pages))
        for index, text in [(0, "first\n"), (3, "second\nthird")]:
            self.assertEqual(text, "".join(frag["text"] for group in groups[index] for frag in group["fragments"]))
        for page, following in zip(pages, pages[1:]):
            remaining, first = page["remaining"], following["files"][0]
            self.assertEqual(first["request_index"], remaining["request_index"])
            if first.get("matching_lines"):
                self.assertEqual(first["matching_lines"][0], remaining["line"])

    def test_find_empty_file_and_range_each_report_zero(self):
        empty = self.write("empty", "")
        single = self.write("single", "wanted")
        requests = [{"path": str(empty), "find": ["wanted"]},
                    {"path": str(single), "start_line": 2, "find": ["wanted"]}]
        groups = [group for page in self.pages(requests) for group in page["files"]]
        self.assertEqual([0, 1], [group["request_index"] for group in groups])
        self.assertEqual([([], 0), ([], 0)], [(group["matching_lines"], group["total_matches"]) for group in groups])

    def test_find_cursor_binds_queries_and_all_file_contents(self):
        source = self.write("source", "needle A\nmiss\n" * 1500)
        other = self.write("other", "still selected\n")
        query = {"path": str(source), "find": ["needle", "A"]}
        requests = [query, {"path": str(other)}]
        page = self.read(requests, max_bytes=1500)
        self.assertFalse(page["complete"])
        self.assertEqual(page, self.read(requests, cursor=page["cursor"], max_bytes=1500))
        cursor = page["next_cursor"]
        for changed in ({**query, "find": ["A", "needle"]}, {**query, "find": ["needle"]},
                        {**query, "end_line": 2000}, {"path": str(source)}):
            with self.subTest(changed=changed), self.assertRaises(reading.ReadingError) as failure:
                self.read([changed, requests[1]], cursor=cursor)
            self.assertEqual("stale_cursor", failure.exception.code)
        # These edits preserve every matching line number. The byte content
        # still belongs to a different source version.
        initial = source.read_bytes()
        source.write_bytes(initial.replace(b"miss", b"omit"))
        with self.assertRaises(reading.ReadingError) as failure:
            self.read(requests, cursor=cursor)
        self.assertEqual("stale_cursor", failure.exception.code)
        source.write_bytes(initial)
        other.write_bytes(b"still selected!\n")
        with self.assertRaises(reading.ReadingError) as failure:
            self.read(requests, cursor=cursor)
        self.assertEqual("stale_cursor", failure.exception.code)

    def test_find_invalid_terms_and_outside_paths_fail(self):
        path = self.write("source", "line\n")
        for terms in (None, "word", [], [""], [True], ["one\ntwo"], ["end\r"], ["bad\u2028line"], ["\ud800"]):
            with self.subTest(terms=terms), self.assertRaises(reading.ReadingError) as failure:
                self.read([{"path": str(path), "find": terms}])
            self.assertEqual("invalid_request", failure.exception.code)
        outside = self.root / "outside"
        outside.write_bytes(b"word\n")
        with self.assertRaises(reading.ReadingError) as failure:
            self.read([{"path": str(outside), "find": ["word"]}])
        self.assertEqual("invalid_path", failure.exception.code)

    def test_large_batch_shares_one_budget_including_empty_files(self):
        requests = [{"path": str(self.write(f"file-{index}", "" if index % 2 else "selected\n"))}
                    for index in range(40)]
        pages = self.collect(requests, max_bytes=1500)
        self.assertGreater(len(pages), 10)

    def test_complete_lines_omit_only_default_metadata(self):
        path = self.write("source", "short line\n" * 40)
        page = self.read([{"path": str(path)}])
        self.assertTrue(page["complete"])
        self.assertEqual([{"line": line, "text": "short line\n"} for line in range(1, 41)],
                         page["files"][0]["fragments"])
        expanded = deepcopy(page)
        for fragment in expanded["files"][0]["fragments"]:
            fragment.update(column=1, ends_line=True)
        # Each complete line removes 28 JSON bytes plus four wrapper escapes.
        self.assertEqual(40 * 32, reading.output_size(expanded) - reading.output_size(page))

    def test_long_paths_do_not_repeat_in_remaining_metadata(self):
        short = self.write("short", "before\n")
        long = self.write("/".join(["directory" * 10] * 6) + "/source", "body" * 1000)
        requests = [{"path": str(short)}, {"path": str(long)}]
        pages = self.collect(requests, max_bytes=1500)
        self.assertGreater(len(pages), 2)
        page = next(page for page in pages if page["remaining"] and page["files"][0]["request_index"] == 1)
        repeated = deepcopy(page)
        repeated["remaining"]["path"] = str(long)
        self.assertGreater(reading.output_size(repeated), 1500)

    def test_source_metadata_that_cannot_fit_fails_without_cursor(self):
        path = self.write("/".join(["directory" * 12] * 6) + "/source", "body")
        requests = [{"path": str(path)}]
        with self.assertRaises(reading.ReadingError) as failure:
            self.read(requests, max_bytes=1024)
        self.assertEqual("page_too_small", failure.exception.code)
        self.collect(requests, max_bytes=6000)

    def test_direct_cli_from_unrelated_cwd_creates_no_files(self):
        path = self.write("source with spaces", "中文\\\"\r\n" * 100)
        command = [sys.executable, "-B", str(Path(reading.__file__).resolve()),
                   "--root", str(self.sources), "--root", str(self.artifacts),
                   "--requests", json.dumps([{"path": str(path)}]), "--max-bytes", "1500"]
        before = sorted(self.root.rglob("*"))
        result = subprocess.run(command, cwd=self.artifacts, capture_output=True, check=True,
                                env={**os.environ, "PYTHONIOENCODING": "ascii"})
        page = json.loads(result.stdout)
        self.assertFalse(page["complete"])
        self.assertEqual(b"", result.stderr)
        self.assertLessEqual(len(json.dumps(result.stdout.decode(), ensure_ascii=False).encode()), 1500)
        continuation = subprocess.run(command + ["--cursor", page["next_cursor"]], cwd=self.artifacts,
                                      capture_output=True, check=True)
        self.assertNotEqual(page["cursor"], json.loads(continuation.stdout)["cursor"])
        self.assertEqual(before, sorted(self.root.rglob("*")))
        failure = subprocess.run(command + ["--cursor", "invalid"], cwd=self.artifacts, capture_output=True)
        self.assertEqual(2, failure.returncode)
        self.assertEqual(b"", failure.stderr)
        self.assertEqual("invalid_cursor", json.loads(failure.stdout)["error"])


if __name__ == "__main__":
    unittest.main()
