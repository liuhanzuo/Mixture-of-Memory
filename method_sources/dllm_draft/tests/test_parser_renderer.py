from __future__ import annotations

import ast
import textwrap
import unittest

from scaffold_coder.errors import UnsupportedSyntaxError
from scaffold_coder.parser import normalize_source, parse_source
from scaffold_coder.renderer import SpanKind, render_with_source_map


COUNT_PAIRS = '''
def count_pairs(nums, target):
    """Count pairs summing to target."""
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                count += 1
    return count
'''


class ParserRendererTests(unittest.TestCase):
    def test_count_pairs_round_trip_and_doc_strip(self) -> None:
        normalized = normalize_source(textwrap.dedent(COUNT_PAIRS))
        expected = textwrap.dedent(
            """\
            def count_pairs(nums, target):
                count = 0
                for i in range(len(nums)):
                    for j in range(i + 1, len(nums)):
                        if nums[i] + nums[j] == target:
                            count += 1
                return count
            """
        )
        self.assertEqual(normalized, expected)
        self.assertEqual(normalize_source(normalized), normalized)
        ast.parse(normalized)

    def test_if_elif_else_and_loop_else(self) -> None:
        source = textwrap.dedent(
            """\
            def classify(xs):
                for x in xs:
                    if x > 0:
                        return 1
                    elif x == 0:
                        return 0
                    else:
                        continue
                else:
                    return -1
            """
        )
        normalized = normalize_source(source)
        self.assertEqual(normalized, source)
        ast.parse(normalized)

    def test_docstring_only_body_becomes_pass(self) -> None:
        source = 'def f():\n    """documentation only"""\n'
        self.assertEqual(normalize_source(source), "def f():\n    pass\n")

    def test_source_map_covers_every_character(self) -> None:
        module = parse_source("def f(x):\n    return x + 1\n")
        rendered = render_with_source_map(module)
        rendered.validate()
        self.assertEqual(rendered.text, "def f(x):\n    return x + 1\n")
        self.assertTrue(any(span.kind is SpanKind.CONTENT for span in rendered.spans))
        self.assertEqual(rendered.spans[0].start, 0)
        self.assertEqual(rendered.spans[-1].end, len(rendered.text))

    def test_top_level_spacing_is_canonical(self) -> None:
        source = "import math\n\ndef f():\n    return math.pi\n\ndef g():\n    return 2\n"
        expected = (
            "import math\n\n\ndef f():\n    return math.pi\n\n\ndef g():\n    return 2\n"
        )
        self.assertEqual(normalize_source(source), expected)

    def test_identifiers_named_like_compound_keywords_are_simple(self) -> None:
        source = (
            "import re\n"
            "\n"
            "def f(s, p):\n"
            "    match = re.fullmatch(p, s)\n"
            "    return match is not None\n"
        )
        normalized = normalize_source(source)
        self.assertIn("match = re.fullmatch(p, s)", normalized)
        ast.parse(normalized)

    def test_rejects_v0_unsupported_constructs(self) -> None:
        cases = [
            "@dec\ndef f():\n    pass\n",
            "class C:\n    pass\n",
            "try:\n    x = 1\nexcept Exception:\n    pass\n",
            "with open('x') as f:\n    pass\n",
            "async def f():\n    pass\n",
        ]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(
                UnsupportedSyntaxError
            ):
                parse_source(source)


if __name__ == "__main__":
    unittest.main()
