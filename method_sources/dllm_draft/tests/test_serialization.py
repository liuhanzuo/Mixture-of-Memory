from __future__ import annotations

import json
import textwrap
import unittest

from scaffold_coder.parser import parse_source
from scaffold_coder.renderer import render_module
from scaffold_coder.serialization import module_from_dict, module_to_dict


class SerializationTests(unittest.TestCase):
    def test_ir_json_round_trip(self) -> None:
        source = textwrap.dedent(
            """\
            def f(xs):
                for x in xs:
                    if x:
                        return x
                    else:
                        continue
                else:
                    return None
            """
        )
        original = parse_source(source)
        encoded = json.dumps(module_to_dict(original), sort_keys=True)
        restored = module_from_dict(json.loads(encoded))
        self.assertEqual(restored, original)
        self.assertEqual(render_module(restored), source)


if __name__ == "__main__":
    unittest.main()

