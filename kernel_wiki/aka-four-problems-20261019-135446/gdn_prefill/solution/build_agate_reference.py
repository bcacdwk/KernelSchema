"""Build an Agate-safe input generator without changing input semantics."""

from __future__ import annotations

from pathlib import Path

import python_minifier


ROOT = Path(__file__).resolve().parent
PROBLEM = ROOT.parent
DEST = ROOT / "agate_reference" / "input.py"

reference = (PROBLEM / "reference.py").read_text(encoding="utf-8")
input_source = (PROBLEM / "input.py").read_text(encoding="utf-8")

# The continuation-state generator needs the official reference functions but
# Agate intentionally blocks importlib/pathlib in input.py. Inline only the
# function section; reference.py itself remains byte-identical in the bundle.
reference_functions = reference.split("\nclass Model(", 1)[0]
input_source = input_source.replace("import importlib.util\n", "")
input_source = input_source.replace("from pathlib import Path\n", "")
input_source = input_source.replace(
    '''            ref_path = Path(__file__).with_name("reference.py")
            spec = importlib.util.spec_from_file_location("_gdn_prefix_reference", ref_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _, state_kv = module.blockwise_delta_rule(
''',
    '''            _, state_kv = blockwise_delta_rule(
''',
)

if "importlib" in input_source or "module.blockwise_delta_rule" in input_source:
    raise RuntimeError("unexpected original input.py structure")

combined = reference_functions + "\n\n" + input_source
minified = python_minifier.minify(
    combined,
    remove_annotations=False,
    remove_literal_statements=True,
)
DEST.write_text(minified + "\n", encoding="utf-8")
