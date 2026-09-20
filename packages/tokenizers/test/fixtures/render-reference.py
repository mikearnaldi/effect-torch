"""Regenerate exact rendered strings with Jinja2, independently of MiniJinja.

Run from the workspace root:
    uv run --with jinja2==3.1.6 python packages/tokenizers/test/fixtures/render-reference.py
"""

import hashlib
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

fixtures = Path(__file__).parent
environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)


def raise_exception(message):
    raise ValueError(message)


environment.globals["raise_exception"] = raise_exception

# Method tests use compact JSON from MiniJinja, so compare JSON values for these.
for case in json.loads((fixtures / "python-methods.json").read_text()):
    actual = environment.from_string(case["template"]).render(**case["context"])
    expected = case["expected"]
    if "tojson" in case["template"]:
        assert json.loads(actual) == json.loads(expected), case["name"]
    else:
        assert actual == expected, case["name"]

source = (fixtures / "diffusiongemma/chat_template.jinja").read_bytes()
assert hashlib.sha256(source).hexdigest() == (
    "9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf"
)
template = environment.from_string(source.decode())
case_path = fixtures / "diffusiongemma/cases.json"
cases = json.loads(case_path.read_text())
for case in cases:
    case["expected"] = template.render(**case["context"])
case_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
print(f"Checked 4 Python-method cases and rendered {len(cases)} DiffusionGemma cases.")
