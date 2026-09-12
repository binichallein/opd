import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_reader(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_jsonl


@pytest.mark.parametrize("script", ["regrade_opd_eval_external", "compare_paired_opd_evals"])
@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_jsonl_preserves_unicode_separators_and_source_bytes(tmp_path, script, separator, newline):
    rows = [{"response": f"before{separator}after", "answer": "142.0"},
            {"response": "\\boxed{142}", "answer": "142.0"}]
    path = tmp_path / "outputs.jsonl"
    path.write_bytes((newline.join(json.dumps(row, ensure_ascii=False) for row in rows)
                      + newline).encode("utf-8"))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_reader(script)(path) == rows
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("script", ["regrade_opd_eval_external", "compare_paired_opd_evals"])
def test_jsonl_still_rejects_genuinely_truncated_record(tmp_path, script):
    path = tmp_path / "outputs.jsonl"
    path.write_text('{"response":"unfinished', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_reader(script)(path)
