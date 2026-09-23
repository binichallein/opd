import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/package_iclr2027_paper.py'
    spec = importlib.util.spec_from_file_location('paper_package', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_declared_paths_cannot_escape_package(tmp_path):
    m = module()
    with pytest.raises(ValueError):
        m.local_dir(tmp_path, '../internal')
    with pytest.raises(ValueError):
        m.local_dir(tmp_path, '/tmp/secret')


def test_macro_requires_exactly_one_simple_declaration():
    m = module()
    assert m.asset_macro(r'\newcommand{\qwenresults}{generated/final}', 'qwenresults') == 'generated/final'
    with pytest.raises(ValueError):
        m.asset_macro('', 'qwenresults')
    with pytest.raises(ValueError):
        m.asset_macro(r'\newcommand{\qwenresults}{x}\newcommand{\qwenresults}{y}', 'qwenresults')


def test_anonymity_scan_rejects_identifying_plaintext():
    m = module()
    with pytest.raises(ValueError):
        m.check_text('main.tex', 'source /limx_embap/tos/user/Yaleon/checkpoint')
    m.check_text('main.tex', r'Anonymous authors. $\log p_\theta$')


def test_symlink_rejected(tmp_path):
    m = module()
    source = tmp_path / 'source.tex'
    source.write_text('Anonymous')
    (tmp_path / 'linked.tex').symlink_to(source)
    with pytest.raises(ValueError):
        m.checked_file(tmp_path, tmp_path / 'linked.tex')


def test_actual_package_includes_instruct_tables_and_figures_not_internal_paths():
    m = module()
    paper = Path(__file__).resolve().parents[1] / 'paper/iclr2027'
    files = m.collect_files(paper)
    assert 'generated/qwen17_instruct_20260923/qwen17_step200_en.tex' in files
    assert 'generated/qwen17_instruct_20260923/qwen17_pass_at_8_zh.tex' in files
    assert 'figures/qwen17_instruct_20260923/qwen17_avg_at_8.pdf' in files
    assert not any('internal/' in name or name.endswith('.jsonl') for name in files)
