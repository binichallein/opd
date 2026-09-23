#!/usr/bin/env python3
"""Package bilingual LaTeX sources from an explicit anonymous-artifact allowlist."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


FORBIDDEN = ('/home/tyf', '/limx_embap', 'binichallein', '2928210292', 'Yaleon', 'yaofeng tu')
README = '''Block-Level Credit Assignment in On-Policy Distillation

Anonymous bilingual manuscript sources, using the official ICLR 2027 style.
main_en.tex is the English submission-format manuscript.
main_zh.tex is a Chinese reading companion, not a separate conference submission.

Compile from this directory using:
    tectonic main_en.tex
    tectonic main_zh.tex

The Chinese version requires static Noto Serif CJK SC and Noto Sans CJK SC fonts
from the official Noto CJK distribution. A XeLaTeX/BibTeX toolchain may also be used.
Keep the official style, margins and font sizes unchanged.

This package contains manuscript source, generated tables and figures only.
It does not include training code, private audit records, original model weights,
raw datasets or trajectories. A compilable source bundle is not a claim of a
complete anonymously released experimental-reproduction artifact.

All experimental claims, references, permissions and AI disclosure require author
review before submission. Packaging does not upload anything to OpenReview.
SOURCE_MANIFEST.json lists the SHA-256 of every included source asset.
'''


def asset_macro(text, name):
    values = re.findall(r'\\newcommand\{\\' + re.escape(name) + r'\}\{([^{}]+)\}', text)
    if len(values) != 1:
        raise ValueError(f'Expected one literal asset declaration: {name}')
    return values[0]


def local_dir(root, value):
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe asset path: {value}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f'Asset directory escapes root: {value}')
    return path


def checked_file(root, path):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe/missing source asset: {path}')
    return path.relative_to(root).as_posix()


def check_text(name, text):
    for value in FORBIDDEN:
        if value.lower() in text.lower():
            raise ValueError(f'Potential identifying content in {name}: {value}')


def collect_files(paper):
    asset_text = (paper / 'asset_paths.tex').read_text()
    result_dir = local_dir(paper, asset_macro(asset_text, 'qwenresults'))
    overlap_dir = local_dir(paper, asset_macro(asset_text, 'overlapresults'))
    figure_dirs = [local_dir(paper, asset_macro(asset_text, name))
                   for name in ('qwenfigures', 'qwendiagnostics')]
    figure_dirs += [paper / 'figures/history_20260921']
    if not result_dir.is_dir() or not overlap_dir.is_dir() or any(not p.is_dir() for p in figure_dirs):
        raise ValueError('Required final tables/figures are missing')
    paths = list(paper.glob('*.tex'))
    paths += [paper / name for name in ('references.bib', 'iclr2027_conference.sty',
                                       'iclr2027_conference.bst', 'natbib.sty', 'fancyhdr.sty')]
    paths += list((paper / 'generated').glob('history_*.tex'))
    paths += list(result_dir.glob('*.tex'))
    paths += list(overlap_dir.glob('*.tex'))
    paths += [result_dir / name for name in ('qwen4_results.csv', 'qwen4_paired_ci.csv')]
    paths += [paper / 'figures' / name for name in ('study_overview.png', 'credit_assignment_concept.png')]
    for directory in figure_dirs:
        paths += list(directory.glob('*.pdf'))
    if (paper / 'instruct_appendix_en.tex').exists():
        result = local_dir(paper, asset_macro(asset_text, 'qweninstructresults'))
        figures = local_dir(paper, asset_macro(asset_text, 'qweninstructfigures'))
        paths += list(result.glob('*.tex'))
        paths += [result / 'qwen17_results.csv', result / 'qwen17_summary.json']
        paths += [figures / f'qwen17_{metric}.pdf' for metric in ('avg_at_8', 'pass_at_8')]
    return {checked_file(paper, path): path for path in paths}


def package(paper, output):
    from check_iclr2027_paper import inspect_paper
    inspect_paper(paper, require_evaluation=True)
    files = collect_files(paper)
    if output.exists():
        raise FileExistsError(f'Existing package is immutable: {output}')
    payloads = {}
    for name, path in sorted(files.items()):
        value = path.read_bytes()
        if path.suffix in ('.tex', '.bib', '.sty', '.bst', '.json', '.csv'):
            check_text(name, value.decode('utf-8'))
        payloads[name] = value
    payloads['README.txt'] = README.encode()
    manifest = {'schema_version': 1, 'purpose': 'Anonymous bilingual LaTeX source package',
                'files': {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}}
    payloads['SOURCE_MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, value)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payloads):
            raise ValueError('Source package failed integrity validation')
    return {'files': len(payloads), 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.paper.resolve(), args.output.resolve()), indent=2))
