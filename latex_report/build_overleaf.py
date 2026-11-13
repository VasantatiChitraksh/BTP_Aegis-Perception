#!/usr/bin/env python3
"""Package the reports for Overleaf, plus the complete archival source bundle.

Uses Python's standard library. Each upload ZIP has main.tex at its root and
contains every local TeX, bibliography, style and figure dependency. The full
source folders retain evidence, chart generators and validation records.
"""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
STAMP = (2026, 11, 13, 0, 0, 0)

# Flatten the existing workshop wrappers into one selectable root document.
# OT1 is the same font option used by the tested workshop.tex entry point.
WORKSHOP_MAIN = r"""% !TeX program = pdflatex
% Workshop-only entry; original body, bibliography and figures are unchanged.
\RequirePackage[OT1]{fontenc}
\documentclass[10pt,twocolumn,letterpaper]{article}
\def\AegisPaperOnly{1}
\usepackage[pagenumbers]{cvpr}
\input{preamble}
\def\confName{CVPR}
\def\confYear{2027}
\input{title}
\input{authors}
\begin{document}
\maketitle
\input{body}
\nobalance
{\small\bibliographystyle{ieeenat_fullname}\bibliography{references}}
\end{document}
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_zip(path, files):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert all(archive.read(name) == data for name, data in files.items())


def dependencies(folder, overrides):
    files = {}

    def add(name, extension=None):
        name = name.strip()
        if '#' in name:
            return  # Macro parameter, resolved by the concrete call sites.
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Non-portable dependency: ' + name)
        if extension and not rel.suffix:
            rel = rel.with_suffix(extension)
        name = rel.as_posix()
        if name in files:
            return
        data = overrides[name] if name in overrides else (folder / rel).read_bytes()
        files[name] = data
        if rel.suffix not in ('.tex', '.sty', '.cls'):
            return
        text = re.sub(r'(?<!\\)%[^\n]*', '', data.decode('utf-8'))
        for value in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            add(value, '.tex')
        for value in re.findall(r'\\(?:includegraphics|comparisonplot)(?:\[[^]]*\])?\{([^}]+)\}', text):
            add(value)
        for command, value in re.findall(r'\\(bibliography|bibliographystyle)\{([^}]+)\}', text):
            for part in value.split(','):
                add(part, '.bib' if command == 'bibliography' else '.bst')
        for values in re.findall(r'\\(?:usepackage|RequirePackage)(?:\[[^]]*\])?\{([^}]+)\}', text):
            for value in values.split(','):
                if (folder / (value.strip() + '.sty')).is_file():
                    add(value, '.sty')

    add('main.tex')
    return files


def package(folder_name, filename, compiler, overrides=None):
    overrides = overrides or {}
    files = dependencies(ROOT / folder_name, overrides)
    assert b'\\documentclass' in files['main.tex']
    source_files = {name: sha(data) for name, data in sorted(files.items())}
    files['OVERLEAF_README.txt'] = (
        f'{folder_name} — OVERLEAF SOURCE PROJECT\n\n'
        f'Upload this ZIP as a new Overleaf project.\n'
        f'Main document: main.tex\nCompiler: {compiler}\n'
        'All required sections, tables, figures, bibliography and local styles\n'
        'are included. Python and the original experiment repository are not\n'
        'required for compilation. Change the compiler in project settings.\n\n'
        'The complete editable project, evidence and chart-generation scripts\n'
        'are published in latex_report/ in the GitHub repository:\n'
        'https://github.com/VasantatiChitraksh/BTP_Aegis-Perception\n\n'
        + ('The workshop main.tex combines the paper-only wrappers, with the\n'
           'tested OT1 font option. All scientific sections and figures retain\n'
           'their original bytes. The supplied Workshop_Paper.pdf in GitHub\n'
           'remains the canonical PDF; compiler versions can change rendering.\n\n'
           if overrides else '')
        + 'Overleaf upload instructions:\n'
        'https://docs.overleaf.com/managing-projects-and-files/uploading-a-project\n'
    ).encode('utf-8')
    files['package_manifest.json'] = (json.dumps({
        'project': folder_name, 'main_document': 'main.tex', 'compiler': compiler,
        'generated_entries': list(overrides), 'source_sha256': source_files,
    }, indent=2) + '\n').encode()
    assert len(files) <= 180
    assert max(map(len, files.values())) < 50_000_000
    target = ROOT / filename
    write_zip(target, files)
    return {'archive': filename, 'project': folder_name, 'main_document': 'main.tex',
            'compiler': compiler, 'files': len(files), 'bytes': target.stat().st_size,
            'sha256': sha(target.read_bytes()), 'source_sha256': source_files}


def main():
    packages = [
        package('BTP_REPORT', 'BTP_REPORT_Overleaf.zip', 'XeLaTeX'),
        package('WORKSHOP PAPER', 'WORKSHOP_PAPER_Overleaf.zip', 'pdfLaTeX',
                {'main.tex': WORKSHOP_MAIN.encode()}),
        package('BTP_REPORT_EXCALIDRAW', 'BTP_REPORT_Excalidraw_Overleaf.zip', 'XeLaTeX'),
    ]
    record = {'generator': 'build_overleaf.py', 'packages': packages,
              'scope': 'Compilation sources; full evidence and generators remain in the source folders.'}
    (ROOT / 'overleaf_packages.json').write_text(json.dumps(record, indent=2) + '\n')
    archive_files = {}
    for folder in ('BTP_REPORT', 'WORKSHOP PAPER', 'BTP_REPORT_EXCALIDRAW'):
        for path in (ROOT / folder).rglob('*'):
            rel = path.relative_to(ROOT)
            if not path.is_file() or any(p in ('.build', '__pycache__') for p in rel.parts):
                continue
            if path.name == 'Workshop_Recompiled.pdf' or path.suffix in ('.zip', '.pyc', '.log'):
                continue
            archive_files[rel.as_posix()] = path.read_bytes()
    for name in ('README.md', 'README.txt', 'build_overleaf.py', 'overleaf_packages.json'):
        archive_files[name] = (ROOT / name).read_bytes()
    if (ROOT / 'overleaf_validation.json').exists():
        archive_files['overleaf_validation.json'] = (ROOT / 'overleaf_validation.json').read_bytes()
    write_zip(ROOT / 'LaTeX_Projects.zip', archive_files)
    for value in packages:
        print(f"{value['archive']}: {value['files']} files; {value['bytes'] / 1024**2:.2f} MiB; {value['compiler']}")
    print(f'LaTeX_Projects.zip: {len(archive_files)} files; complete archival sources')


if __name__ == '__main__':
    main()
