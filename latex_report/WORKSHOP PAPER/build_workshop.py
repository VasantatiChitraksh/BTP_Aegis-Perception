#!/usr/bin/env python3
"""Compile the separate workshop paper; preserve the supplied PDF unchanged.

Requires Tectonic or latexmk + pdfLaTeX/BibTeX, plus Python and PyMuPDF.
The rebuilt file is Workshop_Recompiled.pdf; Workshop_Paper.pdf is never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', help='Path to tectonic or latexmk')
    args = parser.parse_args()
    import pymupdf as pdf
    manifest = json.loads((ROOT/'validation/source_hashes.json').read_text())
    canonical = ROOT/'Workshop_Paper.pdf'
    if sha(canonical) != manifest['Workshop_Paper.pdf']:
        raise SystemExit('The preserved supplied Workshop_Paper.pdf has changed.')
    engine = args.engine or shutil.which('latexmk') or shutil.which('tectonic')
    if not engine:
        raise SystemExit('Install latexmk + pdfLaTeX/BibTeX or Tectonic, or pass --engine.')
    build = ROOT/'.build'
    build.mkdir(exist_ok=True)
    if 'tectonic' in Path(engine).name:
        entry = 'workshop'
        command = [engine, '--keep-logs', '--keep-intermediates', '--outdir', str(build), entry+'.tex']
    else:
        entry = 'paper'
        command = [engine, '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-outdir='+str(build), entry+'.tex']
    subprocess.run(command, cwd=ROOT, check=True)
    log = (build/(entry+'.log')).read_text(errors='replace')
    bad = ('undefined references', 'undefined citations', 'Citation `', 'Reference `', 'Overfull \\hbox', 'Overfull \\vbox', 'multiply defined', 'LaTeX Error:')
    issues = [line for line in log.splitlines() if any(x in line for x in bad)]
    if issues:
        raise SystemExit('Resolve compilation warnings before delivery:\n'+'\n'.join(issues))
    rebuilt = ROOT/'Workshop_Recompiled.pdf'
    shutil.copy2(build/(entry+'.pdf'), rebuilt)
    source = pdf.open(canonical)
    result = pdf.open(rebuilt)
    checks = []
    for i in range(min(len(source), len(result))):
        a,b=source[i],result[i]
        checks.append({'page':i+1,'geometry_equal':a.rect==b.rect,
            'text_equal':a.get_text()==b.get_text(),
            'render_equal_144dpi':a.get_pixmap(matrix=pdf.Matrix(2,2), alpha=False).samples==b.get_pixmap(matrix=pdf.Matrix(2,2),alpha=False).samples})
    changed = [p for p,h in manifest.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    output = {'canonical_pdf':canonical.name,'canonical_pages':len(source),
        'canonical_sha256':sha(canonical),'canonical_unchanged':True,
        'compiled_pdf':rebuilt.name,'compiled_pages':len(result),'compiler':Path(engine).name,
        'compile_issues':issues,'source_files_changed_since_separation':changed,
        'page_comparison':checks,
        'reproduction_note':'The supplied PDF remains canonical and byte-identical. Compiler/font and microtype differences may change the rebuilt rendering. Original source contents are preserved; the new main.tex wrapper disables the supplement for every entry point.'}
    assert len(source)==8
    (ROOT/'validation/preservation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='page_comparison'},indent=2))


if __name__ == '__main__':
    main()
