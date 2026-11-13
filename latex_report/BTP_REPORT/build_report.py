#!/usr/bin/env python3
"""Compile the standalone BTP report. Never imports or appends workshop pages."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',help='Path to tectonic or latexmk (XeLaTeX)')
    args=parser.parse_args()
    import pymupdf as pdf
    manifest=json.loads((ROOT/'evidence/immutable_inputs.json').read_text())
    for p,h in manifest.items():
        if sha(ROOT/p)!=h:raise SystemExit('Saved evidence/reference asset changed: '+p)
    engine=args.engine or shutil.which('tectonic') or shutil.which('latexmk')
    if not engine:raise SystemExit('Install Tectonic or latexmk + XeLaTeX, or pass --engine.')
    build=ROOT/'.build';build.mkdir(exist_ok=True)
    if 'tectonic' in Path(engine).name:
        command=[engine,'--keep-logs','--keep-intermediates','--outdir',str(build),'main.tex']
    else:
        command=[engine,'-xelatex','-interaction=nonstopmode','-halt-on-error','-outdir='+str(build),'main.tex']
    subprocess.run(command,cwd=ROOT,check=True)
    log=(build/'main.log').read_text(errors='replace')
    forbidden=('undefined references','undefined citations','Citation `','Reference `','Overfull \\hbox','Overfull \\vbox','multiply defined','LaTeX Error:','Missing character:')
    issues=[line for line in log.splitlines() if any(v in line for v in forbidden)]
    if issues:raise SystemExit('Resolve compilation issues:\n'+'\n'.join(issues))
    doc=pdf.open(build/'main.pdf')
    assert 20<=len(doc)<=30, f'Expected standalone BTP within 20–30 pages; got {len(doc)}'
    assert all(abs(p.rect.width-595.276)<.1 and abs(p.rect.height-841.89)<.1 for p in doc)
    target=ROOT/'BTP_Report.pdf';shutil.copy2(build/'main.pdf',target)
    record={'pdf':target.name,'pages':len(doc),'standalone':True,'workshop_pages_included':0,'all_pages_a4':True,'compile_issues':issues,'immutable_inputs_verified':len(manifest),'underfull_hbox_notices':log.count('Underfull \\hbox'),'pdf_sha256':sha(target),'scope':'Document build only; no new model or device experiments.'}
    (ROOT/'validation/build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
