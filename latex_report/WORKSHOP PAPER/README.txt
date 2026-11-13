AEGIS PERCEPTION — SEPARATE WORKSHOP PAPER

Workshop_Paper.pdf is the exact, unchanged eight-page workshop PDF supplied by
the authors. It is independent of the BTP report. No BTP or technical-supplement
pages are included here.

EDIT AND COMPILE
- Open this folder as its own Overleaf project. Select paper.tex and pdfLaTeX.
- main.tex also compiles only the workshop; its wrapper disables the supplement.
- Local build with latexmk (pdfLaTeX) or Tectonic:
    python3 build_workshop.py
  Requires PyMuPDF: python3 -m pip install pymupdf
  To choose an engine: python3 build_workshop.py --engine /path/to/tectonic
- Tectonic uses workshop.tex, which selects CVPR's legacy font encoding.
- The build writes Workshop_Recompiled.pdf and validation/preservation.json.
  It never overwrites Workshop_Paper.pdf.

CONTENTS AND PRESERVATION
paper.tex, body.tex, preamble.tex, authors.tex, title.tex, sections/, figures/,
references.bib and the CVPR style files retain their original bytes. The shared
original main.tex is preserved at original/main.tex; the new main.tex wrapper
always sets paper-only mode before reading it. No appendix inputs are active.
The eight paper sections and ten figure assets are the complete used dependency
set. Unused report/supplement material is excluded.

validation/source_hashes.json records the original-file hashes at separation.
Source edits are reported by the build without preventing normal editing. The
canonical supplied PDF is protected against accidental replacement.

The available Tectonic compiler successfully rebuilds eight pages with no
unresolved references, missing fonts, or overfull boxes. Rendering differs from
the supplied PDF because of compiler/microtype layout differences. Therefore,
Workshop_Paper.pdf remains the canonical unchanged eight-page deliverable.
