# Aegis Perception — LaTeX reports

The complete editable BTP and workshop projects are published separately here.
The BTP report has 25 pages and 29 numbered figures; the supplied workshop PDF
has eight pages.

| Document | Upload-ready source ZIP | Overleaf settings | PDF |
|---|---|---|---|
| BTP report | [BTP_REPORT_Overleaf.zip](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/raw/refs/heads/main/latex_report/BTP_REPORT_Overleaf.zip) | `main.tex`, **XeLaTeX** | [BTP_Report.pdf](BTP_REPORT/BTP_Report.pdf) |
| BTP — Excalidraw-style flowcharts | [BTP_REPORT_Excalidraw_Overleaf.zip](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/raw/refs/heads/main/latex_report/BTP_REPORT_Excalidraw_Overleaf.zip) | `main.tex`, **XeLaTeX** | [BTP_Report.pdf](BTP_REPORT_EXCALIDRAW/BTP_Report.pdf) |
| Workshop paper | [WORKSHOP_PAPER_Overleaf.zip](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/raw/refs/heads/main/latex_report/WORKSHOP_PAPER_Overleaf.zip) | `main.tex`, **pdfLaTeX** | [Workshop_Paper.pdf](WORKSHOP%20PAPER/Workshop_Paper.pdf) |

The Excalidraw-style version changes only six diagram areas. It retains the
25-page report's content, evidence, photographs, results, tables and page order.
The original BTP project remains unchanged.

## Use in Overleaf

1. Download the ZIP for the document you want to edit.
2. In Overleaf, choose **New project → Existing project (.zip)** and upload it.
3. Open **Settings → Compiler**, select the compiler above, and set the main
   document to `main.tex`.
4. Recompile, then edit the sections, tables, bibliography or figure files.

Each upload ZIP contains all compilation dependencies, with `main.tex` at its
root. No Python scripts, model weights or experiment data are required to compile.
The workshop upload entry combines its existing paper-only wrappers; its body,
bibliography and figures retain their original bytes. The supplied workshop PDF
remains the canonical version; compiler versions can change rebuilt typography.

Official instructions: [uploading a project](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project),
[main document](https://docs.overleaf.com/getting-started/recompiling-your-project/the-main-document),
[compiler settings](https://docs.overleaf.com/getting-started/recompiling-your-project/selecting-a-tex-live-version-and-latex-compiler).

## Complete source folders

- [BTP_REPORT/](BTP_REPORT/) includes the LaTeX, figures, bibliography, evidence,
  chart generators, MTP reference, build scripts and validation records.
- [BTP_REPORT_EXCALIDRAW/](BTP_REPORT_EXCALIDRAW/) is an independent copy with
  sketch-style vector flowcharts, their generator and the licensed Virgil font.
- [WORKSHOP PAPER/](WORKSHOP%20PAPER/) includes the original workshop sources,
  styles, figures, bibliography, preserved PDF and build instructions.
- [LaTeX_Projects.zip](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/raw/refs/heads/main/latex_report/LaTeX_Projects.zip)
  is the complete archival bundle of all three folders. For a direct Overleaf upload,
  use the individual ZIPs above.

## Rebuild the upload ZIPs after editing

From the repository root, run:

```sh
python3 latex_report/build_overleaf.py
```

The packager checks every local compilation dependency and writes deterministic
ZIPs plus `overleaf_packages.json` with per-file SHA-256 hashes. Each individual
ZIP contains fewer than 180 files. It also refreshes the complete archival ZIP.
To regenerate the report PDFs locally, follow the `README.txt` in each folder.

The ZIP extraction and local compilation checks are recorded in
[`overleaf_validation.json`](overleaf_validation.json). They use Tectonic's
XeTeX engine; compilation inside an authenticated Overleaf account is not automated.
