TWO INDEPENDENT LATEX PROJECTS

BTP_REPORT/
  BTP_Report.pdf       Standalone 25-page BTP report; no workshop pages.
  main.tex             Editable BTP entry point (XeLaTeX or Tectonic).
  build_report.py      Builds only the BTP report.
  reference/           Unchanged MTP formatting reference.

WORKSHOP PAPER/
  Workshop_Paper.pdf   Exact unchanged, separately supplied eight-page paper.
  paper.tex            Editable workshop entry point (pdfLaTeX).
  build_workshop.py    Rebuilds only the workshop, separately from the BTP.

Each folder includes its own sections, figures, bibliography and build guide.
The BTP cover uses the supplied hardware photos and the confirmed student/guide details.
The downloadable LaTeX_Projects.zip contains these two folders and their PDFs.
For Overleaf, upload BTP_REPORT_Overleaf.zip (XeLaTeX) or
WORKSHOP_PAPER_Overleaf.zip (pdfLaTeX); both use main.tex at the ZIP root.
See README.md for direct downloads and complete upload instructions.

The previous flat folder and combined report have been relocated intact to
../artifacts/report_backups/pre_split_flat_files_20261113/ and backed up in
latex_report_before_folder_split_20261113.tar.gz in the same backup directory.
No experiment data, model weights or source code outside this report area was
modified. The workshop's original PDF is preserved even though rebuilding with
a different compiler can change typography and pagination within its 8 pages.
