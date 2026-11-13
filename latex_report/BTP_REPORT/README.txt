AEGIS PERCEPTION — STANDALONE BTP REPORT

OPEN BTP_Report.pdf — the independent 25-page report, with 29 numbered figures
and an unlabelled cover diagram, without workshop pages.
EDIT main.tex — the BTP entry point, with no sibling-folder dependencies.
The separate workshop project lives in ../WORKSHOP PAPER/.
The current page count and validation results are recorded in validation/.
Page 3 contains the black-labelled weather comparison and principal results;
page 4 contains the Gantt and contributions; page 5 introduces the workflow.
Every main-body page (physical pages 3–17) contains figures or tables.
Thin neutral frames group comparisons; important results use bold type.
Repetition was condensed and the mostly empty closing page was removed.

COVER AND REPORT STRUCTURE
The enlarged, unlabelled cover pipeline uses saved input/restored examples and
schematic network/chip icons. It contains no status text or invented detections.
Chitraksh — CS23B054
Sudhanva — CS23B051
Rishikesh — CS23B015
Guide: Dr Vishnu Chalavadi
The report body presents six completed experimental phases and the one ongoing
RAMiT + YOLOv12 edge-deployment exploration. The supplementary section begins
with the explicit heading "Supplementary Material" and uses S-prefixed pages,
figures and tables. Dataset-chart headings and legends have separate spacing.

BUILD
Python with PyMuPDF, plus Tectonic or latexmk with XeLaTeX/BibTeX:
  python3 -m pip install pymupdf
  python3 build_report.py
  python3 verify_report.py
To choose the compiler:
  python3 build_report.py --engine /path/to/tectonic
Overleaf: use main.tex with XeLaTeX.
The script compiles in .build/ and writes BTP_Report.pdf. It checks the
20–30-page target, A4 geometry, references and immutable input hashes.
It never joins PDFs or reads any workshop file. Inspect validation/ for the
checks performed on the current build; regenerating figures requires rebuilding
and verifying the PDF before those records describe the updated document.

LAYOUT
main.tex, preamble.tex, frontmatter.tex: entry, MTP-style design and cover.
sections/, tables/: modular BTP content and two-decimal result tables.
sections/11_contributions.tex is included early, immediately after the overview.
Results pages use full-width plots/tables with compact two-column explanation.
figures/: plots, diagrams and genuine saved experimental imagery.
figures/comparison_inputs/: compact, checksummed overlays and per-panel results.
evidence/: full-precision numerical snapshots, input hashes and provenance.
scripts/: reproducible chart, gallery and cover-diagram builders.
reference/mtech_ref_report.pdf: unchanged formatting reference.
validation/: numerical, compilation, preservation and visual-review records.

OPTIONAL VISUAL REGENERATION
Install matplotlib, numpy, Pillow and PyMuPDF, then run from this folder:
  python3 scripts/build_project_visuals.py
  python3 scripts/build_results_visuals.py
  python3 scripts/build_cover.py
  python3 scripts/build_work_timeline.py
  python3 scripts/build_comparison_gallery.py
  python3 scripts/build_neutral_comparison.py
  python3 scripts/build_chart_layout_repairs.py
  python3 scripts/build_distribution_charts.py
The included figure PDFs compile without regeneration. These commands use
bundled evidence and images; they do not train models or run new inference.
For a deliberate refresh of histogram snapshots from the original repository's
full observations, use:
  python3 scripts/build_distribution_charts.py --refresh-from-repo
That optional refresh requires the original docs/ and artifacts/ inputs and is
not needed to rebuild this standalone project.

TIMELINE AND WORK ALLOCATION
The restrained full-width Gantt follows the MTP's dated-record presentation. Its bars
connect recorded milestones; they do not measure continuous attendance or
person-hours. The early axis is explicitly condensed before 7 October.
The ongoing phase has a current-status marker and no invented completion date.
An editable LaTeX table maps the six completed phases to individual and
shared contributions on page 4.
Chitraksh's custom quantizer and pruning tooling are supported by the named
edge handoff and commits; Sudhanva's training and runtime integration are
supported by the report and code history. Rishikesh's evaluation/device-analysis
role is retained from the prior contribution record. Joint review is explicit;
no exclusive ownership or effort percentages are inferred. See
  evidence/work_timeline_provenance.json
for dated sources, attribution scope and the current-phase status.

COMPARISON PRESENTATION
Weather and detector-gallery row/column labels are black. Thin rules group
matched comparisons; plotted series retain their quantitative colour encoding.
The weather-grid derivative retains all 16 embedded image streams unchanged.
The detection chart separates its axis title and legend; its plotted region is
unchanged. A stale appendix letter in the export chart now points to Appendix C.
Original figure PDFs remain protected. See evidence/neutral_comparison.json and
evidence/chart_layout_repairs.json.

COMPARISON GALLERIES: SCIENTIFIC SCOPE
Two galleries show 24 saved panels: two labelled scenes, four weather variants
and three processing paths. Each row compares source-resolution YOLOv8n,
128 x 128 resize-only plus YOLOv8n, and 128 x 128 Restormer plus YOLOv8n.
The 128-square size is the resize/restoration canvas; detector letterboxing
remains identical at 640 x 640. Weather is applied once at source dimensions
before branching. The gallery builder retains the saved detection boxes and
scores and adds per-panel TP/FP/FN counts from the recorded matching results.
The vehicle gallery uses the same display crop in every panel; the people
gallery retains its full field of view. Full counts belong to the saved images.
These are local CPU FP32 diagnostics from two scenes containing five annotated
objects, repeated under dependent weather variants. They are not 24 independent
scenes, a new mAP benchmark, or Nano throughput evidence. See
  figures/comparison_inputs/provenance.json
  evidence/comparison_galleries.json
for hashes, geometry, runtime, thresholds and crop details.

HISTOGRAMS: SCIENTIFIC SCOPE
Restoration histograms compare per-image PSNR gains of Restormer and RAMiT on
the same held-out images, with 2 dB bins and separate weather panels. Each image
has equal histogram weight; this differs from the background/weather-balanced
aggregation used for headline benchmark scores.
Nano latency histograms summarize synchronized per-frame pipeline times from
saved labelled replay runs. Warm-up frames are excluded. The raw and restored
runs have different measured-frame counts; percentages are normalized within
each run and the two processing paths use different displayed x-axis ranges.
Repeated replay observations are not independent test images. The figures show
distribution shape and observed variability, without new hardware measurements.
See evidence/restoration_gain_histogram.json and evidence/nano_latency_histogram.json.

EVIDENCE AND PRESERVATION
Dataset/model comparisons, continuation, detection, geometry controls, resource
use, throughput and energy derive from project records in docs/ and artifacts/.
Numerical evidence remains at full precision; displayed supplementary results
use at most two decimals, with labelled scaled units for SSIM and small errors.
The MTP reference, workshop source/PDF and original experiment evidence remain
preserved. The previous flat LaTeX folder and combined edition are archived in
../../artifacts/report_backups/. The immutable input manifest protects reused
reference/evidence assets; editable BTP prose and new figure generators remain
maintainable. No retraining or fresh device benchmark was performed for layout
or figure preparation.

LIMITS
RAMiT + YOLOv12 integration remains ongoing, without measured performance claims.
Six completed experimental work packages include negative/rejected results;
completion does not establish detector improvement or accepted acceleration.
Recorded limitations include single training seeds, limited smoke backgrounds,
short board trials, and incomplete same-runtime resize-only/camera-ground-truth
controls. Institutional certificate/declaration wording has not been supplied
and has not been invented.
