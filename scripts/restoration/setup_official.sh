#!/usr/bin/env bash
# Dependency and checksum setup only; no GPU work on the login node.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m pip install --no-index --find-links artifacts/official_wheels \
    -r configs/restoration/official_v1/requirements.txt
python scripts/restoration/fetch_official_models.py --offline
