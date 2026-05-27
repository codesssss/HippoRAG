#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv-hipporag/bin/python scripts/audit_agsto_clean_rrf.py \
  --datasets 2wikimultihopqa,hotpotqa,musique \
  --limit 100 \
  --pool_k 100 \
  --candidate_limit 120 \
  --proposal_candidate_depth 10 \
  --support_proposal_depth 6 \
  --beam_size 12 \
  --stable_anchor_k 2 \
  --max_endpoint_degree 30 \
  --support_weights 1.0,1.25,1.5,1.75,2.0,2.25 \
  --output_dir run_logs/agsto_clean_audit_default_limit100_20260505
