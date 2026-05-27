# Evidence-Chain Readout Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| ECR-001 | M0 | Implement opt-in policy | ETv4 + `evidence_chain` | local | syntax/import | MUST | TODO | clean default must remain unchanged |
| ECR-002 | M1 | Retrieval gate | ETv4 clean vs evidence_chain | MuSiQue limit100 | exact/title R@5, all@5 | MUST | TODO | pool200, top5, Qwen32B clean index |
| ECR-003 | M2 | Mechanism audit | changed top5 rows | MuSiQue limit100 | strong-edge %, weak-only %, tail recovery | MUST | TODO | only after ECR-002 result |
| ECR-004 | M3 | Full retrieval | evidence_chain | MuSiQue full1000 | R@5/20/100/200, all@5 | GATED | TODO | only if ECR-002 positive |
| ECR-005 | M3 | Safety retrieval | evidence_chain | 2Wiki/HotpotQA full1000 | R@5/20/100/200, all@5 | GATED | TODO | only if MuSiQue full positive |
| ECR-006 | M4 | Reader QA | evidence_chain + GPT-4o-mini | gated datasets | EM/F1 | GATED | TODO | only after retrieval gate |
