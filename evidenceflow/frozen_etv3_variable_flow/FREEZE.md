# Frozen ETv3 Variable-Flow Snapshot

Freeze date: 2026-05-10

This directory is a package-local snapshot of:

```text
evidence_transition_graphragv3_variable_flow/
```

It exists so legacy/fresh PCEC parity runs can execute without depending on a
moving ETv3 development branch.  It is not the upstream retriever for the main
Table-1 EvidenceFlow results.

## Boundary

This snapshot owns the ETv3 expander side of PCEC:

```text
query -> ET variable-flow candidate expansion -> ordered candidate pool
```

PCEC owns the readout side:

```text
ordered candidate pool -> prefix-residual constrained composition -> final top5
```

The frozen package intentionally keeps the original ETv3 method strings in
`contract.py` for report compatibility, but all internal Python imports have
been rewritten to use:

```text
evidenceflow.frozen_etv3_variable_flow
```

instead of the live `evidence_transition_graphragv3_variable_flow` package.

## Rules

- Do not edit this snapshot while changing the live ETv3 branch.
- PCEC native E2E code should import this frozen package, not the live ETv3
  package.
- If the expander must change, create a new dated snapshot and record why.
- Reader/composer code does not belong here; it belongs in the PCEC package
  outside this frozen snapshot.
