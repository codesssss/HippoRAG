"""D-PathRAG utilities.

D-PathRAG treats multi-hop evidence as an ordered path over a fixed candidate
pool.  This package intentionally starts with data/audit/cache primitives; the
trainable selector and differentiable reader are added only after the local
dataset and pool protocol are verified.
"""

