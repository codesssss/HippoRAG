try:
    from .HippoRAG import HippoRAG
except ModuleNotFoundError:  # pragma: no cover - allows utility-only imports in lightweight environments.
    HippoRAG = None
