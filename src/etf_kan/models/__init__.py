try:
    from .kan_model import KANModel
except Exception:  # pragma: no cover
    KANModel = None

__all__ = ["KANModel"]
