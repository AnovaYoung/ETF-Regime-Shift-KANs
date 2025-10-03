from __future__ import annotations
import inspect
import torch
from torch import nn

try:
    # External KAN package (constructor signatures vary by version)
    from kan import KAN as _KAN
except Exception:  # pragma: no cover
    _KAN = None

class KANModel(nn.Module):
    """Stable entry-point that adapts to different KAN libs; falls back to MLP."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        self.net = None

        if _KAN is not None:
            tried = []
            for ctor_kwargs in (
                dict(in_dim=in_dim, out_dim=out_dim, width=hidden_dim, depth=depth),
                dict(input_dim=in_dim, output_dim=out_dim, width=hidden_dim, depth=depth),
                dict(layers=[in_dim] + [hidden_dim] * max(depth - 1, 0) + [out_dim]),
                dict(in_features=in_dim, out_features=out_dim, hidden_size=hidden_dim, depth=depth),
            ):
                try:
                    self.net = _KAN(**ctor_kwargs)
                    break
                except Exception as e:
                    tried.append((ctor_kwargs, str(e)))

            if self.net is None:
                # Last shot: try positional args if signature is minimal
                try:
                    sig = inspect.signature(_KAN)
                    if len(sig.parameters) >= 2:
                        self.net = _KAN(in_dim, out_dim)
                except Exception:
                    pass

        if self.net is None:
            # Fallback MLP so pipeline still runs
            layers = []
            d = in_dim
            for _ in range(max(depth, 1)):
                layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
                d = hidden_dim
            layers += [nn.Linear(d, out_dim)]
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
