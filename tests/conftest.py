"""CPU fixtures remain usable without accelerator packages or hardware."""
import pytest


def pytest_configure(config):
    for marker in ("cpu: backend-independent or tiny CPU qualification", "assets: explicit pinned local assets only", "mps: requires a real Metal device", "mlx: requires native MLX Metal execution"):
        config.addinivalue_line("markers", marker)
