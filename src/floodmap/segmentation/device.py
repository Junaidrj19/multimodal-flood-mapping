"""Device selection and run reproducibility.

Device priority is explicit config, then CUDA, then Apple MPS, then CPU. The
order is a practical one rather than a benchmark claim: CUDA is the only
backend with full op coverage for this model, MPS is the local accelerator on
this machine, and CPU always works. Nothing here requires a GPU, so the unit
tests run anywhere.

Seeding covers Python, NumPy and Torch. ``deterministic`` additionally asks
cuDNN for reproducible kernel selection. It is NOT promoted to
``torch.use_deterministic_algorithms(True)``, which would raise on several
ordinary convolution and pooling backward kernels and would make the baseline
unrunnable; the weaker guarantee is stated rather than overstated.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["available_devices", "describe_device", "seed_everything", "select_device"]


def available_devices() -> dict:
    """What this machine actually offers, for the run record."""

    mps_built = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_built()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cpu": True,
    }


def select_device(explicit: Optional[str] = None) -> torch.device:
    """Resolve the training device.

    Parameters
    ----------
    explicit
        A device string from configuration, e.g. ``cuda``, ``mps``, ``cpu``.
        ``None`` or ``"auto"`` triggers automatic selection. An explicitly
        requested device that is unavailable raises rather than silently
        falling back: a run that was meant to use a GPU and quietly spent hours
        on CPU is worse than a run that refused to start.
    """

    if explicit and explicit.lower() != "auto":
        requested = explicit.lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"device {explicit!r} was requested explicitly but CUDA is not available"
            )
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError(
                    f"device {explicit!r} was requested explicitly but MPS is not available"
                )
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    """Human- and machine-readable device identity for the run record."""

    record = {"type": device.type, "index": device.index}
    if device.type == "cuda" and torch.cuda.is_available():
        record["name"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        record["name"] = "Apple MPS"
    else:
        record["name"] = "cpu"
    return record


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and Torch; return the seed for recording.

    ``PYTHONHASHSEED`` is set for completeness, with the caveat that it only
    takes effect for interpreters started after this call — it is recorded so
    the run record is honest about which knobs were actually turned.
    """

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False
    return seed
