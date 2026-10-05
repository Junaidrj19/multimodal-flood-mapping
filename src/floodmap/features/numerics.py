"""Numerically explicit feature transforms.

Design rules
------------
Every function here returns ``(values, valid)`` and obeys the same contract:

1. A pixel that cannot be computed from valid, in-domain inputs is returned as
   ``invalid`` in the mask. Its value is left as ``nodata`` and must never be
   interpreted.
2. No invalid input is replaced by a plausible scientific value. Substituting,
   for example, a small positive backscatter for a zero would manufacture a
   finite log-ratio where the observation does not support one.
3. The mathematical domain of each transform is checked before it is applied,
   not patched afterwards. Clipping is not used to make a computation succeed;
   ``docs/scientific-assumptions.md`` requires the distinction between "not
   observed" and "observed" to survive, and clipping destroys it.
4. Arithmetic is performed in float64 and cast once at the end, so the declared
   output dtype is a storage decision rather than an accuracy accident.

Floating-point warnings are suppressed locally with :func:`numpy.errstate`
because the invalid cases are identified by an explicit domain mask first. The
suppression therefore hides arithmetic that is already known to be discarded,
not arithmetic whose validity is unknown.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .errors import FeatureConfigurationError

__all__ = [
    "COMPUTE_DTYPE",
    "apply_nodata",
    "cross_polarisation_difference_db",
    "difference",
    "finite_and_valid",
    "linear_to_db",
    "log_ratio_db",
    "normalized_difference",
    "slope_horn_degrees",
]

#: All feature arithmetic happens at this precision regardless of storage dtype.
COMPUTE_DTYPE = "float64"


def _as_compute(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=COMPUTE_DTYPE)


def finite_and_valid(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Intersect an incoming validity mask with finiteness.

    An M2 artifact declares its own valid mask, but a NaN or infinity reaching
    this point would otherwise propagate silently through later arithmetic. Both
    conditions are required, so neither can rescue the other.
    """
    if array.shape != valid.shape:
        raise FeatureConfigurationError(
            f"validity mask shape {valid.shape} does not match data shape {array.shape}"
        )
    return np.asarray(valid, dtype=bool) & np.isfinite(_as_compute(array))


def apply_nodata(values: np.ndarray, valid: np.ndarray, nodata: float) -> np.ndarray:
    """Write ``nodata`` into every invalid pixel.

    Applied as the final step of every feature so that the raster and the mask
    can never disagree about which pixels carry an observation.
    """
    result = np.array(values, dtype=COMPUTE_DTYPE, copy=True)
    result[~np.asarray(valid, dtype=bool)] = float(nodata)
    return result


def difference(
    post: np.ndarray,
    pre: np.ndarray,
    post_valid: np.ndarray,
    pre_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``post - pre`` in the inputs' own units.

    Valid only where both observations are valid: a change value derived from
    one observed and one unobserved date is not a change measurement.
    """
    valid = finite_and_valid(post, post_valid) & finite_and_valid(pre, pre_valid)
    with np.errstate(invalid="ignore"):
        values = _as_compute(post) - _as_compute(pre)
    valid &= np.isfinite(values)
    return values, valid


def linear_to_db(linear: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``10 * log10(linear)``, defined only for strictly positive input.

    Zero and negative backscatter are outside the domain of the logarithm. They
    are marked invalid rather than floored at a small positive number, because
    low backscatter is itself the candidate open-water signal
    (``docs/data-contract.md`` §1.7) and flooring would convert a non-observation
    into a very dark — that is, water-like — value.
    """
    in_domain = finite_and_valid(linear, valid) & (_as_compute(linear) > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 10.0 * np.log10(_as_compute(linear))
    out_valid = in_domain & np.isfinite(values)
    return values, out_valid


def log_ratio_db(
    post_linear: np.ndarray,
    pre_linear: np.ndarray,
    post_valid: np.ndarray,
    pre_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``10 * log10(post / pre)`` for linear-power backscatter.

    Domain: ``pre > 0`` and ``post > 0``. A zero denominator is not regularised
    with an epsilon; the pixel is invalid. The decibel form is used rather than
    the bare ratio because it is symmetric about zero for reciprocal changes,
    which keeps an increase and an equivalent decrease comparable in magnitude.

    This function must not be applied to data already expressed in decibels —
    in that representation the equivalent quantity is the plain difference. The
    caller enforces that distinction via the declared backscatter
    representation; see :mod:`floodmap.features.registry`.
    """
    domain = (
        finite_and_valid(post_linear, post_valid)
        & finite_and_valid(pre_linear, pre_valid)
        & (_as_compute(post_linear) > 0.0)
        & (_as_compute(pre_linear) > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 10.0 * np.log10(_as_compute(post_linear) / _as_compute(pre_linear))
    valid = domain & np.isfinite(values)
    return values, valid


def cross_polarisation_difference_db(
    co_polarised_db: np.ndarray,
    cross_polarised_db: np.ndarray,
    co_valid: np.ndarray,
    cross_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``co - cross`` in decibels, i.e. the log of the co/cross power ratio.

    Both inputs must already be in decibels; the subtraction is only a ratio in
    that representation.
    """
    return difference(co_polarised_db, cross_polarised_db, co_valid, cross_valid)


def normalized_difference(
    first: np.ndarray,
    second: np.ndarray,
    first_valid: np.ndarray,
    second_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(first - second) / (first + second)``.

    The generic normalised-difference form underlying the water and vegetation
    indices in the registry. Dimensionless; bounded to ``[-1, 1]`` whenever both
    inputs are non-negative, which is the case for physically valid surface
    reflectance.

    A zero denominator is invalid, not clipped. ``first + second == 0`` means
    both bands are zero (or cancel, which valid reflectance cannot do), so the
    index carries no information there and any substituted value would be
    fabricated.
    """
    valid = finite_and_valid(first, first_valid) & finite_and_valid(second, second_valid)
    a = _as_compute(first)
    b = _as_compute(second)
    denominator = a + b
    valid &= denominator != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (a - b) / denominator
    valid &= np.isfinite(values)
    return values, valid


def slope_horn_degrees(
    elevation: np.ndarray,
    valid: np.ndarray,
    *,
    resolution_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Terrain slope in degrees, third-order finite difference (Horn, 1981).

    Horn's 3x3 kernel is used rather than a simple two-cell gradient because it
    weights the eight neighbours and is the formulation implemented by common
    terrain tools, which makes the output comparable to them.

    With the window written north-up as::

        z1 z2 z3
        z4 z5 z6
        z7 z8 z9

    the partial derivatives are::

        dz/dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cellsize)
        dz/dy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8 * cellsize)
        slope = degrees(arctan(hypot(dz/dx, dz/dy)))

    Validity rules, both deliberate:

    * the one-pixel border is invalid, because a 3x3 window does not exist
      there and padding would invent elevations;
    * an interior pixel is invalid if **any** of its nine window cells is
      invalid, because a slope computed across a void is not a measurement of
      the terrain.

    ``resolution_m`` must be the analysis-grid pixel size in the same linear
    unit as the elevation values. The caller is responsible for establishing
    that both are metres; this function cannot detect a unit mismatch, and a
    projected CRS in degrees would yield a meaningless gradient.
    """
    if resolution_m <= 0:
        raise FeatureConfigurationError("slope requires a positive grid resolution")
    if elevation.ndim != 2:
        raise FeatureConfigurationError("slope requires a single-band elevation array")
    height, width = elevation.shape
    if height < 3 or width < 3:
        raise FeatureConfigurationError(
            "slope requires at least a 3x3 analysis grid; a smaller grid has no interior pixel"
        )

    z = _as_compute(elevation)
    ok = finite_and_valid(elevation, valid)
    values = np.zeros_like(z)
    out_valid = np.zeros(z.shape, dtype=bool)

    # Named views onto the 3x3 neighbourhood of every interior pixel.
    z1, z2, z3 = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    z4, z6 = z[1:-1, :-2], z[1:-1, 2:]
    z7, z8, z9 = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]

    dz_dx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * float(resolution_m))
    dz_dy = ((z1 + 2.0 * z2 + z3) - (z7 + 2.0 * z8 + z9)) / (8.0 * float(resolution_m))

    with np.errstate(invalid="ignore"):
        interior = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    window_valid = (
        ok[:-2, :-2]
        & ok[:-2, 1:-1]
        & ok[:-2, 2:]
        & ok[1:-1, :-2]
        & ok[1:-1, 1:-1]
        & ok[1:-1, 2:]
        & ok[2:, :-2]
        & ok[2:, 1:-1]
        & ok[2:, 2:]
    )

    values[1:-1, 1:-1] = interior
    out_valid[1:-1, 1:-1] = window_valid & np.isfinite(interior)
    return values, out_valid
