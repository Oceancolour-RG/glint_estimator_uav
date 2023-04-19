#!/usr/bin/env python3

import numexpr as ne
from math import pi
from typing import Union, Tuple
from numpy import ndarray, zeros
from glint_estimator_uav.utils import generate_tiles


def getglint(
    vza: ndarray,
    sza: ndarray,
    raa: ndarray,
    wind_speed: float,
    waa: float,
    return_fresnel: bool = False,
) -> Tuple[ndarray, Union[ndarray, None]]:
    """
    Estimates the wavelength-independent sunglint reflectance using the
    Cox and Munk (1954) algorithm. This a direct port from SeaDAS's
    fortran code, taken from:
    https://oceancolor.gsfc.nasa.gov/docs/ocssw/getglint_8f_source.html

    Parameters
    ----------
    vza : numpy.ndarray, {dims=(nrows, ncols), dtype=float32/64}
        sensor's view zenith angle (degrees)

    sza : numpy.ndarray, {dims=(nrows, ncols), dtype=float32/64}
        solar zenith angle (degrees)

    raa : numpy.ndarray, {dims=(nrows, ncols), dtype=float32/64}
        relative azimuth angle (degrees)

    wind_speed: float
        wind speed (m/s)
    waa : float
        wind direction (degrees)
    return_fresnel : bool
        Return fresnel reflectance array

    Returns
    -------
    p_glint : numpy.ndarray
        Estimated sunglint reflectance

    p_fresnel : None or numpy.ndarray
        Fresnel reflectance of sunglint. Useful for debugging

        if return_fresnel=False then p_fresnel=None
        if return_fresnel=True  then p_fresnel=numpy.ndarray

    Raises
    ------
    ValueError:
        * if input arrays are not two-dimensional
        * if dimension mismatch
    Notes
    -----
     ** minimum allowable wind speed is 0.001 m/s

    """
    if (vza.ndim != 2) or (sza.ndim != 2) or (raa.ndim != 2):
        raise ValueError("\ninput arrays must be two dimensional")

    if (vza.shape != sza.shape) or (vza.shape != raa.shape):
        raise ValueError("\nDimension mismatch")

    d2r = pi / 180.0
    n_sw = 1.34  # index of refraction of seawater

    # Isotropic wind (from Cox & Munk)
    wind_speed = max(wind_speed, 0.001)  # why does NASA do this?
    sigc = 0.04964 * (wind_speed**0.5)  # noqa: F841
    sigu = 0.04964 * (wind_speed**0.5)  # noqa: F841

    # create output array
    nr, nc = vza.shape
    p_glint = zeros([nr, nc], order="C", dtype=vza.dtype)

    p_fresnel = None
    if return_fresnel:
        p_fresnel = zeros([nr, nc], order="C", dtype=vza.dtype)

    # This implementation generates several (potentially) large arrays.
    # We will therefore use tiling to reduce memory consumption.
    tiles = list(generate_tiles(nsamples=nc, nlines=nr, xtile=256, ytile=256))
    for tile in tiles:
        r0, rf = int(tile[0][0]), int(tile[0][1])  # row start, row end
        c0, cf = int(tile[1][0]), int(tile[1][1])  # col start, col end

        vza_r = vza[r0:rf, c0:cf] * d2r
        sza_r = sza[r0:rf, c0:cf] * d2r
        raa_r = raa[r0:rf, c0:cf] * d2r  # noqa: F841

        # initial fortran implementation: if (y1.eq.0.) y1 = 1.e-7
        vza_r[vza_r < 1.0e-7] = 1.0e-7
        sza_r[sza_r < 1.0e-7] = 1.0e-7

        cos_vza = ne.evaluate("cos(vza_r)")  # noqa: F841
        cos_sza = ne.evaluate("cos(sza_r)")  # noqa: F841
        sin_sza = ne.evaluate("sin(sza_r)")  # noqa: F841
        cos_raa = ne.evaluate("cos(raa_r)")  # noqa: F841
        sin_raa = ne.evaluate("sin(raa_r)")  # noqa: F841

        omega = ne.evaluate(
            "arccos(cos_vza * cos_sza - sin(vza_r) * sin_sza * cos_raa) / 2.0"
        )
        omega[omega < 1.0e-7] = 1.0e-7  # unclear why NASA sets to this value

        beta = ne.evaluate("arccos((cos_vza + cos_sza) / (2.0 * cos(omega)))")
        beta[beta < 1.0e-7] = 1.0e-7

        alpha = ne.evaluate(
            "arccos((cos(beta) * cos_sza - cos(omega)) / (sin(beta) * sin_sza))"
        )
        alpha[sin_raa < 0.0] *= -1.0

        alphap = alpha + (waa * d2r)  # noqa: F841
        swig = ne.evaluate("sin(alphap) * tan(beta) / sigc")  # noqa: F841
        eta = ne.evaluate("cos(alphap) * tan(beta) / sigu")  # noqa: F841

        expon = ne.evaluate("-(swig**2 + eta**2) / 2.0")
        expon[expon < -30] = -30  # trap underflow
        expon[expon > 30] = 30  # trap overflow
        prob = ne.evaluate("exp(expon) / (2.0 * pi * sigu * sigc)")  # noqa: F841

        p_fr = reflec(w=omega, n_sw=n_sw)  # noqa: F841
        if return_fresnel:
            p_fresnel[r0:rf, c0:cf] = p_fr

        p_glint[r0:rf, c0:cf] = ne.evaluate(
            "p_fr * prob / (4.0 * cos_vza * (cos(beta) ** 4))"
        )

    return p_glint, p_fresnel


def reflec(w: ndarray, n_sw: float) -> float:
    """
    Parameters
    ----------
    w : numpy.ndarray
        Angle of incidence of a light ray at the water surface (radians)

    n_sw : float
        Index of refraction of seawater

    Returns
    -------
    rho : ndarray
        fresnel reflectance

    Equations
    ---------
    n_air sin(w) = n_sw sin(w_pr)

                 tan(w - w_pr)**2
    Refl(par)  = ----------------
                 tan(w + w_pr)**2

                 sin(w - w_pr)**2
    Refl(perp) = ----------------
                 sin(w + w_pr)**2

    Where:
         w      Incident angle
         n_air  Index refraction of Air
         w_pr   Refracted angle
         n_sw   Index refraction of sea water
    """
    w_pr = ne.evaluate("arcsin(sin(w) / n_sw)")  # noqa: F841
    rho = ne.evaluate(
        "0.5*((sin(w-w_pr)/sin(w+w_pr))**2 + (tan(w-w_pr)/tan(w+w_pr))**2)"
    )
    rho[w < 0.00001] = 0.0204078  # Unclear why NASA sets this value?

    return rho
