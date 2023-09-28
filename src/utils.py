#!/usr/bin/env python3

import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from numpy import ndarray, full, arange
from typing import Optional, Union, Tuple
from pysolar.solar import get_altitude, get_azimuth
from mpl_toolkits.axes_grid1 import make_axes_locatable


def disp_im(
    im: ndarray,
    add_cbar: bool = True,
    title: Optional[str] = None,
    opng: Optional[Union[str, Path]] = None,
    cb_label: Optional[str] = None,
    **kwargs,
) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    impl_ = ax.imshow(im, interpolation="None", **kwargs)
    ax.axis("off")
    if title:
        ax.set_title(title)

    if add_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        cbar = fig.colorbar(impl_, cax=cax, orientation="vertical", pad=0.02)
        if cb_label:
            cbar.set_label(cb_label, size="large")
        cbar.ax.tick_params(labelsize="large")

    if opng:
        fig.savefig(
            opng,
            format="png",
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300,
        )
        print(f"wrote: {opng}")

    return


def solar_geoms(lat: float, lon: float, dt: datetime) -> Tuple[float, float]:
    """
    Get the solar geometry

    Parameters
    ----------
    lat : float
        latitude (decimal degrees North)
    lon : float
        longitude (decimal degrees East)
    dt : datetime
        UTC datetime

    Returns
    -------
    sza : float
        solar zenith angle
    saa : float
        solar azimuth anlge, where,
        0  = north, 90 = east, 180 = south, 270 = west
    """
    sza = 90.0 - get_altitude(lat, lon, dt)

    saa = get_azimuth(lat, lon, dt)  # negative angles are west of North
    saa = saa % 360  # convert to [0, 360]

    return sza, saa


def convert2arr(v: Union[float, int, ndarray], shape: Tuple[int, int]) -> ndarray:
    """
    converts `v` to a single value numpy.ndarray if it's a float or int.

    Parameters
    ----------
    v : float, int or numpy.ndarray
        parameter to check
    shape : Tuple[int, int]
        shape of expected array
    """
    if isinstance(v, (float, int)):
        return full(shape, fill_value=v, order="C", dtype="float64")
    elif isinstance(v, ndarray):
        return v
    else:
        raise ValueError("`v` must be float, int or numpy.ndarray")


def generate_tiles(
    nsamples: int,
    nlines: int,
    sample_start: Optional[int] = 0,
    line_start: Optional[int] = 0,
    xtile: Optional[int] = None,
    ytile: Optional[int] = None,
) -> Tuple:
    """
    Generates a list of tile indices for a 2D array.

    Parameters
    ----------
    nsamples: int
        An integer expressing the total number of samples of an array to tile.

    nlines: int
        An integer expressing the total number of lines of an array to tile.

    sample_start: int
        An integer expressing the sample index of where tiling begins.
        Default is 0 representing the first sample.

    line_start: int
        An integer expressing the line index of where tiling begins.
        Default if 0 representing the first line.

    xtile: int or None
        (Optional) The desired size of the tile in the x-direction.
        Default is all samples

    ytile: int or None
        (Optional) The desired size of the tile in the y-direction.
        Default is min(100, nlines) lines.

    Returns
    -------
        Each tuple in the generator contains
        ((ystart,yend),(xstart,xend)).

    Notes
    -----
    To extract tile indices for a subset of the image, use sample_start
    and line_start,

    Example:
        >>> from tiling import generate_tiles
        >>> tiles = generate_tiles(8624, 7567, xtile=1000, ytile=400)
        >>> for tile in tiles:
        >>>     ystart = int(tile[0][0])
        >>>     yend = int(tile[0][1])
        >>>     xstart = int(tile[1][0])
        >>>     xend = int(tile[1][1])
        >>>     xsize = int(xend - xstart)
        >>>     ysize = int(yend - ystart)
        >>>     # When used to read data from disk
        >>>     subset = gdal_indataset.ReadAsArray(xstart, ystart, xsize, ysize)
        >>>     # The same method can be used to write to disk.
        >>>     gdal_outdataset.WriteArray(array, xstart, ystart)
        >>>     # A rasterio dataset
        >>>     subset = rio_ds.read([4, 3, 2], window=tile)
        >>>     # Or simply move the tile window across an array
        >>>     subset = array[ystart:yend,xstart:xend] # 2D
        >>>     subset = array[:,ystart:yend,xstart:xend] # 3D
    """

    def create_tiles(nsamples_: int, nlines_: int, xstart_: ndarray, ystart_: ndarray):
        """
        Creates a generator object for the tiles.
        """
        for ystep in ystart_:
            if ystep + ytile < nlines_:
                yend = ystep + ytile
            else:
                yend = nlines_
            for xstep in xstart_:
                if xstep + xtile < nsamples_:
                    xend = xstep + xtile
                else:
                    xend = nsamples_
                yield ((ystep, yend), (xstep, xend))

    # check for default or out of bounds
    if xtile is None or xtile < 0:
        xtile = nsamples
    if ytile is None or ytile < 0:
        ytile = min(100, nlines)

    if sample_start < 0:
        sample_start = 0

    if line_start < 0:
        line_start = 0

    xstart = arange(sample_start, sample_start + nsamples, xtile)
    ystart = arange(line_start, line_start + nlines, ytile)

    tiles = create_tiles(sample_start + nsamples, line_start + nlines, xstart, ystart)

    return tiles
