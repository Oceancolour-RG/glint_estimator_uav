#!/usr/bin/env python3

import numexpr as ne
from typing import Union, Tuple
from numpy import ndarray, meshgrid, arange
from scipy.spatial.transform import Rotation as R
from glint_estimator_uav.coxmunk import getglint
from glint_estimator_uav.utils import disp_im, convert2arr


def calc_view_angles(
    heading: float,  # degrees
    pitch: float,  # degrees
    roll: float,  # degrees
    shape: Tuple[int, int],
    pc_image: Tuple[float, float],
    camera_ps: float,
    camera_f: float,
    vaa_0to360: bool = False,
) -> Tuple[ndarray, ndarray]:
    """
    Compute the view zenith and azimuth angles from the camera's intrinsic
    parameters and it's orientation (pitch, roll, heading).

    Parameters
    ----------
    heading : float
        heading or yaw of the image (degrees). This is a
        rotation about the z-axis. Examples,
            if heading == 0 then north is up in the image
            if heading == 90 then east is up in the image
    pitch : float
        pitch of the image (degrees). This is a rotation
        about the y-axis
    roll : float
        roll of the image (degrees). This is a rotation
        about the x-axis
    shape : Tuple[int, int]
        image size (nrows, ncols)
    pc_image : Tuple[float, float]
        perspective centre of the image (in pixels),
        (nrows, ncols)
    camera_ps : float
        camera pixel size (metres)
    camera_f : float
        camera focal length (metres)
    vaa_0to360 : bool
        whether the view azimuth angle ranges from 0 to 360 degrees (True)
        or -180 to 180 (False)

    Returns
    -------
    vza : numpy.ndarray, {dims=(nrows, ncols)}
        view zenith angle
    vaa : numpy.ndarray, {dims=(nrows, ncols)}
        view azimuth angle
    """
    pi = 3.141592653589793

    def collinear(ri: ndarray, x: ndarray, y: ndarray, f: float) -> ndarray:
        ri1, ri2, ri3 = ri[0], ri[1], ri[2]  # noqa: F841
        return ne.evaluate("(ri1 * x) + (ri2 * y) - (ri3 * f)")

    r2d = 180.0 / pi  # noqa: F841

    # Compute rotation matrix ("ZYX" -> Heading, Pitch, Roll respectively)
    r = R.from_euler("ZYX", angles=[heading, pitch, roll], degrees=True)
    rmat = r.as_matrix()  # ndarray

    # Coordinate system:
    # image y-axis: Pointing right from the PC (to match pitch in "ZYX")
    # image x-axis: Pointing up from the PC (to match roll in "ZYX")
    yaxis, xaxis = meshgrid(arange(0, shape[1]), arange(shape[0] - 1, -1, -1))
    # yaxis: stack of row vectors: [0, 1, ..., ncols-1] (pointing to the right)
    # xaxis: stack of col vectors: [nrows-1, nrows-2, ..., 0] (pointing upwards)

    # At the moment the origin is at the bottom left pixel. Move origin
    # to the location of perspective centre and convert from pixels to metres
    ximage = (xaxis - pc_image[0]) * camera_ps
    yimage = (yaxis - pc_image[1]) * camera_ps

    # set up the collinearity equation:
    # [[X{object} - X{PC},            [[r11, r12, r13],     [[X{image},
    #   Y{object} - Y{PC},         =   [r21, r22, r23],  *    Y{image},
    #   Z{object} - Z{PC}]] (MAP)      [r31, r32, r33]]       -f]]
    #
    # Where f = focal length
    rot_x = collinear(ri=rmat[0, :], x=ximage, y=yimage, f=camera_f)  # noqa: F841
    rot_y = collinear(ri=rmat[1, :], x=ximage, y=yimage, f=camera_f)  # noqa: F841
    rot_z = collinear(ri=rmat[2, :], x=ximage, y=yimage, f=camera_f)  # noqa: F841

    # compute the view zenith angle:
    vza = ne.evaluate("r2d * arctan(((rot_x**2 + rot_y**2) ** 0.5) / abs(rot_z))")

    # compute the view azimuth angle. Note arctan2 outputs angles: [-180, 180]
    vaa = ne.evaluate("r2d * arctan2(rot_y, rot_x)")
    if vaa_0to360:
        vaa = vaa % 360

    return vza, vaa


def estimate_glint_loc(
    heading: float,
    pitch: float,
    roll: float,
    shape: Tuple[int, int],
    pc_image: Tuple[float, float],
    camera_ps: float,
    camera_f: float,
    sza: Union[float, ndarray],
    saa: Union[float, ndarray],
    wind_speed: float,
    wind_azi: float,
    disp_angles: bool = False,
) -> Tuple[ndarray, ndarray]:
    """
    Estimate glint reflectance for a camera

    Parameters
    ----------
    heading : float
        heading or yaw of the image (degrees). This is a
        rotation about the z-axis. Examples,
            if heading == 0 then north is up in the image
            if heading == 90 then east is up in the image
    pitch : float
        pitch of the image (degrees). This is a rotation
        about the y-axis
    roll : float
        roll of the image (degrees). This is a rotation
        about the x-axis
    shape : Tuple[int, int]
        image size (nrows, ncols)
    pc_image : Tuple[float, float]
        perspective centre of the image (in pixels),
        (nrows, ncols)
    camera_ps : float
        camera pixel size (metres)
    camera_f : float
        camera focal length (metres)
    vaa_0to360 : bool
        whether the view azimuth angle ranges from 0 to 360 degrees (True)
        or -180 to 180 (False)
    sza: Union[float, numpy.ndarray]
        solar zenith angle (degrees)
    saa: Union[float, numpy.ndarray]
        solar azimuth anlge (degrees)
    wind_speed: float
        wind speed (m/s)
    wind_azi : float
        wind direction from north (degrees)
    disp_angles : bool
        whether to display viewing angles and glint

    Returns
    -------
    p_glint : numpy.ndarray
        Estimated sunglint reflectance

    p_fresnel : numpy.ndarray
        Fresnel reflectance of sunglint.
    """
    # Get the vza and vaa.
    vza, vaa = calc_view_angles(
        heading=heading,
        pitch=pitch,
        roll=roll,
        shape=shape,
        pc_image=pc_image,
        camera_ps=camera_ps,
        camera_f=camera_f,
        vaa_0to360=False,
    )

    # calculate the relative azimuth angle, see:
    # https://forum.earthdata.nasa.gov/viewtopic.php?t=1779
    # https://forum.earthdata.nasa.gov/viewtopic.php?f=7&t=1558
    # According to Sean Bailey:
    # "A relative azimuth of 180 would be the sun behind the sensor", i.e.
    # 0   degs = sensor looking towards the sun
    # 180 degs = sun behind sensor

    raa = abs(vaa - saa)  # 180 = sun behind sensor
    # raa = abs(180 + vaa - saa_)   # 0 = sun behind sensor
    raa[raa > 180] = 360 - raa[raa > 180]
    raa = abs(raa)

    pglint, p_fresnel = getglint(
        vza=vza,  # ndarray
        sza=convert2arr(v=sza, shape=shape),  # ndarray
        raa=raa,  # ndarray
        wind_speed=wind_speed,
        waa=wind_azi,  # wind direction from North (degrees)
        return_fresnel=True,
    )

    if disp_angles:
        disp_im(vza, title="view zenith angle")
        disp_im(raa, title="relative azimuth angle")

    return pglint, p_fresnel
