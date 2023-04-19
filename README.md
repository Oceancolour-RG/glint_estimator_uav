# glint_estimator_uav
This repo roughly estimates the location and spread of sunglint contamination in UAV imagery given the date, time, location, wind speed and direction, the camera's intrinsic parameters and it's pose (pitch, roll, heading).


The Cox-Munk algorithm (Cox and Munk, 1954a, 1954b) is utilised, which does not include impacts of wind age or swell. As stated by Kay et al. (2009): "The [Cox-Munk] techniques ... rely on being able to make a prediction of glint based on a probability distribution of sea surface slopes. This is reasonable at the 100–1,000 m scale of ocean color sensor pixels, but may be less accurate for images at resolutions of 1–10 m, where the pixel size cannot be assumed to be much larger than the features of the water surface and the statistical assumptions about a surface composed of many reflecting facets may not hold"

Note:
The Cox-Munk sunglint code has been converted from NASA SeaDAS's fortran code into Python3. Below is the link to the fortran code:
https://oceancolor.gsfc.nasa.gov/docs/ocssw/getglint_8f_source.html

# References
- Cox, C.S. and Munk, W.H. (1954a), Statistics of the Sea Surface Derived from Sun Glitter. *Journal of Marine Research*, 13, 198-227.
- Cox, C.S. and Munk, W.H. (1954b), Measurement of the Roughness of the Sea Surface from Photographs of the Sun’s Glitter. *Journal of the Optical Society of America*, 44(11), 838-850.
- Kay, S., Hedley, J.D., Lavender, S. (2009). Sun Glint Correction of High and Low Spatial Resolution Images of Aquatic Scenes: a Review of Methods for Visible and Near-Infrared Wavelengths. *Remote Sensing*, 1(4), 697-730; https://doi.org/10.3390/rs1040697

# Examples
## 1. Estimate the glint reflectance in the Micasense RedEdge-MX imagery acquired over a coastal shallow water location.
Here you would need to install the micasense imageprocessing repo (https://github.com/Oceancolour-RG/imageprocessing). In the following example, the RedEdge MX had a heading -45 degrees from North. The solar azimuth angle during the date and time of the acquisition was 71 degrees North and a solar zenith of 26.5 degrees. The `micasense.capture` class reads in the `YAML_FN` and internally loads all required parameters (such as UTC date and time, latitude, longitude, and the solar zenith and azimuth angles). The exact intrinsic parameters of the RedEdge MX such as the focal length and perspective centres are provided in the yaml file, but are specified here to exemplify the process.

```python
from pathlib import Path
import matplotlib.pyplot as plt
import micasense.capture as capture
from glint_estimator_uav.utils import disp_im
from glint_estimator_uav.estimate_glint import estimate_glint_loc


# Specify the following parameters
WIND_SPEED = 5.0  # m/s, 10 m above sea-surface
WIND_AZI = 270  # wind direction from north (degrees)

# Specify the Micasense RedEdge MX's intrinsic parameters
CAMERA_F = 5.4e-3  # focal length (metres)
CAMERA_PS = 3.75e-6  # pixel size (metres)

# Specify the Micasense RedEdge MX's perspective centre (PC). Assume
# the PC is located in the centre of the image of size (960, 1280)
PC_IMAGE = [(960 - 1.0) / 2.0, (1280 - 1.0) / 2.0]

# Specify external camera orientations. The heading was estimated
# using near-coincident Nearmap aerial imagery.
HEADING = -46.0  # direction of the top of image (degrees from North)
PITCH = 0.0  # degrees
ROLL = 0.0  # degrees

# Specify the location of the metadata yaml's
BPATH = Path("/path/to/uav/20211126_WoodmanPoint/micasense/")
META_PATH = BPATH / "metadata" / "SYNC0009SET"

# Specify the Micasense acquisition to load:
YAML_FN = META_PATH / "IMG_0260.yaml"


def main():

    # load the micasense capture
    ms_capture = capture.Capture.from_yaml(YAML_FN)
    # compute the reflectance
    refl_ims = [img.undistorted_reflectance() for img in ms_capture.images]
    shape = refl_ims[0].shape

    # extract the solar zenith and azimuth angles
    sza, saa = ms_capture.solar_geoms()

    pglint, p_fresnel = estimate_glint_loc(
        heading=HEADING,
        pitch=PITCH,
        roll=ROLL,
        shape=shape,
        pc_image=PC_IMAGE,
        camera_ps=CAMERA_PS,
        camera_f=CAMERA_F,
        sza=sza,
        saa=saa,
        wind_speed=WIND_SPEED,
        wind_azi=WIND_AZI,
        disp_angles=True,  # displays the view zenith and azimuth angles
    )

    disp_im(refl_ims[0], **{"vmin": 0.0001, "vmax": 0.6})
    disp_im(pglint, title="glint reflectance")
    plt.show()


if __name__ == "__main__":
    main()
```
![Alt text](images/micasense.png?raw=true "Micasense RedEdge Band 1")
![Alt text](images/pglint1.png?raw=true "Estimate glint")


## 2. Estimate glint from latitude, longitude, UTC datetime, wind speed and direction and camera parameters
```python
from datetime import datetime
import matplotlib.pyplot as plt
from glint_estimator_uav.utils import disp_im, solar_geoms
from glint_estimator_uav.estimate_glint import estimate_glint_loc


LAT = -32.135862  # decimal degrees North
LON = 115.7468254  # decimal degrees East
UTC_TIME = "2021-11-26 02:16:19.332937+0000"  # UTC
WIND_SPEED = 5.0  # m/s, 10 m above sea-surface
WIND_AZI = 270  # wind direction from north (degrees)

# Specify the Micasense camera's image parameters
CAMERA_F = 5.4e-3  # focal length (metres)
CAMERA_PS = 3.75e-6  # pixel size (metres)

# Specify the Micasens camera's perspective centre (PC)
# Here, we assume that the PC is located in the centre
# of the image of size (960, 1280)
SHAPE = (960, 1280)
PC_IMAGE = [(960 - 1.0) / 2.0, (1280 - 1.0) / 2.0]

# Specify external camera orientations
HEADING = -46.0  # direction of the top of image (degrees from North)
PITCH = 5.0  # degrees
ROLL = 10.0  # degrees


def main():

    # Get datetime object
    dt = datetime.strptime("2021-11-26 02:16:19.332937+0000", "%Y-%m-%d %H:%M:%S.%f%z")

    # get solar angles:
    sza, saa = solar_geoms(lat=LAT, lon=LON, dt=dt)

    pglint, p_fresnel = estimate_glint_loc(
        heading=HEADING,
        pitch=PITCH,
        roll=ROLL,
        shape=SHAPE,
        pc_image=PC_IMAGE,
        camera_ps=CAMERA_PS,
        camera_f=CAMERA_F,
        sza=sza,
        saa=saa,
        wind_speed=WIND_SPEED,
        wind_azi=WIND_AZI,
        disp_angles=True,
    )

    disp_im(pglint, title="glint reflectance")
    plt.show()


if __name__ == "__main__":
    main()
```
![Alt text](images/pglint2.png?raw=true "Example 2 output")
