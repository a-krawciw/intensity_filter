import numpy as np


def sph2cart(r, az, el):
    z = r*np.sin(el)
    hyp = r*np.cos(el)
    y = hyp*np.sin(az)
    x = hyp*np.cos(az)

    return x, y, z


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el
