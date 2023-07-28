"""
* numpy: The module needed for calculating BF Data
* matplotlib: The module for visualizing results
* cv2: The module for resizing ndarray.
* numba: The module for caching python code into c++ for speeding up python code.
"""
import numpy as np
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import cv2
from numba import jit


@jit(nopython=True, cache=True)
def convert_raw_2_db_scale(raw_data, gain):
    """
    Convert Raw BF Data into dB Scale data. This needs gain for correcting each value.

    Parameters
    ----------
    raw_data : ndarray
        The raw bf data sent from BATCAM FX
    gain : int
        Microphone gain value set in BATCAM FX.

    Returns
    ----------
    dB_scale : ndarray
        dB Scale data converted from input BF Data
    """
    gain = 1 if gain == 0 else gain
    return raw_data / (gain * gain) * 0.00031921


@jit(nopython=True, cache=True)
def convert_db_scale_2_db(scale):
    """
    Convert dB Scale data into dB Data.

    Parameters
    ----------
    scale : ndarray
        The dB Scale data converted from `convert_raw_2_dB_scale(raw_data, gain)`

    Returns
    ----------
    res : ndarray
        dB Data that can be using for visualization.
    """
    res = np.where(scale <= 0, 0, 10 * np.log10(scale / 0.0000000004))
    return res


@jit(nopython=True, cache=True)
def convert_bf_data_2_2d_db_array(bf_data, gain):
    """
    Convert BF Data array into 2D dB array.

    This function calls 2 functions for changing BF Data into dB Data.

    Check more from `convert_raw_2_dB_scale` and `convert_dB_scale_2_dB` functions.


    Parameters
    ----------
    bf_data : List<float>
        The BF data array sent from BATCAM FX
    gain : int
        The gain value sent from BATCAM FX

    Returns
    ----------
    res : ndarray
        40x30 2D array that converted into dB Data.
    """
    bf_data = np.asarray(bf_data)

    db_scale = convert_raw_2_db_scale(bf_data, gain)
    db_data = convert_db_scale_2_db(db_scale)
    db_data = db_data.reshape(30, 40)
    return db_data


@jit(nopython=True, cache=True)
def normalize_cvt2uint8(db_data, db_range):
    """
    Extracts maximum value from the input dB_data,
    and leaves only max value and dB_range, converts the rest to zero.
    Then normalize calculated values and convert type to uint8 array for applying color table.

    After this process, by applying color table,
    you can create final image can be overlapped to RTSP stream.

    Parameters
    ----------
    db_data : ndarray
        Resized ndarray
    db_range : float
        The factor for including around the portion of maximum value.
        Rest of value will be changed into 0.

    Returns
    ----------
    res : ndarray
        UInt8 2D ndarray for creating image after applying color table.
    """
    db_max = db_data.max()
    db_min = db_max - db_range

    db_data = np.where(db_data <= db_min, 0, db_data)
    db_data = (db_data - np.min(db_data)) / (np.max(db_data) - np.min(db_data))
    db_data *= 255
    db_data = db_data.astype(np.uint8)

    return db_data


def interpolate(bf_data, gain, db_range):
    """
    This runs various method in `interpolator.py` in once.
    

    Parameters
    ----------
    bf_data : list<float>
        The raw bf data sent from BATCAM FX
    gain : int
        The factor for using calculation of lower limit.
        Please see `normalize_cvt2uint8`
    db_range : float
        The factor for using calculation of lower limit.
        Please see `normalize_cvt2uint8`

    Returns
    ----------
    res : ndarray
        UInt8 2D ndarray for creating image after applying color table.
    """
    db_data = convert_bf_data_2_2d_db_array(bf_data, gain)
    db_data_resized = cv2.resize(
        db_data,
        dsize=(1600, 1200),
        interpolation=cv2.INTER_LINEAR
    )
    return normalize_cvt2uint8(db_data_resized, db_range)


mplstyle.use('fast')
plt.ion()


def plt_result(converted_value):
    """
    Visualize 2D ndarray from the input with matplotlib.
    Due to performance issue, this can't keep up to 25 FPS.
    Parameters
    ----------
    converted_value : ndarray
        The ndarray value finished calculation for visualizing.

    Returns
    -------
        Nothing.
    """
    plt.imshow(converted_value)
    plt.draw()
    plt.pause(0.000000001)
    plt.clf()
