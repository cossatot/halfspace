from halfspace import load
import numpy as np


def coord_map_forward_1d(coord, x_step = 1, x_shift = 0, mode = 'linear'):
    """ Maps input coordinate (1D, value) to output coordinate given
        parameters step (m) and shift (b) for linear transformation

        y = m * x + b

        Intended to map meshgrid output arrays back to spatial coordinates.
    """

    x_in = coord

    x_step = float(x_step)

    if mode == 'linear':
        x_map = x_step * x_in + x_shift

    else: raise Exception('Nonlinear maps not yet supported')

    return x_map


def coord_map_inverse_1d(coord, x_step = 1, x_shift = 0, mode = 'linear'):

    """ Maps input coordinate (1D, value) to output coordinate given
        parameters step (m) and shift (b) for inverse linear transformation

        x = (y - b) / m

        Intended to map spatial coordinates to meshgrid output arrays.
    """
    x_in = coord

    x_step = float(x_step)

    if mode == 'linear':
        x_map = (x_in - x_shift) / x_step

    else: raise Exception('Nonlinear maps not yet supported')

    return x_map


def coord_map_inverse_2d(coord, x_step = 1, x_shift = 0, y_step = 1,
    y_shift = 0, mode = 'linear'):

    """ Maps input coordinates to output coordinates given
        parameters step (m) and shift (b) for inverse linear transformation

        x = (y - b) / m

        for each coordinate axis.

        Intended to map spatial coordinates to meshgrid output arrays.
    """
    x_in = coord[0]
    y_in = coord[1]

    # insure that steps are floats, so division works correctly
    x_step = float(x_step)
    y_step = float(y_step)

    if mode == 'linear':
        x_map = (x_in - x_shift) / x_step
        y_map = (y_in - y_shift) / y_step

    else: raise Exception('Nonlinear maps not yet supported')

    coord_map = [x_map, y_map]

    return coord_map


def coord_map_inverse_3d(coord, x_step = 1, x_shift = 0, y_step = 1,
    y_shift = 0, z_step = 1, z_shift = 0, mode = 'linear'):

    """ Maps input coordinates to output coordinates given
        parameters step (m) and shift (b) for inverse linear transformation

        x = (y - b) / m

        for each coordinate axis.

        Intended to map spatial coordinates to meshgrid output arrays.
    """

    x_in = coord[0]
    y_in = coord[1]
    z_in = coord[2]

    # insure that steps are floats, so division works correctly
    x_step = float(x_step)
    y_step = float(y_step)
    z_step = float(z_step)

    if mode == 'linear':
        x_map = (x_in - x_shift) / x_step
        y_map = (y_in - y_shift) / y_step
        z_map = (z_in - z_shift) / z_step

    else: raise Exception('Nonlinear maps not yet supported')

    coord_map = [x_map, y_map, z_map]

    return coord_map


def size_output(a1, a2, mode='full'):
    """ Calculates size of convolution output.  a1, a2 are arrays representing
        dimensions of the matrices being convolved.
    """
    size = a1 + a1 - 1
    if mode == 'full':
        ret_size = size
    elif mode == 'same':
        if np.product(a1, axis=0) > np.product(a2, axis=0):
            ret_size = a1
        else:
            ret_size = a2
    elif mode == 'valid':
        ret_size = abs(a2 - a1) + 1

    return ret_size


