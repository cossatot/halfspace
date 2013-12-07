

from halfspace import load
#from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from scipy.fftpack import fft, ifft, ifftshift, fft2, ifft2, fftn, \
         ifftn, fftfreq
#import scitools.numpyutils as snp

__metaclass__ = type


"""
Various functions that are in their testing phase, or awaiting placement in
another (sub)module

things I'd like to do here:

Make some plotting functions:
    -- easy way to take 3d array and slice, make images, save things (!!)
    -- exploit different schemes: ndimage or Image, gdal, etc. to make geotiffs
    -- use meshgrid and georeferencing schemes from gdal to plot output arrays
        with coordinates
    -- make vtk files (?)

test ways to further automate the convolution scripts
    -- do the Boussinesq convolutions, then the Cerruti, etc., in a quick way
    -- maybe a quick function to calculate the size of output arrays given the
        convolution mode.  this should be quick and easy when i figure out
        what it looks like in 2d
    -- maybe have more/better parallelization (??)

calculate stresses (full tensor) on arbitrary planes in arbitrary locations.
this is pretty important for resolving stresses on faults.

"""

"""fault stress stuff
"""



class Fault:
    """ Represents a plane in the halfspace, which can have stresses resolved
        on it, or have offset and cause stresses/strains in the nearby space,
        etc.  Currently will just be a receptacle for stresses.  Note the
        reference frame is in right-hand rule (for strike and dip)

        Takes dip in degrees.  Converts internally to radians.  2d faults
        are available as a subclass.  3d will be around soon.  The conventions
        between them are different (such as acceptable fault dip ranges).
    """

    def __init__(self, strike = 0, dip_deg = 0, pts = 0, upper_pt = 0,
                 lower_pt = 0, n_dim = 1, dip_rad = None):
        self.strike = strike
        self.dip_deg = dip_deg
        self.pts = pts


        if dip_rad == None:
            self.dip_rad = np.deg2rad(self.dip_deg)
        else:
            self.dip_rad = self.dip_rad
        #TODO: convert everything to right-hand rule



class Fault_2d:
    """ A fault (line) in a cross-section on which to project stresses, etc.
        Dip is defined from -0 to -90, 90 to 0.  -90 = 90.  Negative dip means
        the fault dips in the negative x direction (towards y axist).


    """

    def __init__(self, upper_pt = None, dip = None, lower_pt = None,
                 max_depth = None, min_depth = None):

        #if x_res != None and z_res != None:
        #    raise Exception('Pick x or z resolution')

        if dip != None:
            self.dip = dip
            self.dip_rad = np.deg2rad(dip)

        if (upper_pt != None and lower_pt != None) and dip != None:
            raise Exception('pick upper and lower points or one point and dip')

        if (upper_pt != None and lower_pt != None) and dip == None:
            self.upper_pt = upper_pt
            self.lower_pt = lower_pt

            x_diff = self.lower_pt[0] - self.upper_pt[0]
            z_diff = self.lower_pt[1] - self.upper_pt[1]

            self.dip_rad = np.arctan( (z_diff / float(x_diff) ) )
            self.dip = np.rad2deg(self.dip_rad)

        elif upper_pt != None and dip != None:
            self.upper_pt = upper_pt
            self.dip = dip
            self.dip_rad = np.deg2rad(self.dip)
            self.max_depth = max_depth

            z_diff = max_depth - upper_pt[1]
            x_diff = z_diff / np.tan(self.dip_rad)

            self.lower_pt = np.array([ (upper_pt[0] + x_diff), max_depth ])


    def sample_from_array(self, array, x_step = 1, x_shift = 0, z_step = 1,
                          z_shift = 0, num_pts = 0, x_res = None, z_res = None,
                          inv_map_mode = 'linear', map_coord_mode ='constant'):

        """ takes Fault_2d parameters and samples from an array using
            coord_map_inverse_2d and scipy.ndimage.map_coordinates.

            Returns [3,n] vector as [x, z, value]
        """
        if num_pts ==0: raise Exception('for now need num_pts')

        upper_pt = self.upper_pt
        lower_pt = self.lower_pt

        upper_pt_map = coord_map_inverse_2d(upper_pt, x_step = x_step,
                                    x_shift = x_shift, z_step = z_step,
                                    z_shift = z_shift, mode = inv_map_mode)

        lower_pt_map = coord_map_inverse_2d(lower_pt, x_step = x_step,
                                    x_shift = x_shift, z_step = z_step,
                                    z_shift = z_shift, mode = inv_map_mode)

        xu, zu = upper_pt_map[0], upper_pt_map[1]
        xl, zl = lower_pt_map[0], lower_pt_map[1]

        x_vec = np.linspace(xu, xl, num = num_pts) # replace with meshgrid ??
        z_vec = np.linspace(zu, zl, num = num_pts)

        value_vec = nd.map_coordinates(array, np.vstack( (x_vec, z_vec) ) )

        x_spatial = coord_map_forward_1d(x_vec, x_step = x_step,
                                    x_shift = x_shift, mode = inv_map_mode)

        z_spatial = coord_map_forward_1d(z_vec, x_step = z_step,
                                    x_shift = z_shift, mode = inv_map_mode)

        out_array = np.vstack( (x_spatial, z_spatial, value_vec) )

        return out_array


#class Fault_3d:


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




"""Fault methods to add:
    slice: get a slice of the fault (say for a planar fault, get a line)

    stuff do deal with non-planar faults (e.g. listric)

    coordinate transformations (very important!)
"""

def _centered(arr, newsize):
    # Return the center newsize portion of the array
    # copied from scipy.signal (c) Travis Oliphant, 1999-2002
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) / 2
    endind = startind + newsize
    myslice = [ slice( startind[k], endind[k]) for k in range( len(endind) ) ]
    return arr[tuple(myslice)]


def half_fft_convolve(in1, in2, size, mode = 'full'):
    """rewrite of fftconvolve from scipy.signal ((c) Travis Oliphant 1999-2002)
        to deal with fft convolution where one signal is not fft transformed
        and the other one is.  Application is, for example, in a loop where
        convolution happens repeatedly with different filters over the same
        signal.  First input is not transformed, second input is.
    """
    s1 = np.array(in1.shape)
    s2 = size - s1 + 1
    complex_result = (np.issubdtype( in1.dtype, np.complex) or
                      np.issubdtype( in2.dtype, np.complex) )

    # Always use 2**n-sized FFT
    fsize = 2 **np.ceil( np.log2( size) )
    IN1 = fftn(in1, fsize)
    IN1 *= in2
    fslice = tuple( [slice( 0, int(sz)) for sz in size] )
    ret = ifftn(IN1)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == 'full':
        return ret
    elif mode == 'same':
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret, osize)
    elif mode == 'valid':
        return _centered(ret, abs(s2 - s1) + 1)

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



def meshgrid(x=None, y=None, z=None, sparse=False, indexing='xy',
             memoryorder=None):
    """
    Function taken from Python SciTools (scitools.googlecode.com).
    Copyright those authors (?), 2012.
    Function appropriated because the scitools.numpyutils module breaks
    matplotlib.

    Extension of ``numpy.meshgrid`` to 1D, 2D and 3D problems, and also
    support of both "matrix" and "grid" numbering.

    This extended version makes 1D/2D/3D coordinate arrays for
    vectorized evaluations of 1D/2D/3D scalar/vector fields over
    1D/2D/3D grids, given one-dimensional coordinate arrays x, y,
    and/or, z.

    >>> x=linspace(0,1,3)        # coordinates along x axis
    >>> y=linspace(0,1,2)        # coordinates along y axis
    >>> xv, yv = meshgrid(x,y)   # extend x and y for a 2D xy grid
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x,y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])

    >>> # 2D slice of a 3D grid, with z=const:
    >>> z=5
    >>> xv, yv, zc = meshgrid(x,y,z)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> zc
    5

    >>> # 2D slice of a 3D grid, with x=const:
    >>> meshgrid(2,y,x)
    (2, array([[ 0.,  1.],
           [ 0.,  1.],
           [ 0.,  1.]]), array([[ 0. ,  0. ],
           [ 0.5,  0.5],
           [ 1. ,  1. ]]))
    >>> meshgrid(0,1,5, sparse=True)  # just a 3D point
    (0, 1, 5)
    >>> meshgrid(y)      # 1D grid; y is just returned
    array([ 0.,  1.])
    >>> meshgrid(x,y, indexing='ij')  # change to matrix indexing
    (array([[ 0. ,  0. ],
           [ 0.5,  0.5],
           [ 1. ,  1. ]]), array([[ 0.,  1.],
           [ 0.,  1.],
           [ 0.,  1.]]))

    Why does SciTools has its own meshgrid function when numpy has
    three similar functions, ``mgrid``, ``ogrid``, and ``meshgrid``?
    The ``meshgrid`` function in numpy is limited to two dimensions
    only, while the SciTools version can also work with 3D and 1D
    grids. In addition, the numpy version of ``meshgrid`` has no
    option for generating sparse grids to conserve memory, like we
    have in SciTools by specifying the ``sparse`` argument.

    Moreover, the numpy functions ``mgrid`` and ``ogrid`` does provide
    support for, respectively, full and sparse n-dimensional
    meshgrids, however, these functions uses slices to generate the
    meshgrids rather than one-dimensional coordinate arrays such as in
    Matlab. With slices, the user does not have the option to generate
    meshgrid with, e.g., irregular spacings, like::

    >>> x = array([-1,-0.5,1,4,5], float)
    >>> y = array([0,-2,-5], float)
    >>> xv, yv = meshgrid(x, y, sparse=False)

    >>> xv
    array([[-1. , -0.5,  1. ,  4. ,  5. ],
           [-1. , -0.5,  1. ,  4. ,  5. ],
           [-1. , -0.5,  1. ,  4. ,  5. ]])

    >>> yv
    array([[ 0.,  0.,  0.,  0.,  0.],
           [-2., -2., -2., -2., -2.],
           [-5., -5., -5., -5., -5.]])


    In addition to the reasons mentioned above, the ``meshgrid``
    function in numpy supports only Cartesian indexing, i.e., x and y,
    not matrix indexing, i.e., rows and columns (on the other hand,
    ``mgrid`` and ``ogrid`` supports only matrix indexing). The
    ``meshgrid`` function in SciTools supports both indexing
    conventions through the ``indexing`` keyword argument. Giving the
    string ``'ij'`` returns a meshgrid with matrix indexing, while
    ``'xy'`` returns a meshgrid with Cartesian indexing. The
    difference is illustrated by the following code snippet::

      nx = 10
      ny = 15

      x = linspace(-2,2,nx)
      y = linspace(-2,2,ny)

      xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
      for i in range(nx):
          for j in range(ny):
              # treat xv[i,j], yv[i,j]

      xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
      for i in range(nx):
          for j in range(ny):
              # treat xv[j,i], yv[j,i]

    It is not entirely true that matrix indexing is not supported by the
    ``meshgrid`` function in numpy because we can just switch the order of
    the first two input and output arguments::

    >>> yv, xv = numpy.meshgrid(y, x)
    >>> # same as:
    >>> xv, yv = meshgrid(x, y, indexing='ij')

    However, we think it is clearer to have the logical "x, y"
    sequence on the left-hand side and instead adjust a keyword argument.
    """

    import types
    def fixed(coor):
        return isinstance(coor, (float, complex, int, types.NoneType))

    if not fixed(x):
        x = np.asarray(x)
    if not fixed(y):
        y = np.asarray(y)
    if not fixed(z):
        z = np.asarray(z)

    def arr1D(coor):
        try:
            if len(coor.shape) == 1:
                return True
            else:
                return False
        except AttributeError:
            return False

    # if two of the arguments are fixed, we have a 1D grid, and
    # the third argument can be reused as is:

    if arr1D(x) and fixed(y) and fixed(z):
        return x
    if fixed(x) and arr1D(y) and fixed(z):
        return y
    if fixed(x) and fixed(y) and arr1D(z):
        return z

    # if x,y,z are identical, make copies:
    try:
        if y is x: y = x.copy()
        if z is x: z = x.copy()
        if z is y: z = y.copy()
    except AttributeError:  # x, y, or z not numpy array
        pass

    if memoryorder is not None:
        import warnings
        msg = "Keyword argument 'memoryorder' is deprecated and will be " \
              "removed in the future. Please use the 'indexing' keyword " \
              "argument instead."
        warnings.warn(msg, DeprecationWarning)
        if memoryorder == 'xyz':
            indexing = 'ij'
        else:
            indexing = 'xy'

    # If the keyword argument sparse is set to False, the full N-D matrix
    # (not only the 1-D vector) should be returned. The mult_fact variable
    # should then be updated as necessary.
    mult_fact = 1

    # if only one argument is fixed, we have a 2D grid:
    if arr1D(x) and arr1D(y) and fixed(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = np.ones((len(x),len(y)))
            if z is None:
                return x[:,np.newaxis]*mult_fact, y[np.newaxis,:]*mult_fact
            else:
                return x[:,np.newaxis]*mult_fact, y[np.newaxis,:]*mult_fact, z
        else:
            if not sparse:
                mult_fact = np.ones((len(y),len(x)))
            if z is None:
                return x[np.newaxis,:]*mult_fact, y[:,np.newaxis]*mult_fact
            else:
                return x[np.newaxis,:]*mult_fact, y[:,np.newaxis]*mult_fact, z

    if arr1D(x) and fixed(y) and arr1D(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = np.ones((len(x),len(z)))
            if y is None:
                return x[:,np.newaxis]*mult_fact, z[np.newaxis,:]*mult_fact
            else:
                return x[:,np.newaxis]*mult_fact, y, z[np.newaxis,:]*mult_fact
        else:
            if not sparse:
                mult_fact = np.ones((len(z),len(x)))
            if y is None:
                return x[np.newaxis,:]*mult_fact, z[:,np.newaxis]*mult_fact
            else:
                return x[np.newaxis,:]*mult_fact, y, z[:,np.newaxis]*mult_fact

    if fixed(x) and arr1D(y) and arr1D(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = np.ones((len(y),len(z)))
            if x is None:
                return y[:,np.newaxis]*mult_fact, z[np.newaxis,:]*mult_fact
            else:
                return x, y[:,np.newaxis]*mult_fact, z[np.newaxis,:]*mult_fact
        else:
            if not sparse:
                mult_fact = np.ones((len(z),len(y)))
            if x is None:
                return y[np.newaxis,:]*mult_fact, z[:,np.newaxis]*mult_fact
            else:
                return x, y[np.newaxis,:]*mult_fact, z[:,np.newaxis]*mult_fact

    # or maybe we have a full 3D grid:
    if arr1D(x) and arr1D(y) and arr1D(z):
        if indexing == 'ij':
            if not sparse:
                mult_fact = np.ones((len(x),len(y),len(z)))
            return x[:,np.newaxis,np.newaxis]*mult_fact, \
                   y[np.newaxis,:,np.newaxis]*mult_fact, \
                   z[np.newaxis,np.newaxis,:]*mult_fact
        else:
            if not sparse:
                mult_fact = np.ones((len(y),len(x),len(z)))
            return x[np.newaxis,:,np.newaxis]*mult_fact, \
                   y[:,np.newaxis,np.newaxis]*mult_fact, \
                   z[np.newaxis,np.newaxis,:]*mult_fact

    # at this stage we assume that we just have scalars:
    l = []
    if x is not None:
        l.append(x)
    if y is not None:
        l.append(y)
    if z is not None:
        l.append(z)
    if len(l) == 1:
        return l[0]
    else:
        return tuple(l)
