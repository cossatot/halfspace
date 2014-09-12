import numpy as np
import scipy.signal as sc
from scipy.fftpack import fftn, ifftn
#import pyfftw

#fftn = pyfftw.interfaces.scipy_fftpack.fftn
#ifftn = pyfftw.interfaces.scipy_fftpack.ifftn

""" Formulations from Liu and Zoback, 1992 JGR.  Equations numbers from that
reference are in parentheses.  'lamb' == lambda, because 'lambda' is a python
keyword

Boussonesq (vertical load)

(68) b_stress_xx = (Fv / 2 pi) * [ (3x**2 * z/ r**5) + (mu * (y**2 + z **2) )
                    / ( (lambda + mu) * r**3 * (z + r) )
                    - (mu * z) / ( (lambda + r) * r **3) - (mu * z) /
                    ( (lambda + mu) * r **2 * (z + r) ) ]

(69) b_stress_yy = (Fv / 2 pi) * [ (3y**2 * z/ r**5) + (mu * (x**2 + z **2) )
                    / ( (lambda + mu) * r**3 * (z + r) )
                     - (mu * z) / ( (lambda + r) * r **3) - (mu * z) /
                     ( (lambda + mu) * r **2 * (z + r) ) ]

(70) b_stress_zz = 3 * Fv * z**3 / (2 * pi * r **5)

(71) b_stress_yz = 3 * Fv * y * z**2 / (2 * pi * r **5)

(72) b_stress_xz = 3 * Fv * x * z**2 / (2 * pi * r **5)

(73) b_stress_xy = Fv / (2 * pi) * [ (3 * x * y * z / r **5) -
                    (mu * x * y * (z + 2 * r) /
                    ( (lambda + mu) * r **3 * (z + r) **2) ) ]

"""

def get_r( x, y = 0, z = 1):
    """Makes r (length between origin and point) for x, y, z

    Returns r (scalar or array, depending on inputs)"""

    r = ( x **2 + y **2 + z **2 ) **0.5

    return r


""" Boussinesq stresses Green's functions """

def calc_b_stress_xx( x, y = 0, z = 1, Fv = 1, mu = 1, lamb = 1 ):
    """
    Boussonesq solution for stresses acting on x in the x direction,
    from Liu and Zoback 1992 JGR (equation 68)
    """

    r = get_r( x, y, z)

    term1 = Fv / (2 * np.pi)

    term2 = 3 * x **2 * z / r **5

    term3 = mu * ( y **2 + z **2) / ( (lamb + mu) * (z + r) * r **3 )

    term4 = (mu * z) / ( (lamb + mu) * r **3)

    term5 = (mu * x **2) / ( (lamb + mu) * r **2 * (z + r) **2 )

    b_stress_xx = term1 * (term2 + term3 - term4 - term5)

    return b_stress_xx


def calc_b_stress_yy( x, y, z = 1, Fv = 1, mu = 1, lamb = 1 ):
    """
    Boussonesq solution for stresses acting on y in the y direction,
    from Liu and Zoback 1992 JGR (equation 69)
    """

    r = get_r( x, y, z)

    term1 = Fv / (2 * np.pi)

    term2 = 3 * y **2 * z / r **5

    term3 = mu * ( x **2 + z **2) / ( (lamb + mu) * (z + r) * r **3 )

    term4 = (mu * z) / ( (lamb + mu) * r **3)

    term5 = (mu * y **2) / ( (lamb + mu) * r **2 * (z + r) **2 )

    b_stress_yy = term1 * (term2 + term3 - term4 - term5)

    return b_stress_yy


def calc_b_stress_zz( x, y = 0, z = 1, Fv = 1 ):
    """
    Boussonesq solution for stresses acting on z in the z direction,
    from Liu and Zoback 1992 JGR (equation 70)
    """

    r = get_r( x, y, z)

    b_stress_zz = 3 * Fv * z **3 / (2 * np.pi * r **5)

    return b_stress_zz


def calc_b_stress_yz( x, y, z = 1, Fv = 1 ):
    """
    Boussonesq solution for stresses acting on y in the z direction
    (or vice versa), from Liu and Zoback 1992 JGR (equation 71)
    """

    r = get_r( x, y, z)

    b_stress_yz = 3 * Fv * y * z **2 / (2 * np.pi * r **5)

    return b_stress_yz


def calc_b_stress_xz( x, y = 0, z = 1, Fv = 1 ):
    """
    Boussonesq solution for stresses acting on x in the z direction
    (or vice versa) from Liu and Zoback 1992 JGR (equation 72)
    """

    r = get_r( x, y, z)

    b_stress_xz = 3 * Fv * x * z **2 / (2 * np.pi * r **5)

    return b_stress_xz


def calc_b_stress_xy( x, y, z = 1, Fv = 1, mu = 1, lamb = 1):
    """
    Boussonesq solution for stresses acting on x in the y direction
    (or vice versa) from Liu and Zoback 1992 JGR (equation 73)
    """

    r = get_r( x, y, z)

    term1 = Fv / (2 * np.pi)

    term2 = (3 * x * y * z / r **5)

    term3 = mu * x * y * (z + 2 * r) / ( (lamb + mu) * r **3 * (z + r) **2)

    b_stress_xy = term1 * (term2 - term3)

    return b_stress_xy


""" Boussinesq Green's function kernel constructors"""

def make_b_kernel_2d( component = None, z = 1, Fv = 1, kernel_radius = 100,
                     kernel_res = 1, mu = 1, lamb = 1, circular = True):
    """ 
    Makes a 2d horizontal meshgrid of the Boussinesq stress load for 2d
    convolution.
    """

    kernel_len = kernel_radius * 2 / kernel_res + 1
    kernel_len = int( kernel_len)

    x = np.linspace( -kernel_radius, kernel_radius, num=kernel_len)
    y = x.copy()

    xx, yy = np.meshgrid( x, y)

    conv_kernel = _get_b_kernel_2d( component = component, x = xx, y = yy,
                                   z = z, Fv = Fv, mu = mu, lamb = lamb)

    # scale for kernel resolution
    conv_kernel *= kernel_res **2

    if circular == True:
        circle = np.sqrt(xx **2 + yy **2)
        kernel_mask = circle <= kernel_radius
        conv_kernel *= kernel_mask

    return conv_kernel


def _get_b_kernel_2d( component = None, x = None, y = None, z = None,
                     Fv = None, mu = None, lamb = None):
    """ 
    Calculates the approprate Green's function on the grid given
    the components and the stress component.
    """

    if component == 'xx':
        conv_kernel = calc_b_stress_xx( x = x, y = y, z = z, Fv = Fv,
                                       mu = mu, lamb = lamb)
    elif component == 'xy':
        conv_kernel = calc_b_stress_xy( x = x, y = y, z = z, Fv = Fv,
                                       mu = mu, lamb = lamb)
    elif component == 'yy':
        conv_kernel = calc_b_stress_yy( x = x, y = y, z = z, Fv = Fv,
                                       mu = mu, lamb = lamb)
    elif component == 'xz':
        conv_kernel = calc_b_stress_xz( x = x, y = y, z = z, Fv = Fv)

    elif component == 'yz':
        conv_kernel = calc_b_stress_yz( x = x, y = y, z = z, Fv = Fv)
    
    elif component == 'zz':
        conv_kernel = calc_b_stress_zz( x = x, y = y, z = z, Fv = Fv)

    else:
        raise Exception('stress component not specified or supported')

    return conv_kernel



""" Cerruti stress Green's functions """

"""Cerruti functions for horizontal load in +x direction """
def calc_c_stress_xx_x( x, y = 0, z=1, Fh = 1, mu = 1, lamb = 1):
    """
    Cerruti solutions for horizontal stresses (in x direction) acting on x in
    the x direction, from Liu and Zoback 1992 JGR (equation 77)
    """

    r = get_r( x, y, z)

    term1 = Fh * x / (2 * np.pi * r **3 )

    term2 = 3 * x **2 / r **2

    term3 = mu / ( (lamb + mu) * (z + r) **2 )

    term4 = r **2 - y **2 - (2 * r * y **2 / ( r + z ) )

    c_stress_xx_x = term1 * ( term2 - term3 * term4 )

    return c_stress_xx_x


def calc_c_stress_yy_x( x, y, z = 1, Fh = 1, mu = 1, lamb = 1):
    """
    Cerruti solutions for horizontal stresses (in x direction) acting on y in
    the y direction, from Liu and Zoback 1992 JGR (equation 78)
    """

    r =  get_r( x, y, z)

    term1 = Fh * x / (2 * np.pi * r **3 )

    term2 = 3 * y **2 / r **2

    term3 = mu / ( (lamb + mu) * (z + r) **2 )

    term4 = 3 * r **2 - x **2 - (2 * r * x **2 / ( r + z) )

    c_stress_yy_x = term1 * ( term2 - term3 * term4 )

    return c_stress_yy_x


def calc_c_stress_xy_x( x, y, z = 1, Fh = 1, mu = 1, lamb = 1):
    """
    Cerruti solutions for horizontal stresses (in x direction) acting on x in
    the y direction, from Liu and Zoback 1992 JGR (equation 79)
    """

    r = get_r( x, y, z)

    term1 = Fh * y / (2 * np.pi * r **3 )

    term2 = 3 * x **2 / r **2

    term3 = mu / ( (lamb + mu) * (z + r) **2 )

    term4 = r **2 - x **2 - (2 * r * x **2 / ( r + z ) )

    c_stress_xy_x = term1 * ( term2 + term3 * term4 )

    return c_stress_xy_x


def calc_c_stress_zz_x( x, y = 0, z = 1, Fh = 1):
    """Cerruti solutions for horizontal stresses (in x direction) acting on z
    in the z direction, from Liu and Zoback 1992 JGR (equation 80)
    """

    r = get_r( x, y, z)

    c_stress_zz_x = 3 * Fh * x * z **2 / (2 * np.pi * r **5)

    return c_stress_zz_x


def calc_c_stress_yz_x( x, y, z = 1, Fh = 1):
    """Cerruti solutions for horizontal stresses (in x direction) acting on y
    in the z direction, from Liu and Zoback 1992 JGR (equation 81)
    """

    r = get_r( x, y, z)

    c_stress_yz_x = 3 * Fh * x * y * z / (2 * np.pi * r **5)

    return c_stress_yz_x


def calc_c_stress_xz_x( x, y = 0, z = 1, Fh = 1):
    """Cerruti solutions for horizontal stresses (in x direction) acting on x
    in the z direction, from Liu and Zoback 1992 JGR (equation 82)
    """

    r = get_r( x, y, z)

    c_stress_xz_x = 3 * Fh * x **2 * z / (2 * np.pi * r **5)

    return c_stress_xz_x


""" flipped Cerruti functons: for horizontal load in +y direction"""

def calc_c_stress_yy_y( x, y = 0, z=1, Fh = 1, mu = 1, lamb = 1):
    """
    Cerruti solutions for horizontal stresses (in y direction) acting on y in
    the y direction, from Liu and Zoback 1992 JGR (equation 77).

    x and y are flipped vs. the published function; this is the main d_topo/ dy
    correction.
    """

    r = get_r( x, y, z)

    term1 = Fh * y / (2 * np.pi * r **3 )

    term2 = 3 * y **2 / r **2

    term3 = mu / ( (lamb + mu) * (z + r) **2 )

    term4 = r **2 - x **2 - (2 * r * x **2 / ( r + z ) )

    c_stress_yy_y = term1 * ( term2 - term3 * term4 )

    return c_stress_yy_y


def calc_c_stress_xx_y( x, y, z = 1, Fh = 1, mu = 1, lamb = 1):
    """
    Cerruti solutions for horizontal stresses (in x direction) acting on x in
    the x direction, from Liu and Zoback 1992 JGR (equation 78).

    x and y are flipped vs. the published function; this is the main d_topo/ dy
    correction.
    """

    r =  get_r( x, y, z)

    term1 = Fh * y / (2 * np.pi * r **3 )

    term2 = 3 * x **2 / r **2

    term3 = mu / ( (lamb + mu) * (z + r) **2 )

    term4 = 3 * r **2 - y **2 - (2 * r * y **2 / ( r + z) )

    c_stress_xx_y = term1 * ( term2 - term3 * term4 )

    return c_stress_xx_y


def calc_c_stress_xy_y( x, y, z = 1, Fh = 1, mu = 1, lamb = 1):
    """
    Cerruti solutions for horizontal stresses (in x direction) acting on x in
    the y direction, from Liu and Zoback 1992 JGR (equation 79).

    x and y are flipped vs. the published function; this is the main d_topo/ dy
    correction.
    """

    r = get_r( x, y, z)

    term1 = Fh * x / (2 * np.pi * r **3 )

    term2 = 3 * y **2 / r **2

    term3 = mu / ( (lamb + mu) * (z + r) **2 )

    term4 = r **2 - y **2 - (2 * r * y **2 / ( r + z ) )

    c_stress_xy_y = term1 * ( term2 + term3 * term4 )

    return c_stress_xy_y


def calc_c_stress_zz_y( x, y = 0, z = 1, Fh = 1):
    """Cerruti solutions for horizontal stresses (in x direction) acting on z
    in the z direction, from Liu and Zoback 1992 JGR (equation 80)
    """

    r = get_r( x, y, z)

    c_stress_zz_y = 3 * Fh * y * z **2 / (2 * np.pi * r **5)

    return c_stress_zz_y


def calc_c_stress_xz_y( x, y, z = 1, Fh = 1):
    """Cerruti solutions for horizontal stresses (in x direction) acting on y
    in the z direction, from Liu and Zoback 1992 JGR (equation 81)
    """

    r = get_r( x, y, z)

    c_stress_xz_y = 3 * Fh * x * y * z / (2 * np.pi * r **5)

    return c_stress_xz_y


def calc_c_stress_yz_y( x, y = 0, z = 1, Fh = 1):
    """Cerruti solutions for horizontal stresses (in x direction) acting on x
    in the z direction, from Liu and Zoback 1992 JGR (equation 82)
    """

    r = get_r( x, y, z)

    c_stress_xz_y = 3 * Fh * y **2 * z / (2 * np.pi * r **5)

    return c_stress_xz_y



""" Cerruti Green's function kernel constructors """

""" For stresses in the +x direction """

def make_c_kernel_2d( component = None, z = 1, Fh = 1, kernel_radius = 100,
                     kernel_res = 1, mu = 1, lamb = 1, circular = True, 
                     f_dir = None):
    """ 
    Makes a 2d horizontal meshgrid of the Cerruti stress load for 2d
    convolution.
    """

    kernel_len = kernel_radius * 2 / kernel_res + 1
    kernel_len = int( kernel_len)

    x = np.linspace( -kernel_radius, kernel_radius, num=kernel_len)
    y = x.copy()

    xx, yy = np.meshgrid( x, y)

    conv_kernel = _get_c_kernel_2d( component = component, x = xx, y = yy,
                                   z = z, Fh = Fh, mu = mu, lamb = lamb,
                                   f_dir = f_dir)

    # scale for kernel resolution
    conv_kernel *= kernel_res **2

    if circular == True:
        circle = np.sqrt(xx **2 + yy **2)
        kernel_mask = circle <= kernel_radius
        conv_kernel *= kernel_mask

    return conv_kernel


def _get_c_kernel_2d( component = None, x = None, y = None, z = None,
                     Fh = None, mu = None, lamb = None, f_dir = None):
    """    
    Calculates the approprate Green's function on the grid given
    the components and the stress component.
    """
    # xx
    if component == 'xx' and f_dir == 'x':
        conv_kernel = calc_c_stress_xx_x( x = x, y = y, z = z, Fh = Fh,
                                       mu = mu, lamb = lamb)

    elif component == 'xx' and f_dir == 'y':
        conv_kernel = calc_c_stress_xx_y( x = x, y = y, z = z, Fh = Fh,
                                       mu = mu, lamb = lamb)
    #xy
    elif component == 'xy' and f_dir == 'x':
        conv_kernel = calc_c_stress_xy_x( x = x, y = y, z = z, Fh = Fh,
                                       mu = mu, lamb = lamb)

    elif component == 'xy' and f_dir == 'y':
        conv_kernel = calc_c_stress_xy_y( x = x, y = y, z = z, Fh = Fh,
                                       mu = mu, lamb = lamb)
    #yy
    elif component == 'yy' and f_dir == 'x':
        conv_kernel = calc_c_stress_yy_x( x = x, y = y, z = z, Fh = Fh,
                                       mu = mu, lamb = lamb)

    elif component == 'yy' and f_dir == 'y':
        conv_kernel = calc_c_stress_yy_y( x = x, y = y, z = z, Fh = Fh,
                                       mu = mu, lamb = lamb)
    #xz
    elif component == 'xz' and f_dir == 'x':
        conv_kernel = calc_c_stress_xz_x( x = x, y = y, z = z, Fh = Fh)

    elif component == 'xz' and f_dir == 'y':
        conv_kernel = calc_c_stress_xz_y( x = x, y = y, z = z, Fh = Fh)

    #zz
    elif component == 'zz' and f_dir == 'x':
        conv_kernel = calc_c_stress_zz_x( x = x, y = y, z = z, Fh = Fh)

    elif component == 'zz' and f_dir == 'y':
        conv_kernel = calc_c_stress_zz_y( x = x, y = y, z = z, Fh = Fh)

    #yz
    elif component == 'yz' and f_dir == 'x':
        conv_kernel = calc_c_stress_yz_x( x = x, y = y, z = z, Fh = Fh)

    elif component == 'yz' and f_dir == 'y':
        conv_kernel = calc_c_stress_yz_y( x = x, y = y, z = z, Fh = Fh)

    else:
        raise Exception('stress component not specified or supported')

    return conv_kernel



""" Functions to convolve loads and Green's function kernels """

def _centered(arr, newsize):
    # Return the center newsize portion of the array
    # copied from scipy.signal (c) Travis Oliphant, 1999-2002
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) / 2
    endind = startind + newsize
    myslice = [ slice( startind[k], endind[k]) for k in range( len(endind) ) ]
    return arr[tuple(myslice)]


def half_fft_convolve(in1, in2, size, mode = 'full', return_type='real'):
    """
    Rewrite of fftconvolve from scipy.signal ((c) Travis Oliphant 1999-2002)
    to deal with fft convolution where one signal is not fft transformed
    and the other one is.  Application is, for example, in a loop where
    convolution happens repeatedly with different kernels over the same
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
    if return_type == 'real':
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


"""Boussinesq convolution functions"""


def do_b_convo( component = None, z = 1, Fv = 1, load = None, lamb = 1, mu = 1,
               kernel_radius = None, kernel_res = None, load_mode = 'topo',
               size = None, conv_mode = 'valid', circular = True):

    kernel = make_b_kernel_2d( component = component, z = z, Fv = Fv, mu = mu,
                              lamb = lamb, kernel_radius = kernel_radius,
                              kernel_res = kernel_res, circular = circular)

    if load_mode == 'topo':
        b_stress_out = sc.fftconvolve(load, kernel, mode = conv_mode)

    elif load_mode == 'fft':
        b_stress_out = half_fft_convolve(load, kernel, size, mode = conv_mode)
    
    else:
        raise Exception('load mode not specified or supported')

    return b_stress_out


def do_c_convo( component = None, f_dir = None, z = 1, Fh = 1, lamb = 1,
               mu = 1, load = None, kernel_radius = None, kernel_res = None,
               conv_mode = 'same', load_mode = 'topo', circular = True,
               size = None):

    kernel = make_c_kernel_2d( component = component, z = z, Fh = Fh, mu = mu,
                              lamb = lamb, circular = circular, f_dir = f_dir,
                              kernel_radius = kernel_radius,
                              kernel_res = kernel_res)

    if load_mode == 'topo':
        c_stress_out = sc.fftconvolve(load, kernel, mode = conv_mode)

    elif load_mode == 'fft':
        c_stress_out = half_fft_convolve(load, kernel, size, conv_mode)

    else:
        raise Exception('load mode not specified or supported')

    return c_stress_out

