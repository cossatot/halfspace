from __future__ import division
import numpy as np
from scipy.stats import linregress

"""
All code (c) Richard Styron, 2013 unless otherwise noted.  All rights reserved.
"""

def get_norm_vector_from_sd(strike, dip, angle='degrees', out_format='matrix'):
    """ 
    Takes strike, dip input (in degrees by default, right hand rule)
    and returns unit normal vector in X,Y,Z = East, North, Down
    coordinates.  Vector points towards surface/hanging wall.

    Set angle = 'radians' to input coords in radians (still from north=0)

    Returns (3,1) numpy matrix: [n_E, n_N, n_D]
    """

    if angle == 'degrees':
        strike_rad = np.deg2rad(strike)
        dip_rad = np.deg2rad(dip)
    elif angle == 'radians':
        strike_rad = strike
        dip_rad = dip
    
    nE =  np.sin( dip_rad) * np.cos( strike_rad)
    nN = -np.sin( dip_rad) * np.sin( strike_rad)
    nD = -np.cos( dip_rad)
    
    if out_format == 'matrix':
        return  np.matrix([nE, nN, nD])
    elif out_format == 'array':
        return np.array([nE, nN, nD])
    elif out_format == 'list':
        return [nE, nN, nD]


def get_sd_from_norm_vec( norm_vec, output='degrees', in_format = 'array'):
    """
    Takes normal vector to plane in [E, N, D] format and 
    calculates strike and dip in right hand rule.

    Returns strike, dip
    """
    if in_format == 'matrix':
        norm_vec = np.array(norm_vec)

    E = norm_vec[0]
    N = norm_vec[1]
    D = norm_vec[2]

    vec_len = np.sqrt(E**2 + N**2 + D**2)  # normalize vector to unit length
    E = E / vec_len
    N = N / vec_len
    D = D / vec_len

    dip = np.arccos(-D)
    sin_dip = np.sin(dip)

    strike_cos = np.arccos( E / sin_dip )
    strike_sin = np.arcsin( N / -sin_dip)
    
    # fix to solve for integer ambiguity in the trig inverse results
    strike = strike_cos.copy()
    
    if np.isscalar(strike):
        if np.isnan(strike):
            strike = 0.
        if strike_sin < 0.:
            strike += 2 * (np.pi - strike)
    else:
        strike[np.isnan(strike)] = 0.
        strike[strike_sin < 0] += 2 * (np.pi - strike[strike_sin < 0])

    if np.isscalar(strike):
        if dip > np.pi/2.:
            dip = np.pi - dip
            strike = strike - np.pi if strike > np.pi else strike + np.pi

    #TODO: add array boolean to get strikes correct for dips > pi/2

    if output == 'degrees':
        strike, dip = np.rad2deg( (strike, dip) )
    return strike, dip


def get_strike_vector(strike, angle='degrees'):
    """
    Gets vector that points in direction of strike in right-hand-rule
    convention.

    Returns [3,1] matrix[n_E, n_N, n_D = 0] (horizontal...)
    """
    if angle == 'degrees':
        strike_rad = np.deg2rad(strike)
    elif angle == 'radians':
        strike_rad = strike
    
    return np.matrix([np.sin( strike_rad), np.cos( strike_rad), 0])
    

def get_dip_vector(strike = None, dip = None, angle='degrees'):
    """
    Gets vector that points down dip in XYZ/East, North, Down convention.

    Returns [3,1] matrix[E, N, D = 0]
    """
    norm = get_norm_vector_from_sd(strike, dip, angle=angle)
    s = get_strike_vector(strike, angle=angle)
    
    return np.cross(norm, s)


def get_rake_from_shear_components(strike_shear = 0, dip_shear = 0,
                                   angle = 'degrees'):
    """
    Takes components of fault shear (strike or dip) and returns
    rake in Aki and Richards convention (default units are degrees).

    Specify angle='radians' to get output in radians.
    """
    
    rake = np.arctan2(dip_shear, -strike_shear)
    if angle == 'degrees':
        rake = np.degrees(rake)

    return rake


def normal_stress_from_xyz(strike = None, dip = None, stress_tensor = None, 
                           angle = 'degrees'):
    """
    Takes a plane orientation (in strike, dip) and a stress tensor 
    (in XYZ/END coords) and calculates the normal stress on the plane.

    Returns scalar stress value (float)
    """
    N = get_norm_vector_from_sd( strike, dip, angle)

    T = stress_tensor

    return np.float( N * T * N.T)


def norm_stress_from_xyz(strike = None, dip = None, stress_tensor = None, 
                         angle = 'degrees'):
    """ Deprecated name for 'normal_stress_from_xyz"""

    normal_stress =  normal_stress_from_xyz(strike = strike, dip = dip, 
                                            stress_tensor = stress_tensor, 
                                            angle = angle)
    return normal_stress


def dip_shear_stress_from_xyz(strike = None, dip = None, stress_tensor = None,
                              angle = 'degrees'):
    """
    Takes a plane orientation (in strike, dip) and a stress tensor
    (in XYZ/END coords) and calculates the down-dip shear stress on the
    plane.  Positive shear stress means the upper side of the plane
    (e.g. the hanging wall) moves up, i.e. reverse-sense shear stress.

    Returns scalar stress value (float).
    """
    N = get_norm_vector_from_sd( strike, dip, angle)
    D = get_dip_vector( strike, dip, angle)

    T = stress_tensor

    return np.float( D * T * N.T )


def strike_shear_stress_from_xyz(strike = None, dip = None,
                                 stress_tensor = None, angle = 'degrees'):
    """
    Takes a plane orientation (in strike, dip) and a stress tensor
    (in XYZ/END coords) and calculates the along-strike shear stress on the
    plane.  Positive shear stress means right-lateral shear.

    Returns scalar stress value (float).
    """
    N = get_norm_vector_from_sd( strike, dip, angle)
    S = get_strike_vector( strike, angle)

    T = stress_tensor

    return np.float( S * T * N.T)


def max_shear_stress_from_xyz(strike = None, dip = None, stress_tensor = None,
                              angle = 'degrees'):
    """
    Takes a plane orientation (in strike, dip) and a stress tensor
    (in XYZ/END coords) and calculates the maximum shear stress on the
    plane, as well as the rake of the maximum shear stress value.

    Returns len(2) tuple, stress magnitude and rake (-180-180).
    """
    T = stress_tensor

    tau_ss = strike_shear_stress_from_xyz(strike, dip, stress_tensor = T,
                                           angle = angle)

    tau_dd = dip_shear_stress_from_xyz(strike, dip, stress_tensor = T,
                                        angle = angle)

    tau_max = (tau_ss **2 + tau_dd **2) **0.5
    
    tau_rake = get_rake_from_shear_components(strike_shear=tau_ss,
                                              dip_shear=tau_dd, angle=angle)    
    return [tau_max, tau_rake]


def coulomb_shear_stress_from_xyz(strike = None, dip = None, 
                                  stress_tensor = None, friction = 0.6, 
                                  pressure = 0, angle = 'degrees'):
    """
    Calculates the Coulomb shear stress on the fault:
        
    tau_cs = tau_max - friction * (sig_nn - pressure) # Stein 1999 Nature
       
    Returns scalar stress (float)
    """
    T = stress_tensor
    tau_max = max_shear_stress_from_xyz(strike, dip, T, angle)
    sig_nn = norm_stress_from_xyz(strike, dip, T, angle)
    
    tau_cs = tau_max[0] - friction * (sig_nn - pressure)
    
    return tau_cs


def shear_stress_on_optimal_plane(T, friction_angle = 30, 
                                  friction_coefficent = None):
    """
    Calculates shear stress on optimal fault plane, given a stress tensor T.
    
    Returns scalar.
    """

    strike, dip = find_optimal_plane(T, friction_angle, friction_coefficent)

    return max_shear_stress_from_xyz(strike, dip, T)


def normal_stress_on_optimal_plane(T, friction_angle = 30,
                                   friction_coefficient = None):
    """
    Calculates normal stress on optimal fault plane, given a stress tensor T.

    Returns scalar.
    """
    
    strike, dip = find_optimal_plane(T, friction_angle, friction_coefficient)

    return normal_stress_from_xyz(strike, dip, T)


def find_optimal_plane(T, friction_angle=None, friction_coefficient=None,
                       angle_input='degrees', output_normal_vector=False):
    '''
    docs2
    '''

    vals, vecs = sorted_eigens(T)

    R = -(vecs.T)

    beta = get_optimal_fault_angle(friction_angle=friction_angle,
                                   friction_coefficient=friction_coefficient,
                                   angle_input=angle_input, output='radians')

    opt_plane_normal_vec_rot = np.array( [np.cos(beta), 0., np.sin(beta)] )

    opt_plane_normal_vec = R.T.dot(opt_plane_normal_vec_rot)

    if output_normal_vector == True:
        return opt_plane_normal_vec

    else:
        opt_strike, opt_dip = get_sd_from_norm_vec( opt_plane_normal_vec)
        return opt_strike, opt_dip


def get_optimal_fault_angle(friction_coefficient=None, friction_angle=None,
                            angle_input='degrees', output='degrees'):
    '''
    Returns the angle of the optimal fault plane from \sigma_1 (in the 
    plane defined by \sigma_1 and \sigma_3), given the friction coefficient
    or friction angle.

    Equivalent to \beta = (arctan( 1/ \mu)) /2 from King, Stein, Lin, 1994.

    Returns a scalar or array (float) of the input size.
    '''

    if friction_coefficient == None and friction_angle == None:
        raise Exception('Need to specify friction angle or coefficient!')

    if friction_angle == None:
        friction_angle = get_friction_angle(friction_coefficient,
                                            output=angle_input)
    if angle_input in ['degrees', 'deg']:
        friction_angle = np.radians(friction_angle)
    elif angle_input in ['radians', 'rad']:
        pass
    else:
        raise Exception('angle_input needs to be in degrees or radians')

    optimal_fault_angle = (np.pi/2. - friction_angle) / 2.

    if output in ['degrees', 'deg']:
        optimal_fault_angle = np.degrees(optimal_fault_angle)
    elif output in ['radians', 'rad']:
        pass
    else:
        raise Exception('output needs to be degrees or radians')

    return optimal_fault_angle


def get_friction_angle(friction_coefficient, output='degrees'):
    '''
    Takes the coefficient of friction and returns the friction angle.

    Output is by default in degrees.  Specify 'radians' or 'rad' if desired.
    Returns a scalar or vector (float), equal to size of input.
    '''

    friction_angle = np.arctan(friction_coefficient)

    if output in ['degrees', 'deg']:
        friction_angle = np.degrees( friction_angle)

    return friction_angle


def make_xyz_stress_tensor(sig_xx = 0, sig_yy = 0, sig_zz = 0, sig_xy = 0,
                           sig_xz = 0, sig_yz = 0):
    """
    Take stresses and make tensor

    Returns [3x3] matrix.
    """
    
    T = np.matrix([ [ sig_xx,  sig_xy, sig_xz],
                    [ sig_xy,  sig_yy, sig_yz],
                    [ sig_xz,  sig_yz, sig_zz] ])
    return T


def make_xy_stress_tensor(sig_xx = 0, sig_yy = 0, sig_xy = 0):
    """Takes stress components and returns a 2x2 tensor."""

    T = np.matrix([ [ sig_xx, sig_xy],
                    [ sig_xy, sig_yy] ])

    return T


def decomp_xyz_tensor(tensor):
    A = tensor
    sig_xx = A[0,0]
    sig_xy = A[0,1]
    sig_xz = A[0,2]
    sig_yy = A[1,1]
    sig_yz = A[1,2]
    sig_zz = A[2,2]
    
    return [sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz]
    

def xyz_tensor_to_dict(tensor):
    A = tensor
    A_dict = dict([])
    A_dict['xx'] = A[0,0]
    A_dict['xy'] = A[0,1]
    A_dict['xz'] = A[0,2]
    A_dict['yy'] = A[1,1]
    A_dict['yz'] = A[1,2]
    A_dict['zz'] = A[2,2]
    
    return A_dict


def first_tensor_invariant(A):
    """
    Calculates the first tensor invariant of a symmetric 3x3 matrix.

    Returns a scalar.
    """    
    return A[0,0] + A[1,1] + A[2,2]


def second_tensor_invariant(A):
    """
    Calculates the second tensor invariant of a symmetric 3x3 matrix.

    Returns a scalar.
    """
    I2 = ( (A[1,1] * A[2,2]) + (A[2,2] * A[0,0]) + (A[0,0] * A[1,1]) 
          - A[1,2]**2 - A[2,0]**2 - A[0,1]**2 )
    
    return I2


def third_tensor_invariant(A):
    """
    Calculates the third tensor invariant of a summetric 3x3 matrix.

    Returns a scalar.
    """
    term1 = A[0,0] * A[1,1] * A[2,2]
    term2 = 2 * A[0,1] * A[1,2] * A[2,0]
    term3 = A[0,1]**2 * A[2,2] + A[1,2]**2 * A[0,0] + A[2,0]**2 * A[1,1]

    return term1 + term2 - term3


def strike_slip_from_rake_mag(rake, slip_mag = 1., input='degrees'):
    """
    Calculates the strike slip magnitude from slip rake and magnitude.
    Positive values indicate right-lateral slip.  Rake is in Aki and Richards
    convention (0 = right-lateral, 90 = reverse, 180/-180 = left-lateral,
    -90 = normal).

    'input' should be 'degrees'(default) or 'radians', for unit of rake.

    Returns strike slip magnitude (distance) in units of slip_mag input.
    """
    if input=='degrees':
        rake = np.deg2rad( rake)
    elif input == 'radians':
        pass
    else:
        raise Exception('Please specify radians or degrees for input.')

    return -1 * np.cos(rake) * slip_mag
    

def dip_slip_from_rake_mag(rake, slip_mag = 1., input='degrees'):
    """
    Calculates the dip slip magnitude from slip rake and magnitude.
    Positive values indicate reverse slip.  Rake is in Aki and Richards
    convention (0 = right-lateral, 90 = reverse, 180/-180 = left-lateral,
    -90 = normal).

    'input' should be 'degrees'(default) or 'radians', for unit of rake.

    Returns dip slip magnitude (distance) in units of slip_mag input.
    """
    if input=='degrees':
        rake = np.deg2rad( rake)
    elif input == 'radians':
        pass
    else:
        raise Exception('Please specify radians or degrees for input.')

    return np.sin(rake) * slip_mag


def slip_components_from_rake_mag( rake, slip_mag = 1., input='degrees'):
    """
    Calculates the strike and dip slip magnitude from slip rake and magnitude.
    Positive dip slip values indicate reverse slip, and positive strike slip
    values indicate right-lateral slip.  Rake is in Aki and Richards
    convention (0 = right-lateral, 90 = reverse, 180/-180 = left-lateral,
    -90 = normal).

    'input' should be 'degrees'(default) or 'radians', for unit of rake.

    Returns [strike slip, dip slip] magnitude (distance) in units of slip_mag 
    input.

    """
    strike_slip = strike_slip_from_rake_mag(rake, slip_mag, input=input)
    dip_slip = dip_slip_from_rake_mag(rake, slip_mag, input=input)

    return strike_slip, dip_slip


def sorted_eigens(A):
    """ 
    Takes a Hermitian or symmetric matrix and returns the sorted eigenvalues
    and eigenvectors

    Modified from a StackOverflow answer by unutbu

    Returns eigenvalues [vector] and eigenvectors [array]
    """
    eig_vals, eig_vecs = np.linalg.eigh(A)
    idx = eig_vals.argsort()
    
    eig_vals = eig_vals[idx]
    eig_vecs = np.array( eig_vecs[:,idx] )

    return eig_vals, eig_vecs
    
    
def strike2angle(strike, output='radians', input='degrees'):
    """ Takes strike angle (in degrees by default) and changes it into
    unit vector rotation angle (e.g. CCW from x axis/horizontal).

    defaults to output in radians, specify 'degrees' for degrees.

    Returns angle
    """
        
    return azimuth_to_angle(strike, output, input)


def azimuth_to_angle(azimuth, output='radians', input='degrees'):
    """ Takes azimuth (in degrees by default) and changes it into
    unit vector rotation angle (e.g. CCW from x axis/horizontal).

    defaults to output in radians, specify 'degrees' for degrees.

    Returns angle
    """
    if input == 'radians':
        azimuth = np.rad2deg(azimuth)

    angle_deg = (-azimuth) + 90
    angle_rad = np.deg2rad(angle_deg)

    return angle_deg if output == 'degrees' else angle_rad

    
def angle2strike(angle, output='degrees', input='radians'):
    if input=='radians':
        angle = np.rad2deg(angle)
    
    strike = - (angle - 90)
    if strike < 0:
        strike += 360
    if strike == -0:
        strike = 0
    if strike > 360:
        strike -= 360
    
    return strike if output == 'degrees' else np.deg2rad(strike)


def xy_to_azimuth(x1, y1, x0=0, y0=0, output='degrees'):
    """ Calculates the azimuth of a line extending from (x0, y0)
    to (x1, y1).  (x0, y0) defaults to the origin.

    Returns azimuth (0=North, 90=E) by default.  Set output='radians'
    for output in radians, if there is some reason to do so.

    Can operate on scalars or vectors.

    """
    
    rad = np.arctan2( (y1-y0), (x1-x0) )
    
    az = angle_to_azimuth(rad, output=output)
    
    return az


def angle_to_azimuth_scalar(angle):
    """
    Helper function for angle_to_azimuth, for scalar inputs.

    Takes an angle (in unit circle coordinates) and returns an azimuth
    (in compass direction coordinates).
    """
    az = - (angle - 90)
    
    while az < 0:
        az += 360
    while az > 360:
        az -= 360
    
    return az


def angle_to_azimuth_vector(angle):
    """
    Helper function for angle_to_azimuth, for scalar inputs.

    Takes an angle (in unit circle coordinates) and returns an azimuth
    (in compass direction coordinates).
    """
    az = - (angle - 90)
    
    az[az < 0] += 360
    az[az > 360] += 360
    
    return az
    

def angle_to_azimuth(angle, input='radians', output='degrees'):
    """
    Takes an angle (in unit circle coordinates) and returns an azimuth
    (in compass direction coordinates, i.e. N=0 degrees, E=90 degrees).  
    Specify input='degrees' or output='radians' if need be.
    
    Works on scalars, vectors, and Pandas Series.
    """
    if input == 'radians':
        angle = np.rad2deg(angle)
    
    if np.isscalar(angle):
        az = angle_to_azimuth_scalar(angle)
    
    else:
        az = angle_to_azimuth_vector(angle)
        
    if output=='radians':
        az = np.deg2rad(az)
    
    return az


def pts2strike(r_point, l_point, output = 'degrees'):
    """ Takes two (x,y) points and calculates the right-
    hand-rule strike between them, where the right point
    and left points are defined from a perspective looking
    down dip."""
    
    rx, ry = r_point[0], r_point[1]
    lx, ly = l_point[0], l_point[1]
    
    angle = np.arctan2( (ly - ry), (lx - rx) )
    
    strike = angle2strike(angle, output=output)
    
    return strike

    
def get_slope_from_pts(x_pts = None, y_pts = None, fit_type='lsq',
                       output = 'radians', return_intercept = False):
    """
    Takes 2 series of points and finds slope.  Returns scalar.  This
    is to be used when 

    Default is linear least squares fit, but other fits can be
    implemented in the futsure.
    
    TODO:  consider finding a way to determine if there is a way to
    tell direction (dealing with dip dir).  Maybe output 2 values?
    Or did I already take care of the issue in extrude_fault_trace?"""
       
    if fit_type == 'lsq':
        slope, intr, r_val, p_val, err = linregress(x_pts, y_pts)
    else:
        raise Exception('no other fit types implemented')

    if output == 'degrees':
        slope = np.rad2deg(slope)
    
    if return_intercept == False:
        return slope
    
    elif return_intercept == True:
        return slope, intr


def get_strike_from_pts(lon_pts = None, lat_pts = None, fit_type = 'lsq',
                        output = 'degrees'):
    
    """
    Takes 2 series/vectors of points and finds the strike, in right-hand-rule.
    Returns strike (scalar).  Assumes points are all at same elevation.

    Series of points cannot be Pandas Series types; need to input
    'series.values' instead.
    """

    slope, intercept = get_slope_from_pts(lon_pts, lat_pts, 
                                          fit_type=fit_type,
                                          return_intercept = True)

    strike = pts2strike( [lon_pts[-1], lon_pts[-1] * slope + intercept],
                         [lon_pts[0], lon_pts[0] * slope + intercept],
                         output = output)
    return strike


def strike_dip_from_3_xyz(pt1, pt2, pt3, output='degrees'):
    """ 
    Takes 3 points in xyz [east, north, down] 
    and returns strike and dip in radians or degrees, based on 'output'
    parameter.

    Returns: strike, dip
    """
    a = np.matrix([[pt1[0], pt1[1], 1],
                   [pt2[0], pt2[1], 1],
                   [pt3[0], pt3[1], 1]])
    
    z_vec = np.array([-pt1[2], -pt2[2], -pt3[2]])
    
    mx, my, z0 = np.linalg.solve(a, z_vec)
    
    dip = np.arctan( np.sqrt(mx **2 + my **2) )
    
    dip_dir = np.arctan2(mx, my)
    strike_angle = dip_dir + np.pi / 2
    
    strike = angle2strike(strike_angle, input='radians', output = output)
    
    if output == 'degrees':
        dip = np.degrees( dip)
    
    return strike, dip


def extrude_fault_trace(lon_pts = None, lat_pts = None, elev_pts = None, 
                        depth_vec = None, strike = None, dip = None, 
                        h_coords = 'degrees', deg_per_m = None, 
                        dip_input_type = 'degrees', output_shape = 'array'):
    
    """
    Makes 3D point sets of faults, projecting them down dip based on
    the best-fit strike for the (lon, lat) point series.  

    Spacing for the points is determined by the spacing in the
    depth vector (input).

    Returns 3 lat, lon, depth arrays that are 2d (output_shape 'array')
    or 1d (output shape 'vector').

    if pts are pandas series, need to input series.values 
        
    """
    # set some constants

    d_len = len(depth_vec)
    
    if h_coords == 'degrees' and deg_per_m == None:
        deg_per_m = 1/100000.

    elif h_coords == 'm':
        deg_per_m = 1.

    if dip_input_type == 'degrees':
        dip = np.deg2rad(dip)

    if strike == None:
        strike = get_strike_from_pts(lon_pts = lon_pts, lat_pts = lat_pts,
                                     fit_type = 'lsq', output = 'degrees')
    if elev_pts == None:
        elev_pts = np.zeros(lat_pts.shape)

    dip_dir = strike2angle( strike + 90)

    # Make 'base arrays', or repeated arrays of values to which
    # changes will be added.
    #lon_tile = np.tile(lon_pts, (d_len, 1) )
    lat_tile = np.tile(lat_pts, (d_len, 1) )
    elev_tile = np.tile(elev_pts, (d_len, 1) )
    
    # make 2d arrays of coordinates.  These are used below to calculate
    # arrays that are position changes to add to the base arrays.
    lon_tile, depth_grid = np.meshgrid(lon_pts, depth_vec)

    # take base arrays and apply changes, based on strike, dip, depth
    lon_pts_grid = depth_grid * (np.cos( dip) * np.cos( dip_dir) * deg_per_m)
    lat_pts_grid = depth_grid * (np.cos( dip) * np.sin( dip_dir) * deg_per_m)
    
    out_lon_pts = lon_tile + lon_pts_grid
    out_lat_pts = lat_tile + lat_pts_grid
    out_depth_pts = elev_tile - depth_grid

    if output_shape == 'vector':
        out_lon_pts = out_lon_pts.ravel()
        out_lat_pts = out_lat_pts.ravel()
        out_depth_pts = out_depth_pts.ravel()

    return out_lon_pts, out_lat_pts, out_depth_pts


def rotate_pts_2d(x_pts, y_pts, x0 = 0, y0 = 0, rotation_angle = 0,
                  angle_input = 'radians'):
    """
    Rotates vectors of x and y points around point [x0, y0] with 
    rotation_angle.  Specify 'degrees' for angle_input if rotation angle is
    in degrees.

    Returns list of [x_rotated, y_rotated] points
    """

    if angle_input == 'degrees':
        rotation_angle = np.deg2rad( rotation_angle)

    xn = x_pts - x0
    yn = y_pts - y0

    x_rot = xn * np.cos( rotation_angle) - yn * np.sin( rotation_angle)
    y_rot = xn * np.sin( rotation_angle) + yn * np.cos( rotation_angle)

    return np.array([x_rot, y_rot])
    

def get_princ_axes_xyz(tensor):
    """ 
    Gets the principal stress axes from a stress tensor.
    Modified from beachball.py from ObsPy, written by Robert Barsch.
    That code is modified from Generic Mapping Tools (gmt.soest.hawaii.edu)
    
    Returns 'PrincipalAxis' classes, which have attributes val, trend, plunge

    Returns T, N, P 
       
    """
    tensor = np.array(tensor)

    (D, V) = sorted_eigens(tensor)
    pl = np.arcsin( -V[2] ) # 2
    az = np.arctan2( V[0], -V[1] ) # 0 # 1

    for i in range(0, 3):
        if pl[i] <= 0:
            pl[i] = -pl[i]
            az[i] += np.pi
        if az[i] < 0:
            az[i] += 2 * np.pi
        if az[i] > 2 * np.pi:
            az[i] -= 2 * np.pi

    pl *= 180 / np.pi
    az *= 180 / np.pi

    T = PrincipalAxis( D[0], az[0], pl[0] ) # 0 0 0 
    N = PrincipalAxis( D[1], az[1], pl[1] )
    P = PrincipalAxis( D[2], az[2], pl[2] ) # 2 2 2 

    return(T, N, P)


class PrincipalAxis(object):
    """
    Modified from ObsPy's beachball.py, by Robert Barsch

    A principal axis.

    Trend and plunge values are in degrees.

    >>> a = PrincipalAxis(1.3, 20, 50)
    >>> a.plunge
    50
    >>> a.trend
    20
    >>> a.val
    1.3
    """
    def __init__(self, val=0, trend=0, plunge=0):
        self.val = val
        self.trend = trend
        self.plunge = plunge


def sphere_to_xyz(lon, lat, elev = 1):
    """Takes geographic/spherical coordinates and converts to XYZ
    coordinates.
    """
    
    x = elev * np.cos( lat) * np.cos( lon)
    y = elev * np.cos( lat) * np.sin (lon)
    z = elev * np.sin( lat)
    
    return np.array([x, y, z])


def rotate_XY_tensor(T, theta=0, input_angle='radians', out_type='matrix'):
    """ Rotates a 2x2 tensor by angle theta (in unit circle convention,
    not azimuthal convention).  Theta is by default in radians.  Specify
    angle_input = 'degrees' for input angle in degrees.

    Returns 2x2 rotated tensor.
    """
    theta = np.radians(theta) if input_angle == 'degrees' else theta

    s_xx = (T[0,0] * np.cos(theta)**2 + T[1,1] * np.sin(theta)**2
           - 2 * T[1,0] * np.sin(theta) * np.cos(theta) )
    
    s_yy = (T[0,0] * np.sin(theta)**2 + T[1,1] * np.cos(theta)**2
           + 2 * T[1,0] * np.sin(theta) * np.cos(theta) )
    
    s_xy = ( (T[0,0] - T[1,1]) * np.sin(theta) * np.cos(theta)
           + T[0,1] * (np.cos(theta)**2 - np.sin(theta)**2) )
    
    T_rot = make_xy_stress_tensor(s_xx=s_xx, s_yy=s_yy, s_xy=s_xy)

    T_rot = np.array(T_rot) if out_type == 'array' else T_rot

    return T_rot


def get_cartesian_xy_stress_dirs(T):
    ''' Takes a horizontal stress tensor and calculates the (x,y)
    values of the maximum and minimum principal stresses.
    '''

    vals, vecs = sorted_eigens(T)
    vecs = np.array(vecs)

    max_x = vecs[0,1] * vals[1]
    max_y = vecs[1,1] * vals[1]
    min_x = vecs[0,0] * vals[0]
    min_y = vecs[1,0] * vals[0]
    
    return max_x, max_y, min_x, min_y


def calc_xy_princ_stresses_from_stress_comps(s_xx=0, s_yy=0, s_xy=0):
    ''' Takes the three independent stress tensor components for a
    2 dimensional stress tensor and returns the (x,y) values of the
    maximum and minimum principal stresses.
    '''
    T = make_xy_stress_tensor(sig_xx=s_xx, sig_yy=s_yy, sig_xy=s_xy)
    
    max_x, max_y, min_x, min_y = get_cartesian_xy_stress_dirs(T)

    return max_x, max_y, min_x, min_y


def calc_xy_max_stress_from_stress_comps(s_xx=0., s_yy=0., s_xy=0.):
    '''
    Takes the three indepenent stress tensor components for a 2 dimensional
    stress tensor and returns the magnitude and azimuth of the maximum
    principal stress.

    Returns a tuple: max_magnitude, max_azimuth
    '''
    max_x, max_y, min_x, min_y = calc_xy_princ_stresses_from_stress_comps(
                                        s_xx=s_xx, s_yy=s_yy, s_xy=s_xy)

    max_mag = np.sqrt(max_x**2 + max_y**2)

    max_az = angle_to_azimuth( np.arctan2(max_y, max_x) )

    return max_mag, max_az
