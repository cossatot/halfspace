import numpy as np

def rot_matrix_3_x_axis(angle = 0):
    a = angle

    Rx = np.matrix([ [1, 0,          0        ],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)] ])
    return Rx


def rot_matrix_3_y_axis(angle = 0):
    a = angle

    Ry = np.matrix([ [ np.cos(a), 0, np.sin(a)],
                     [         0, 1,         0],
                     [-np.sin(a), 0, np.cos(a)] ])
    return Ry


def rot_matrix_3_z_axis(angle = 0):
    a = angle

    Rz = np.matrix([ [ np.cos(a), -np.sin(a), 0],
                     [ np.sin(a),  np.cos(a), 0],
                     [        0,           0, 1] ])
    return Rz


def strike2angle(strike, output='rad'):
    """ Takes strike angle (in degrees) and changes it into
    unit vector rotation angle (e.g. CCW from x axis/horizontal).

    defaults to output in radians, specify 'deg' for degrees.

    Returns angle
    """
    angle_deg = (-strike) + 90
    angle_rad = np.deg2rad(angle_deg)

    return angle_deg if output == 'deg' else angle_rad


def xyz_to_strike_dip_rot_matrix(strike, dip):
    """Rotates from XYZ to new plane coords (strike, dip) by rotating
       around the Z axis (strike rotation) and then
       rotating around the Y' axis (dip rotation).

       Strike, dip in degress (right hand rule)
       Returns rotation matrix.
    """
    z_angle = strike2angle(strike, output='rad')
    y_prime_angle = np.deg2rad(dip)

    strike_rot_mat = rot_matrix_3_z_axis(z_angle)
    dip_rot_mat = rot_matrix_3_y_axis(y_prime_angle)

    str_dip_rot_mat = strike_rot_mat * dip_rot_mat

    return str_dip_rot_mat


def make_xyz_stress_tensor(sig_xx = 0, sig_yy = 0, sig_zz = 0, sig_xy = 0,
                           sig_xz = 0, sig_yz = 0):
    '''take stresses and make tensor'''

    SIG = np.matrix([ [ sig_xx,  sig_xy, sig_xz],
                      [ sig_xy,  sig_yy, sig_yz],
                      [ sig_xz,  sig_yz, sig_zz]])
    return SIG


def decomp_xyz_tensor(tensor):
    A = tensor
    sig_xx = A[0,0]
    sig_xy = A[0,1]
    sig_xz = A[0,2]
    sig_yy = A[1,1]
    sig_yz = A[1,2]
    sig_zz = A[2,2]

    return [sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz]