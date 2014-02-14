import halfspace.projections as hsp
import numpy as np
import numpy.testing as npt

def test_rotate_XY_tensor():

   tt = [[3,0], [0,2]]
   t_good = [[2.25, 0.4330127], [0.4330127, 2.75]]
   tr = hsp.rotate_XY_tensor(tt, np.pi/3.)

   npt.assert_array_almost_equal(tr, t_good)
    

