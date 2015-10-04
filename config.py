import numpy as np

ai = np.array([1./5,
               3./40, 9./40,
               44./45, -56./15, 32./9,
               19372./6561, -25360./2187, 64448./6561, -212./729,
               9017./3168, -355./33, 46732./5247, 49./176, -5103./18656,
               35./384, 0., 500./1113, 125./192, -2187./6784, 11./84], dtype = np.float32)

bi = np.array([5179./57600, 0, 7571./16695, 393./640, -92097./339200, 187./2100, 1./40], dtype = np.float32)

step = 1e-2
scale = 1e-2
block_size = 8
block_shape = (block_size,block_size,block_size)

glEnable = False
