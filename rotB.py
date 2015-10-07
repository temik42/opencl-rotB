import pyopencl as cl
import sys
import threading
import numpy as np
from config import *



class CL(threading.Thread):
    def __init__(self, X, B):
        threading.Thread.__init__(self) 
        self.shape = X.shape[0:3]
        self.step = np.float32(step)
        self.scale = np.float32(scale)
        self.clinit()
        self.loadData(X, B)
        self.loadProgram("rotB.cl")
        
        
    def clinit(self):
        plats = cl.get_platforms()
        
        if glEnable:
            from pyopencl.tools import get_gl_sharing_context_properties
            if sys.platform == "darwin":
                self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                                 devices=[])
            else:
                self.ctx = cl.Context(properties=[
                    (cl.context_properties.PLATFORM, plats[0])]
                    + get_gl_sharing_context_properties(), devices=None)
                
        else:
            self.ctx = cl.create_some_context()
            
        self.queue = cl.CommandQueue(self.ctx)
        self.group_shape = (self.shape[0]/block_size, self.shape[1]/block_size, self.shape[2]/block_size)
        
        

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        kernel_params = {"block_size": block_size, "scale": self.scale, "nx": self.shape[0], "ny": self.shape[1], "nz": self.shape[2],
                        "mx": self.group_shape[0], "my": self.group_shape[1], "mz": self.group_shape[2]}
        self.program = cl.Program(self.ctx, fstr % kernel_params).build()
        
        
    def loadData(self, X, B):
        self._X = X
        self._B = B
        self._Error = np.zeros(self.shape, dtype = np.float32)
        self._Current = np.zeros(self.shape+(4,), dtype = np.float32)
        mf = cl.mem_flags
        self.size = X.nbytes
        
        self.X = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)
        self.X1 = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)
        self.B = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        self.DB = cl.Buffer(self.ctx, mf.READ_WRITE, self.size*3)  
        self.Current = cl.Buffer(self.ctx, mf.READ_WRITE, self.size)
        self.Error = cl.Buffer(self.ctx, mf.READ_WRITE, self.size/4)

        self.queue.finish()     


        
    def run(self):
        self.program.Jacobian(self.queue, self.shape, block_shape, self.B, self.DB)
        
        while (True):
            self.program.Dopr(self.queue, self.shape, block_shape, 
                              self.X, self.X1, self.B, self.DB, np.float32(self.step), self.Error, self.Current)
            cl.enqueue_barrier(self.queue)
            cl.enqueue_read_buffer(self.queue, self.Error, self._Error)       
            error = np.max(self._Error)
            if (not np.isnan(error)):
                cl.enqueue_read_buffer(self.queue, self.Current, self._Current)
                self.step /= error**0.1+0.5
                if (error < 1):
                    cl.enqueue_copy(self.queue, self.X, self.X1)
                    cl.enqueue_read_buffer(self.queue, self.X1, self._X)
            else:
                self.step /= 2

            self.queue.finish()
            


     

