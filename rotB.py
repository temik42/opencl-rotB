import pyopencl as cl
import sys
import threading
import numpy as np
from config import *



class CL(threading.Thread):
    def __init__(self, X, B):
        threading.Thread.__init__(self)  
        self.clinit()
        self.loadData(X, B)
        self.loadProgram("rotB.cl")
        self.set_params(step,scale)
        
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
        

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        kernel_params = {"block_size": block_size, "nx": self.shape[0], "ny": self.shape[1], "nz": self.shape[2]}
        self.program = cl.Program(self.ctx, fstr % kernel_params).build()
        
        
    def loadData(self, X, B):
        mf = cl.mem_flags

        self.shape = X.shape[0:3]
        self.size = X.nbytes

        self.B = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        self.J = cl.Buffer(self.ctx, mf.READ_WRITE, self.size*3)
        self.Ix = cl.Buffer(self.ctx, mf.READ_WRITE, self.size)
        self.X = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=X)
        self.ki = [cl.Buffer(self.ctx, mf.READ_WRITE, self.size) for i in range(0,8)]
        
        self.params = cl.Buffer(self.ctx, mf.READ_WRITE, 12)
        
        self.queue.finish()

    def set_params(self, _step, _scale):
        cl.enqueue_write_buffer(self.queue, self.params, np.array([_step, 0, _scale], dtype = np.float32)).wait()
        self.queue.finish()
        
           
    def force(self, X, out):
        #self.program.Hessian(self.queue, self.shape, block_shape, X)
        self.program.Jacobian(self.queue, self.shape, block_shape, X, self.J)
        self.queue.finish()
        self.program.Force(self.queue, self.shape, block_shape, X, self.J, self.B, self.Ix, out)
        self.queue.finish()

        
    def adjust(self):
        _params = np.zeros((3,), dtype = np.float32)
        cl.enqueue_read_buffer(self.queue, self.params, _params).wait()
        _params[0] /= _params[1]**0.1 + 0.5
        self.set_params(_params[0], _params[2])
        
    def run(self):
        while (True):
    
            self.program.Copy(self.queue, self.shape, None, self.X, self.ki[0])
            self.program.Copy(self.queue, self.shape, None, self.X, self.ki[7])
            #self.enqueue_copy(self.queue, self.ki[0], self.X)
            

            for i in range(0,6):
                self.force(self.ki[0], self.ki[i+1]) ## ki

                self.program.Step(self.queue, self.shape, None, self.ki[7], self.ki[i+1], self.params, 
                                  np.float32(bi[i]))
                self.program.Copy(self.queue, self.shape, None, self.X, self.ki[0])
                
                for j in range(0,i+1):
                    self.program.Step(self.queue, self.shape, None, self.ki[0], self.ki[j+1], self.params, 
                                      np.float32(ai[i*(i+1)/2+j]))

            self.program.Error(self.queue, self.shape, None, self.ki[0], self.ki[7], self.params)
            self.program.Update(self.queue, self.shape, None, self.X, self.ki[0], self.params)
            self.adjust()
            
    
    def get(self):
        X = np.zeros(self.shape + (4,), dtype = np.float32)
        cl.enqueue_read_buffer(self.queue, self.X, X).wait()
        return X


     

