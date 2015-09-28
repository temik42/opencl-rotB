import pyopencl as cl
import sys
import threading
import numpy as np

ai = np.array([1./5,
               3./40, 9./40,
               44./45, -56./15, 32./9,
               19372./6561, -25360./2187, 64448./6561, -212./729,
               9017./3168, -355./33, -46732./5247, 49./176, -5103./18656,
               35./384, 0., 500./1113, 125./192, -2187./6784, 11./84], dtype = np.float32)

bi = np.array([5179./57600, 0, 7571./16695, 393./640, -92097./339200, 187./2100, 1./40], dtype = np.float32)

class CL(threading.Thread):
    def __init__(self, X, B, step = 0.05, scale = 1e-4, gl_enable = False):
        threading.Thread.__init__(self)
        self.clinit(gl_enable)
        self.loadProgram("rotB.cl")
        self.loadData(X, B, step, scale)
        
    def clinit(self, gl_enable):
        plats = cl.get_platforms()
        
        if gl_enable:
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
        self.program = cl.Program(self.ctx, fstr).build()
        
        
    def loadData(self, X, B, step, scale):
        mf = cl.mem_flags

        self.shape = X.shape[0:3]
        self.size = X.nbytes

        self.B = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        self.J = cl.Buffer(self.ctx, mf.READ_WRITE, self.size*3)
        self.X = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        self.ki = [cl.Buffer(self.ctx, mf.READ_WRITE, self.size) for i in range(0,8)]
        
        params = np.array([step,0.,scale], dtype = np.float32)  
        self.params = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 12, hostbuf=params)
        
        self.queue.finish()

           
    def rotB(self, X, out):
        self.program.Jacobian(self.queue, self.shape, None, X, self.J)
        self.program.rotB(self.queue, self.shape, None, X, self.J, self.B, out)
        self.queue.finish()
        return out 

    def run(self):
        while (True):
    
            self.program.Copy(self.queue, self.shape, None, self.X, self.ki[0])
            self.program.Flush(self.queue, self.shape, None, self.ki[7])

            for i in range(0,6):
                self.rotB(self.ki[0], self.ki[i+1]) ## ki

                self.program.Step(self.queue, self.shape, None, self.ki[7], self.ki[i+1], self.params, 
                                  np.float32(bi[i]))
                self.program.Copy(self.queue, self.shape, None, self.X, self.ki[0]) 
                for j in range(0,i+1):
                    self.program.Step(self.queue, self.shape, None, self.ki[0], self.ki[j+1], self.params, 
                                      np.float32(ai[i*(i+1)/2+j]))

            self.program.Error(self.queue, self.shape, None, self.X, self.ki[0], self.ki[7], self.params)
            self.program.Update(self.queue, self.shape, None, self.X, self.ki[0], self.params)
            self.program.AdjustParams(self.queue, (1,), None, self.params)
            self.queue.finish()
    
    def get(self):
        X = np.zeros(self.shape + (4,), dtype = np.float32)
        cl.enqueue_read_buffer(self.queue, self.X, X).wait()
        return X


     

