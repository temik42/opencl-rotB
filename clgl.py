#from OpenGL.GL import *
#from OpenGL.GLU import *
#from OpenGL.GLUT import *

import pyopencl as cl
import sys
import threading
#import time

import numpy as np

ai = np.array([1./5,
               3./40, 9./40,
               44./45, -56./15, 32./9,
               19372./6561, -25360./2187, 64448./6561, -212./729,
               9017./3168, -355./33, -46732./5247, 49./176, -5103./18656,
               35./384, 0., 500./1113, 125./192, -2187./6784, 11./84], dtype = np.float32)

bi = np.array([5179./57600, 0, 7571./16695, 393./640, -92097./339200, 187./2100, 1./40], dtype = np.float32)


class CLGL(threading.Thread):
    def __init__(self, X, B, step = 0.05):
        threading.Thread.__init__(self)
        self.clinit()
        self.loadProgram("rotB.cl");
        self.loadData(X, B, step)
        #self.num = num
        #self.dt = numpy.float32(dt)

        #self.timings = timings

    def loadData(self, X, B, step):
        mf = cl.mem_flags

        self.shape = X.shape[0:3]
        self.size = X.nbytes

        self.B = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        self.J = cl.Buffer(self.ctx, mf.READ_WRITE, self.size*3)
        self.X = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        self.temp = [cl.Buffer(self.ctx, mf.READ_WRITE, self.size) for i in range(0,8)]
        self.err = cl.Buffer(self.ctx, mf.READ_WRITE, 4)
        self.step = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 4, hostbuf=np.float32(step))
        self.queue.finish()
        
        #self.gl_objects = [self.X, self.col_cl]
           
    def rotB(self, X, out):
        self.program.Jacobian(self.queue, self.shape, None, X, self.J)
        self.program.rotB(self.queue, self.shape, None, X, self.J, self.B, out)
        self.queue.finish()
        return out 

    def run(self):
        while (True):
    
            self.program.Copy(self.queue, self.shape, None, self.X, self.temp[0])
            self.program.Flush(self.queue, self.shape, None, self.temp[7])

            for i in range(0,6):
                self.rotB(self.temp[0], self.temp[i+1]) ## F1

                self.program.Step(self.queue, self.shape, None, self.temp[7], self.temp[i+1], self.step, 
                                  np.float32(bi[i]), np.byte(0))
                self.program.Copy(self.queue, self.shape, None, self.X, self.temp[0]) 
                for j in range(0,i+1):
                    self.program.Step(self.queue, self.shape, None, self.temp[0], self.temp[j+1], self.step, 
                                      np.float32(ai[i*(i+1)/2+j]), np.byte(1))


            self.program.flush(self.queue, (1,), None, self.err)
            #self.program.Error(self.queue, self.shape, None, self.X, self.temp[0], self.temp[7], np.float32(1e-3), self.err)
            self.program.Update(self.queue, self.shape, None, self.X, self.temp[0], self.step, 
                                self.err, np.byte(1))

            #self.program.AdjustStep(self.queue, (1,), None, self.step, self.err)
            self.queue.finish()
            
    
    def clinit(self):
        plats = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        import sys 
        if sys.platform == "darwin":
            self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
        else:
            self.ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, plats[0])]
                + get_gl_sharing_context_properties(), devices=None)
                
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #print fstr
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()


     

