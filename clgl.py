from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import pyopencl as cl
import sys

import numpy as np

ai = np.array([1./5,
               3./40, 9./40,
               44./45, -56./15, 32./9,
               19372./6561, -25360./2187, 64448./6561, -212./729,
               9017./3168, -355./33, -46732./5247, 49./176, -5103./18656,
               35./384, 0., 500./1113, 125./192, -2187./6784, 11./84], dtype = np.float32)

bi = np.array([5179./57600, 0, 7571./16695, 393./640, -92097./339200, 187./2100, 1./40], dtype = np.float32)


class CLGL():
    def __init__(self):
        self.clinit()
        self.loadProgram("rotB.cl");

        #self.num = num
        #self.dt = numpy.float32(dt)

        #self.timings = timings

    def loadData(self, pos_vbo, col_vbo, B):
        import pyopencl as cl
        mf = cl.mem_flags
        
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo
        self.pos = pos_vbo.data
        self.col = col_vbo.data
        
        self.shape = self.pos.shape[0:3]
        self.size = self.pos.nbytes
        self.num = self.shape[0]*self.shape[1]*self.shape[2]

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.pos_vbo.bind()

        try:
            self.X = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffer))
            self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffer))
        except AttributeError:
            self.X = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffers[0]))
            self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffers[0]))
        self.col_vbo.bind()

        #pure OpenCL arrays
        self.B = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        self.J = cl.Buffer(self.ctx, mf.READ_WRITE, self.size*3)  
        self.temp = [cl.Buffer(self.ctx, mf.READ_WRITE, self.size) for i in range(0,8)]
        self.err = cl.Buffer(self.ctx, mf.READ_WRITE, 4)
        self.step = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 4, hostbuf=np.float32(1e-6))
        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.X, self.col_cl]
        
    def rotB(self, X, out):
        self.program.Jacobian(self.queue, self.shape, None, X, self.J)
        self.program.rotB(self.queue, self.shape, None, X, self.J, self.B, out)
        return out 

    def execute(self, step):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
        """
        self.program.Flush(self.queue, self.shape, None, self.temp[2])
        
        self.rotB(self.X, self.temp[0]) ## F1
        self.program.Step(self.queue, self.shape, None, self.X, self.temp[0], np.float32(step*0.5), np.byte(1), self.temp[1]) ## X1 
        self.program.Step(self.queue, self.shape, None, self.temp[2], self.temp[0], np.float32(1/6.), np.byte(0), self.temp[2]) ## F
        
        self.rotB(self.temp[1], self.temp[0]) ## F2
        self.program.Step(self.queue, self.shape, None, self.X, self.temp[0], np.float32(step*0.5), np.byte(1), self.temp[1]) ## X2
        self.program.Step(self.queue, self.shape, None, self.temp[2], self.temp[0], np.float32(1/3.), np.byte(0), self.temp[2]) ## F
        
        self.rotB(self.temp[1], self.temp[0]) ## F3
        self.program.Step(self.queue, self.shape, None, self.X, self.temp[0], np.float32(step), np.byte(1), self.temp[1]) ## X3
        self.program.Step(self.queue, self.shape, None, self.temp[2], self.temp[0], np.float32(1/3.), np.byte(0), self.temp[2]) ## F  
        
        self.rotB(self.temp[1], self.temp[0]) ## F4
        self.program.Step(self.queue, self.shape, None, self.temp[2], self.temp[0], np.float32(1/6.), np.byte(0), self.temp[2]) ## F
        self.program.Step(self.queue, self.shape, None, self.X, self.temp[2], np.float32(step), np.byte(1), self.X) ## X
        """
        
        
        self.program.Copy(self.queue, self.shape, None, self.X, self.temp[0])        
        self.program.Flush(self.queue, self.shape, None, self.temp[7])
        
        
        for i in range(0,6):
            self.rotB(self.temp[0], self.temp[i+1]) ## F1
            
            self.program.Step(self.queue, self.shape, None, self.temp[7], self.temp[i+1], self.step, 
                              np.float32(bi[i]), np.byte(0)) ## F
            self.program.Copy(self.queue, self.shape, None, self.X, self.temp[0]) 
            for j in range(0,i+1):
                self.program.Step(self.queue, self.shape, None, self.temp[0], self.temp[j+1], self.step, 
                                  np.float32(ai[i*(i+1)/2+j]), np.byte(1)) ## X1
        
        #self.program.Step(self.queue, self.shape, None, self.X, self.temp[7], np.float32(step), np.byte(1), self.X) ## X
        #"""
        
        self.program.flush(self.queue, (1,), None, self.err)
        self.program.Error(self.queue, self.shape, None, self.X, self.temp[0], self.temp[7], np.float32(1e-14), self.err)
        self.program.Update(self.queue, self.shape, None, self.X, self.temp[0], self.step, 
                            self.err, np.byte(1))
        
        #self.program.AdjustStep(self.queue, (1,), None, self.step, self.err)

        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)
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

    def render(self):
        
        glEnable(GL_POINT_SMOOTH)
        glPointSize(4)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, None)
        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, None)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        #for i in range(self.shape[0]/4,self.shape[0]/4*3, 4):
        #    for j in range(self.shape[1]/4,self.shape[1]/4*3, 4):
        #        glDrawArrays(GL_LINE_STRIP, i*self.shape[1]*self.shape[2] + j*self.shape[2], self.shape[2])
                
        for i in range(0,self.shape[0], 4):
            for j in range(0,self.shape[1], 4):
                glDrawArrays(GL_LINE_STRIP, i*self.shape[1]*self.shape[2] + j*self.shape[2], self.shape[2])
        
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)
     

