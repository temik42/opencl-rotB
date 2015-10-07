import cv2
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import numpy as np
#helper modules
import glutil
from vector import Vec
from datetime import datetime


class window():
    def __init__(self, cle, capture = False, video_dir = ''):
        self.cle = cle
        self.shape = cle._X.shape[0:3]
     
        #mouse handling for transforming scene
        self.mouse_down = False
        self.mouse_old = Vec([0., 0.])
        self.rotate = Vec([0., 0., 0.])
        self.translate = Vec([0., 0., 0.])
        self.initrans = Vec([0., 0., -self.shape[2]])

        self.width = 800
        self.height = 800
        
        self.capture = capture
        self.video_dir = video_dir
        if self.capture:
            self.captureInit()
        
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow("")

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)
        
        #this will call draw every 30 ms
        glutTimerFunc(30, self.timer, 30)

        #setup OpenGL scene
        self.glinit()
        self.loadVBO()
        glutMainLoop()

    
    def glinit(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60., self.width / float(self.height), .1, 1000.)
        glMatrixMode(GL_MODELVIEW)

    def captureInit(self):
        systime = datetime.now()
        hms = str(systime.hour).zfill(2)+str(systime.minute).zfill(2)+str(systime.second).zfill(2)
        self.fname = self.video_dir+'/video'+hms+'.avi'
        self.video = cv2.VideoWriter(self.fname,cv2.VideoWriter_fourcc('X','2','6','4'),30,(self.width,self.height))
        self.pbyData = np.zeros((self.width,self.height,3), dtype = np.ubyte)
    
    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            self.cle.exit()
            sys.exit(0)
            
        if args[0] == 's':
            if self.capture:
                print 'Stopping video capture'
                self.capture = False
                self.video.release()
            else:
                self.captureInit()
                print 'Starting video capture. Saving to '+self.fname
                self.capture = True

    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            self.rotate.x += dy * .2
            self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate.z -= dy * .1 
        self.mouse_old.x = x
        self.mouse_old.y = y
    ###END GL CALLBACKS    
    
    def render(self):
        
        glLineWidth(2)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, None)
        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, None)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
  
        for i in range(0,self.shape[0], 4):
            for j in range(0,self.shape[1], 4):
                glDrawArrays(GL_LINE_STRIP, i*self.shape[1]*self.shape[2] + j*self.shape[2], self.shape[2])
        
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)
        
    def draw(self):      
        
        self.pos_vbo.set_array(self.cle._X)
        self.pos_vbo.bind()
        
        red = 5*np.sqrt(np.sum(self.cle._Current[:,:,:,0:3]**2, axis = 3))/self.CNorm
        self.Color[:,:,:,0] = red
        self.Color[:,:,:,1] = 1-red
        
        self.col_vbo.set_array(self.Color)
        self.col_vbo.bind()
        
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans.x, self.initrans.y, self.initrans.z)
        glTranslatef(self.translate.x, self.translate.y, self.translate.z)
        glRotatef(self.rotate.x, 1, 0, 0)
        glRotatef(self.rotate.y, 0, 1, 0)
        
        self.render()
        
        if self.capture:
            glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE, self.pbyData)
            self.video.write(self.pbyData)

        glutSwapBuffers()
        
        
        
    def loadVBO(self):    
        self.Color = np.zeros_like(self.cle._X)
        self.Color[:,:,:,0] = 0.
        self.Color[:,:,:,1] = 1.
        self.Color[:,:,:,2] = 0.
        self.Color[:,:,:,3] = 1.
        self.CNorm = np.sqrt(np.sum(self.cle._B[:,:,:,0:3]**2, axis = 3))
        #create the Vertex Buffer Objects
        from OpenGL.arrays import vbo 
        self.pos_vbo = vbo.VBO(data=self.cle._X, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.pos_vbo.bind()
        self.col_vbo = vbo.VBO(data=self.Color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.col_vbo.bind()
        return self
