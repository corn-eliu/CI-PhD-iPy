# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.GL import *
from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays, \
                                                  glBindVertexArray

import pygame

import numpy as np

def run():
    pygame.init()
    screen = pygame.display.set_mode((800,600), pygame.OPENGL|pygame.DOUBLEBUF)

    #Create the VBO
    vertices = np.array([[0,1,0],[-1,-1,0],[1,-1,0]], dtype='f')
    vertexPositions = vbo.VBO(vertices)

    #Create the index buffer object
    indices = np.array([[0,1,2]], dtype=np.int32)
    indexPositions = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)

    #Now create the shaders
    VERTEX_SHADER = shaders.compileShader("""
    #version 330
    layout(location = 0) in vec4 position;
    void main()
    {
        gl_Position = position;
    }
    """, GL_VERTEX_SHADER)

    FRAGMENT_SHADER = shaders.compileShader("""
    #version 330
    out vec4 outputColor;
    void main()
    {
        outputColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
    }
    """, GL_FRAGMENT_SHADER)

    shader = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

    #The draw loop
    while True:
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)

        indexPositions.bind()

        vertexPositions.bind()
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)

        #glDrawArrays(GL_TRIANGLES, 0, 3) #This line still works
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, None) #This line does work too!

        # Show the screen
        pygame.display.flip()

run()

