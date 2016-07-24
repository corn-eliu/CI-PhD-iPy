# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
from __future__ import print_function
import sys
import numpy as np
import time
import os
import glob
import pyassimp
import cv2
from scipy import ndimage as spimg

from PySide import QtGui, QtCore, QtOpenGL
from PySide.QtOpenGL import QGLWidget, QGLFormat

import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import OpenGL.GL.EXT.texture_filter_anisotropic as tfa
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D

app = QtGui.QApplication(sys.argv)

DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_NUM_SEMANTICS = "number_of_semantic_classes"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"
# DICT_FRAME_COMPATIBILITY_LABELS = 'compatibiliy_labels_per_frame'
DICT_LABELLED_FRAMES = 'labelled_frames' ## includes the frames labelled for the semantic labels (the first [DICT_FRAME_SEMANTICS].shape[1])
DICT_NUM_EXTRA_FRAMES = 'num_extra_frames' ## same len as DICT_LABELLED_FRAMES
DICT_CONFLICTING_SEQUENCES = 'conflicting_sequences'
DICT_DISTANCE_MATRIX_LOCATION = 'sequence_precomputed_distance_matrix_location' ## for label propagation
DICT_SEQUENCE_LOCATION = "sequence_location"

# <codecell>

def compile_vertex_shader(source):
    """Compile a vertex shader from source."""
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, source)
    gl.glCompileShader(vertex_shader)
    # check compilation error
    result = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(vertex_shader))
    return vertex_shader

def compile_fragment_shader(source):
    """Compile a fragment shader from source."""
    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, source)
    gl.glCompileShader(fragment_shader)
    # check compilation error
    result = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(fragment_shader))
    return fragment_shader

def link_shader_program(vertex_shader, fragment_shader):
    """Create a shader program from compiled shaders."""
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    # check linking error
    result = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program

def compileShaders(vs, fs) :
    # compile the vertex shader
    try :
        compiledVS = compile_vertex_shader(vs)
    except Exception as e :
        print("VS COMPILE ERROR:", e.message, file=sys.stderr)
        sys.stderr.flush()
    # compile the fragment shader
    try :
        compiledFS = compile_fragment_shader(fs)
    except Exception as e :
        print("FS COMPILE ERROR:", e.message, file=sys.stderr)
        sys.stderr.flush()
    # link shader program
    try :
        return link_shader_program(compiledVS, compiledFS)
    except Exception as e :
        print("LINK PROGRAM ERROR:", e.message, file=sys.stderr)
        sys.stderr.flush()

# Vertex shader
VS_HEAD_LIGHT = """
#version 330
// Attribute variable that contains coordinates of the vertices.
layout(location = 0) in vec3 position_model;
layout(location = 1) in vec2 uv_model;
layout(location = 2) in vec3 normal_model;
layout(location = 3) in vec3 barycentric_model;

// the data to be sent to the fragment shader
out Data {
    vec2 uv_model;
    vec3 position_world;
    vec3 normal_camera;
    vec3 eye_camera;
    vec3 l_dir_camera;
    vec3 barycentric_model;
} DataOut;

uniform mat4 m_pvm;
uniform mat4 m_m;
uniform mat4 m_v;
uniform vec3 l_pos_world;

// Main function, which needs to set `gl_Position`.
void main()
{
    gl_Position = m_pvm * vec4(position_model, 1.0);

    DataOut.position_world = (m_m * vec4(position_model, 1.0)).xyz;

    vec3 position_camera = (m_v * vec4(DataOut.position_world, 1.0)).xyz;
    DataOut.eye_camera = -position_camera;

    vec3 l_pos_camera = (m_v * vec4(l_pos_world,1.0)).xyz;
    DataOut.l_dir_camera = l_pos_camera - position_camera;

    DataOut.normal_camera = (m_v * m_m * vec4(normal_model, 0.0)).xyz; // Only correct if ModelMatrix does not scale the model ! Use its inverse transpose if not.

    DataOut.uv_model = uv_model;
    DataOut.barycentric_model = barycentric_model;
}
"""

# Fragment shader
FS_HEAD_LIGHT = """
#version 330

uniform vec4 diffuse_m;
uniform vec4 ambient_m;
uniform vec4 specular_m;
uniform float shininess_m;

uniform vec3 l_pos_world;
uniform vec3 l_color;
uniform float l_power;
uniform bool show_edges;

// the data received from the vertex shader
in Data {
    vec2 uv_model;
    vec3 position_world;
    vec3 normal_camera;
    vec3 eye_camera;
    vec3 l_dir_camera;
    vec3 barycentric_model;
} DataIn;

out vec4 out_color;

float edgeFactor()
{
    vec3 d = fwidth(DataIn.barycentric_model);
    vec3 a3 = smoothstep(vec3(0.0), d*1.0, DataIn.barycentric_model);
    return min(min(a3.x, a3.y), a3.z);
}

// Main fragment shader function.
void main()
{
    float distance = length(l_pos_world - DataIn.position_world);

    vec3 n = normalize(DataIn.normal_camera);
    vec3 l = normalize(DataIn.l_dir_camera);
    float cosTheta = clamp(dot(n,l), 0.0, 1.0);

    vec3 e = normalize(DataIn.eye_camera);
    vec3 r = reflect(-l, n);
    float cosAlpha = clamp(dot(e,r), 0.0, 1.0);
    
    vec4 shading_color = ambient_m +
                         diffuse_m*vec4(l_color, 1.0)*l_power*cosTheta +//(distance*distance) + 
                         specular_m*vec4(l_color, 1.0)*l_power*pow(cosAlpha,shininess_m);//(distance*distance);
    
    if (show_edges) {
        out_color.xyz = mix(vec3(0.0), shading_color.xyz, edgeFactor());
    }
    else {
        out_color = shading_color;
    }
}
"""

# Vertex shader
VS_DIR_LIGHT = """
#version 330

layout(location = 0) in vec3 position_model;
layout(location = 1) in vec2 uv_model;
layout(location = 2) in vec3 normal_model;
layout(location = 3) in vec3 barycentric_model;

uniform mat4 m_pvm;
uniform mat4 m_m;
uniform mat4 m_v;

// the data to be sent to the fragment shader
out Data {
    vec3 normal_camera;
    vec4 eye_camera;
    vec3 barycentric_model;
} DataOut;

void main () {

    DataOut.normal_camera = normalize(m_v * m_m * vec4(normal_model, 0.0)).xyz;
    DataOut.eye_camera = vec4(-(m_v * m_m * vec4(position_model, 1.0)).xyz, 1.0);

    gl_Position = m_pvm * vec4(position_model, 1.0);
    
    DataOut.barycentric_model = barycentric_model;
}
"""

# Fragment shader
FS_DIR_LIGHT = """
#version 330

uniform vec4 diffuse_m;
uniform vec4 ambient_m;
uniform vec4 specular_m;
uniform float shininess_m;

uniform vec3 l_dir;
uniform vec3 l_color;
uniform float l_power;
uniform bool show_edges;

in Data {
    vec3 normal_camera;
    vec4 eye_camera;
    vec3 barycentric_model;
} DataIn;

out vec4 out_color;

float edgeFactor()
{
    vec3 d = fwidth(DataIn.barycentric_model);
    vec3 a3 = smoothstep(vec3(0.0), d*1.0, DataIn.barycentric_model);
    return min(min(a3.x, a3.y), a3.z);
}

void main() {
    // set the specular term to black
    vec4 spec = vec4(0.0);

    // normalize both input vectors
    vec3 n = normalize(DataIn.normal_camera);
    vec3 e = normalize(vec3(DataIn.eye_camera));

    float intensity = max(dot(n,l_dir), 0.0);

    // if the vertex is lit compute the specular color
    if (intensity > 0.0) {
        // compute the half vector
        vec3 h = normalize(l_dir + e);
        // compute the specular term into spec
        float intSpec = max(dot(h,n), 0.0);
        spec = specular_m * pow(intSpec,shininess_m);
    }
    vec4 shading_color = max((intensity * diffuse_m + spec)*vec4(l_color, 1.0)*l_power, ambient_m);
    
    if (show_edges) {
        out_color.xyz = mix(vec3(0.0), shading_color.xyz, edgeFactor());
    }
    else {
        out_color = shading_color;
    }
}
"""

# Vertex shader
VS_COLOR_NO_SHADE = """
#version 330

layout(location = 0) in vec3 position_model;
layout(location = 1) in vec3 color;

uniform mat4 m_pvm;
uniform float camera_dist;

out Data {
    vec3 color;
} DataOut;

void main () {
    gl_Position = m_pvm * vec4(position_model*camera_dist, 1.0);
    DataOut.color = color;
}
"""

# Fragment shader
FS_COLOR_NO_SHADE = """
#version 330

out vec4 out_color;

in Data {
    vec3 color;
} DataIn;

void main() {
    out_color = vec4(DataIn.color, 1.0);
}
"""

# Vertex shader
VS_IMAGE = """
#version 330

layout(location = 0) in vec3 position_model;
layout(location = 1) in vec2 uv_model;

uniform mat4 m_pvm;
uniform mat4 m_m;
uniform mat4 m_v;

out Data {
    vec2 uv_model;
} DataOut;

void main () {
    gl_Position = m_pvm * vec4(position_model, 1.0);
    DataOut.uv_model = uv_model;
}
"""

# Fragment shader
FS_IMAGE = """
#version 330

uniform sampler2D texture_sampler;

out vec4 out_color;

in Data {
    vec2 uv_model;
} DataIn;

void main() {
    out_color = texture(texture_sampler, DataIn.uv_model);
}
"""

# <codecell>

def readObj(lines) :
    vertices = []
    uvs = []
    normals = []
    faces = []
    barycentrics = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    for line in lines.split("\n") :
        elements = line.split(" ")
        if elements[0] == "v" :
            vertices.append(np.array(elements[1:]).astype(np.float32))
        elif elements[0] == "vt" :
            uvs.append(np.array(elements[1:]).astype(np.float32))
        elif elements[0] == "vn" :
            normals.append(np.array(elements[1:]).astype(np.float32))
        elif elements[0] == "f" :
            faces.append(np.empty(0, np.int32))
            for i, element in enumerate(elements[1:]) :
                faces[-1] = np.concatenate([faces[-1], np.concatenate([np.array(element.split("/")), [i+1]]).astype(np.int32)])
    return np.array(vertices), np.array(uvs), np.array(normals), np.array(faces), barycentrics
    
def reIndexTriangleMesh(vertices, uvs, normals, faces, barycentrics) :
    indices_per_vertex = faces.shape[1]/3
    in_indices = faces.reshape([len(faces)*3, indices_per_vertex])
    kept_indices = np.empty([0, indices_per_vertex], np.uint32)
    out_vertices = np.empty([0, vertices.shape[1]], vertices.dtype)
    out_uvs = np.empty([0, uvs.shape[1]], uvs.dtype)
    out_normals = np.empty([0, normals.shape[1]], normals.dtype)
    out_barycentrics = np.empty([0, barycentrics.shape[1]], barycentrics.dtype)
    out_indices = np.empty(0, in_indices.dtype)
    
    for i in xrange(len(in_indices)):
        in_index = in_indices[i].reshape([1, indices_per_vertex])
        mapDiffs = np.sqrt(np.sum((kept_indices-in_index)**2, axis=1))
        if np.any(mapDiffs == 0) :
            out_indices = np.concatenate([out_indices, [np.int32(int(np.argwhere(mapDiffs == 0)))]])
            continue
        kept_indices = np.concatenate([kept_indices, in_index])
        in_index = np.copy(in_index)-1
        out_vertices = np.concatenate([out_vertices, vertices[in_index[0, 0], :].reshape([1, out_vertices.shape[1]])])
        out_uvs = np.concatenate([out_uvs, uvs[in_index[0, 1], :].reshape([1, out_uvs.shape[1]])])
        out_normals = np.concatenate([out_normals, normals[in_index[0, 2], :].reshape([1, out_normals.shape[1]])])
        out_barycentrics = np.concatenate([out_barycentrics, barycentrics[in_index[0, 3], :].reshape([1, out_barycentrics.shape[1]])])
        out_indices = np.concatenate([out_indices, [np.int32(len(kept_indices))-np.int32(1)]])
        
    
    return out_vertices, out_uvs, out_normals, out_barycentrics, out_indices

# with open("../data/cube.obj") as objFile :
#     objlines = objFile.read()
#     a, b, c, d, e = reIndexTriangleMesh(*readObj(objlines))
#     print(a.shape)
#     print(b.shape)
#     print(c.shape)
#     print(d.shape)
#     print(e.shape)

# <codecell>

def getWorldSpacePosAndNorm(transform, normDirEnd= np.array([[0.0], [0.0], [1.0], [1.0]]), posOnly=False) :
    pos = np.dot(transform, np.array([[0.0], [0.0], [0.0], [1.0]])).T
    pos = pos[0, :3]/pos[0, 3]
    if posOnly :
        return pos
    norm = np.dot(transform, normDirEnd).T
    norm = norm[0, :3]/norm[0, 3]
    norm -= pos
    norm /= np.linalg.norm(norm)
    
    return pos, norm

def quaternionTo4x4Rotation(quaternion, inverted=False):
    x, y, z, w = quaternion
    ## quaternion rotation
    M = np.array([[1.0 - 2.0*(y**2) - 2.0*(z**2), 2*x*y + 2*w*z, 2*x*z - 2*w*y, 0.0],
                  [2*x*y - 2*w*z, 1.0 - 2.0*(x**2) - 2.0*(z**2), 2*y*z + 2*w*x, 0.0],
                  [2*x*z + 2*w*y, 2*y*z - 2*w*x, 1.0 - 2.0*(x**2) - 2.0*(y**2), 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    ## invert it
    if inverted :
        M[:-1, :-1] = M[:-1, :-1].T
        
    return M

def angleAxisToQuaternion(angle, axis) :
    return np.array([axis[0]*np.sin(angle/2.0), axis[1]*np.sin(angle/2.0), axis[2]*np.sin(angle/2.0), np.cos(angle/2.0)])

def rotateAboutPoint(matrix, quaternion, centerPoint) :
    M = quaternionTo4x4Rotation(quaternion)
    T = np.array([[1.0, 0.0, 0.0, centerPoint[0]],
                  [0.0, 1.0, 0.0, centerPoint[1]],
                  [0.0, 0.0, 1.0, centerPoint[2]],
                  [0.0, 0.0, 0.0, 1.0]])
    
    return np.dot(T, np.dot(M, np.dot(np.linalg.inv(T), matrix)))

# <codecell>

class GLMaterial() :
    def __init__(self) :
        self.diffuseComponent = np.array([0.6, 0.6, 0.6, 1.0], np.float32)
        self.ambientComponent = np.array([0.06, 0.06, 0.06, 1.0], np.float32)
        self.specularComponent = np.array([0.2, 0.2, 0.2, 1.0], np.float32)
        self.shininess = np.float32(5.0)
    
    ## not sure this actually cleans up properly
    def __del__(self) :
        del self.diffuseComponent, self.ambientComponent, self.specularComponent, self.shininess
        
class AxesWidget() :
    def __init__(self) :
        arrowLength = 0.12
        arrowSpacing = arrowLength*0.2
        self.arrowBodyVerticesBuffer = glvbo.VBO(np.array([[arrowSpacing, 0.0, 0.0], [arrowLength, 0.0, 0.0],
                                                           [0.0, arrowSpacing, 0.0], [0.0, arrowLength, 0.0],
                                                           [0.0, 0.0, arrowSpacing], [0.0, 0.0, arrowLength]], np.float32), gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.arrowBodyColorsBuffer = glvbo.VBO(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], np.float32), gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.arrowBodyIndexBuffer = glvbo.VBO(np.array([0, 1, 2, 3, 4, 5], np.int32), gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        
        self.initDone = False
        
        arrowMesh = pyassimp.load("arrowTop.obj").meshes[0]
        inputVertices = arrowMesh.vertices.astype(np.float32)*np.float32(0.65)
        verticesArray = np.concatenate([np.dot(inputVertices, np.array([[np.cos(np.pi/2.0), -np.sin(np.pi/2.0), 0.0],
                                                                        [np.sin(np.pi/2.0), np.cos(np.pi/2.0), 0.0], [0.0, 0.0, 1.0]], np.float32))+np.array([arrowLength, 0.0, 0.0], np.float32),  #### X arrow
                                        inputVertices+np.array([0.0, arrowLength, 0.0], np.float32),                                                                                                #### Y arrow
                                        np.dot(inputVertices, np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-np.pi/2.0), -np.sin(-np.pi/2.0)],
                                                                        [0.0, np.sin(-np.pi/2.0), np.cos(-np.pi/2.0)]], np.float32))+np.array([0.0, 0.0, arrowLength], np.float32)],                #### Z arrow
                                       axis = 0)
        
        colorArray = np.zeros([3*len(arrowMesh.vertices), 3], np.float32)
        colorArray[:len(arrowMesh.vertices), 0] = np.float32(1.0)                          #### X arrow
        colorArray[len(arrowMesh.vertices):2*len(arrowMesh.vertices), 1] = np.float32(1.0) #### Y arrow
        colorArray[2*len(arrowMesh.vertices):, 2] = np.float32(1.0)                        #### Z arrow
        
        indicesArray = np.concatenate([arrowMesh.faces.flatten().astype(np.int32),                              #### X arrow
                                       arrowMesh.faces.flatten().astype(np.int32)+len(arrowMesh.vertices),      #### Y arrow
                                       arrowMesh.faces.flatten().astype(np.int32)+2*len(arrowMesh.vertices)])   #### Z arrow
        
        self.arrowVerticesBuffer = glvbo.VBO(verticesArray.astype(np.float32), gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.arrowColorsBuffer = glvbo.VBO(colorArray.astype(np.float32), gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.arrowIndexBuffer = glvbo.VBO(indicesArray.astype(np.int32), gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        self.numIndices = len(indicesArray)
        
    def setShaders(self) :
        self.shaders_program = compileShaders(VS_COLOR_NO_SHADE, FS_COLOR_NO_SHADE)
        if np.any(self.shaders_program == None) :
            self.initDone = False
            return
        self.initDone = True
        
    def draw(self, cameraDist, pvm) :
#         self.setShaders()
        if self.initDone :
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            
            gl.glUseProgram(self.shaders_program)
            
            ## send mvp
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shaders_program, "m_pvm"), 1, gl.GL_FALSE, pvm.T)
            ## send camera distance
            gl.glUniform1f(gl.glGetUniformLocation(self.shaders_program, "camera_dist"), cameraDist)
#             print(cameraDist)

            ################ RENDER BODY ################
    
            ## bind the index buffer
            self.arrowBodyIndexBuffer.bind()

            ## bind the VBO with vertex data
            self.arrowBodyVerticesBuffer.bind()
            gl.glEnableVertexAttribArray(0)
            # tell OpenGL that the VBO contains an array of vertices
            # these vertices contain 3 single precision coordinates
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            
            ## bind the VBO with color data
            self.arrowBodyColorsBuffer.bind()
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## draw points from the VBO
            gl.glDrawElements(gl.GL_LINES, 6, gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)

            ################ RENDER ARROWS ################
            
            ## bind the index buffer
            self.arrowIndexBuffer.bind()

            ## bind the VBO with vertex data
            self.arrowVerticesBuffer.bind()
            gl.glEnableVertexAttribArray(0)
            # tell OpenGL that the VBO contains an array of vertices
            # these vertices contain 3 single precision coordinates
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            
            ## bind the VBO with color data
            self.arrowColorsBuffer.bind()
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## draw points from the VBO
            gl.glDrawElements(gl.GL_TRIANGLES, self.numIndices, gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)


            gl.glUseProgram(0)

class GLMesh() :
    def __init__(self, mesh, shaders_program) :
        self.modelMat = np.eye(4, dtype=np.float32)
        self.material = GLMaterial()
        self.shaders_program = shaders_program
        self.indices = []
        self.vertices = []
        
        if len(mesh.faces) == 0 and len(mesh.vertices) == 0 :
            self.isInvalidMesh = True
            self.invalidMeshMessage = "No Faces or Vertices"
        elif len(mesh.faces) > 0 and len(mesh.vertices) == 0 :
            self.isInvalidMesh = True
            self.invalidMeshMessage = "No Vertices even though there are Faces"
        else :
            if len(mesh.normals) == 0 :
                self.isInvalidMesh = True
                self.invalidMeshMessage = "No Normals"
            else :
                self.isInvalidMesh = False
                self.invalidMeshMessage = ""

                if len(mesh.faces) > 0 :
                    self.barycentrics = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], np.float32).repeat(len(mesh.faces), axis=0).flatten().reshape([len(mesh.faces.flatten()), 3])
                    self.indices = mesh.faces.flatten().astype(np.int32)
                    if len(mesh.texturecoords) > 0 :
                        self.uvs = mesh.texturecoords.astype(np.float32)
                    else :
                        self.uvs = np.zeros([len(self.indices), 2], np.float32)
                        
                    self.indexBuffer = glvbo.VBO(self.indices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
                    self.barycentricsBuffer = glvbo.VBO(self.barycentrics, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
                    self.uvsBuffer = glvbo.VBO(self.uvs, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
                else :
                    self.barycentrics = []
                    self.indices = []
                    self.uvs = []

                self.vertices = mesh.vertices.astype(np.float32)
                self.verticesBuffer = glvbo.VBO(self.vertices, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)

                self.normals = mesh.normals.astype(np.float32)
                self.normalsBuffer = glvbo.VBO(self.normals, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)

    ## not sure this actually cleans up properly
    def __del__(self) :
        del self.material, self.vertices, self.uvs, self.normals, self.barycentrics, self.indices
        
    def draw(self, projectionMat, viewMat) :
        if self.shaders_program != 0 :
            if len(self.indices) > 0 :
                gl.glUseProgram(self.shaders_program)
                
                ## send mvp
                gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shaders_program, "m_pvm"), 1, gl.GL_FALSE, np.dot(projectionMat, np.dot(viewMat, self.modelMat)).T)
                ## send model
                gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shaders_program, "m_m"), 1, gl.GL_FALSE, self.modelMat.T)

                # send material data
                gl.glUniform4fv(gl.glGetUniformLocation(self.shaders_program, "diffuse_m"), 1, self.material.diffuseComponent)
                gl.glUniform4fv(gl.glGetUniformLocation(self.shaders_program, "ambient_m"), 1, self.material.ambientComponent)
                gl.glUniform4fv(gl.glGetUniformLocation(self.shaders_program, "specular_m"), 1, self.material.specularComponent)
                gl.glUniform1f(gl.glGetUniformLocation(self.shaders_program, "shininess_m"), self.material.shininess)

                ## bind the index buffer
                self.indexBuffer.bind()

                ## bind the VBO with vertex data
                self.verticesBuffer.bind()
                gl.glEnableVertexAttribArray(0)
                # tell OpenGL that the VBO contains an array of vertices
                # these vertices contain 3 single precision coordinates
                gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

                ## bind the VBO with uv data
                self.uvsBuffer.bind()
                gl.glEnableVertexAttribArray(1)
                gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

                ## bind the VBO with normal data
                self.normalsBuffer.bind()
                gl.glEnableVertexAttribArray(2)
                gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

                ## bind the VBO with barycentrics data
                self.barycentricsBuffer.bind()
                gl.glEnableVertexAttribArray(3)
                gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

                ## draw points from the VBO
                gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)

                ## clean up
                gl.glDisableVertexAttribArray(0)
                gl.glDisableVertexAttribArray(1)
                gl.glDisableVertexAttribArray(2)
                gl.glDisableVertexAttribArray(3)

                gl.glUseProgram(0)
            else :
                print("SHOULD BE RENDERING POINT CLOUD", file=sys.stderr)
                
class GLPolyline() :
    def __init__(self, points, drawColor=np.array([0, 255.0, 0])) :
        self.initDone = False
        
        self.drawColor = drawColor
        self.points = np.copy(points)
        
        self.setGeometryAndBuffers()
        
    def __del__(self) :
        del self.points
        
    def setShaders(self) :
        self.shaders_program = compileShaders(VS_COLOR_NO_SHADE, FS_COLOR_NO_SHADE)
        if np.any(self.shaders_program == None) :
            self.initDone = False
            return
        self.initDone = True
        
    def setGeometryAndBuffers(self) :
        self.indices = np.concatenate([[0], np.arange(1, len(self.points)-1).repeat(2), [len(self.points)-1]]).astype(np.int32)
        self.indexBuffer = glvbo.VBO(self.indices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        self.verticesBuffer = glvbo.VBO(self.points, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
#         if np.sum(self.drawColor - np.array([0, 255, 255])) == 0 :
#             print(self.points, self.indices)
        colorArray = np.repeat(np.array([self.drawColor], np.float32)/np.float32(255.0), len(self.points), 0)
        self.colorBuffer = glvbo.VBO(colorArray, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        
    def draw(self, pvm) :
        if self.initDone :
            gl.glUseProgram(self.shaders_program)
            
            ## send mvp
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shaders_program, "m_pvm"), 1, gl.GL_FALSE, pvm.T)
            ## send camera distance
            gl.glUniform1f(gl.glGetUniformLocation(self.shaders_program, "camera_dist"), np.float32(1.0))
#             print(cameraDist)

            ################ RENDER BODY ################
    
            ## bind the index buffer
            self.indexBuffer.bind()

            ## bind the VBO with vertex data
            self.verticesBuffer.bind()
            gl.glEnableVertexAttribArray(0)
            # tell OpenGL that the VBO contains an array of vertices
            # these vertices contain 3 single precision coordinates
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            
            ## bind the VBO with color data
            self.colorBuffer.bind()
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## draw points from the VBO
            gl.glDrawElements(gl.GL_LINES, len(self.indices), gl.GL_UNSIGNED_INT, None)
#             gl.glDrawElements(gl.GL_POINTS, len(self.indices), gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)

            gl.glUseProgram(0)
 
                
class GLTrajectory() :
    def __init__(self, cameraTrajectoryPoints, cameraIntrinsics, cameraExtrinsics, drawColor=np.array([0, 255.0, 0]), doDrawProjectedPoints = True, doSmoothing = True) :
        self.initDone = False
        
        self.doSmoothing = doSmoothing
        self.trajectorySmoothness = 5
        self.cameraTrajectoryPoints = cameraTrajectoryPoints.astype(np.float32)
        self.worldTrajectoryPoints = np.empty([0, 3], np.float32)
        self.worldTrajectoryDirections = np.empty([0, 3], np.float32)
        self.cameraIntrinsics = cameraIntrinsics.astype(np.float32)
        self.cameraExtrinsics = cameraExtrinsics.astype(np.float32)
        self.drawColor = drawColor
        
        self.projectPoints()

        if doDrawProjectedPoints :
            pointsToDraw = np.copy(self.worldTrajectoryPoints)
#             if len(pointsToDraw) > 10 :
#                 global tmpTrajPoints
#                 tmpTrajPoints = np.copy(pointsToDraw)
        else :
# #             pointsToDraw = np.copy(self.cameraTrajectoryPoints)
#             pointsToDraw = np.dot(np.linalg.inv(cameraIntrinsics), np.concatenate([self.cameraTrajectoryPoints, np.ones([len(self.cameraTrajectoryPoints), 1])], axis=1).T)
#             pointsToDraw = (pointsToDraw[:-1, :]/pointsToDraw[-1, :]).T
# #             pointsToDraw[:, 0] *= (self.cameraIntrinsics[1, -1]/self.cameraIntrinsics[0, -1])
#             print("README BITCH", self.cameraTrajectoryPoints[-1, :])
            width, height = self.cameraIntrinsics[:-1, -1]*2
            pointsToDraw = np.copy(self.cameraTrajectoryPoints).astype(np.float32)
            ## make points between -0.5 and 0.5
            pointsToDraw -= np.array([[width/2.0, height/2.0]], np.float32)
            pointsToDraw /= np.array([[width, height]], np.float32)
            ## scale x-axis to get same aspect ratio as input image
            pointsToDraw[:, 0] *= np.float32(width/height)
            pointsToDraw = np.concatenate([pointsToDraw, np.zeros([len(pointsToDraw), 1], dtype=np.float32)], axis=1)
            
        self.polyline = GLPolyline(pointsToDraw, drawColor)
        
    def __del__(self) :
        del self.cameraTrajectoryPoints, self.worldTrajectoryPoints, self.cameraIntrinsics, self.cameraExtrinsics, self.worldTrajectoryDirections, self.polyline
        
        
    def projectPoints(self) :
        if True :
            inverseT = np.linalg.inv(np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]]))
            self.worldTrajectoryPoints = np.dot(inverseT, np.concatenate([self.cameraTrajectoryPoints, np.ones([len(self.cameraTrajectoryPoints), 1], np.float32)], axis=1).T)
            self.worldTrajectoryPoints /= self.worldTrajectoryPoints[-1, :]
            self.worldTrajectoryPoints[-1, :] = 0
            self.worldTrajectoryPoints = self.worldTrajectoryPoints.T.astype(np.float32)
        else :
            ### HACK : USE HOMOGRAPHY INSTEAD OF CAMERA MATRICES ###
            homography = np.array([[11.6261525276, 185.257281938, 818.145590521],
                                   [-24.7005245641, 14.5276400234, 272.499203107],
                                   [-0.197073111956, 0.178268418299, 1.0]])
            self.worldTrajectoryPoints = np.dot(np.linalg.inv(homography), np.concatenate([self.cameraTrajectoryPoints, np.ones([len(self.cameraTrajectoryPoints), 1], np.float32)], axis=1).T)
            self.worldTrajectoryPoints /= self.worldTrajectoryPoints[-1, :]
            self.worldTrajectoryPoints[-1, :] = 0
            self.worldTrajectoryPoints = self.worldTrajectoryPoints.T.astype(np.float32)
        
        if self.doSmoothing :
            ## smooth trajectory
            self.worldTrajectoryPoints = np.array([spimg.filters.gaussian_filter1d(self.worldTrajectoryPoints[:, 0], self.trajectorySmoothness, axis=0), 
                                                   spimg.filters.gaussian_filter1d(self.worldTrajectoryPoints[:, 1], self.trajectorySmoothness, axis=0), 
                                                   spimg.filters.gaussian_filter1d(self.worldTrajectoryPoints[:, 2], self.trajectorySmoothness, axis=0)]).T.astype(np.float32)

            ## reproject points into image space after smoothing
            T = np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]])
            self.cameraTrajectoryPoints = np.dot(T, np.concatenate([self.worldTrajectoryPoints[:, :-1], np.ones([len(self.worldTrajectoryPoints), 1])], axis=1).T)
            self.cameraTrajectoryPoints = (self.cameraTrajectoryPoints[:-1, :]/self.cameraTrajectoryPoints[-1, :]).T
            
#             print(self.camera)

        self.worldTrajectoryDirections = np.array([self.worldTrajectoryPoints[i, :]-self.worldTrajectoryPoints[j, :] for i, j in zip(xrange(1, len(self.worldTrajectoryPoints)),
                                                                                                                                     xrange(0, len(self.worldTrajectoryPoints)-1))])
        self.worldTrajectoryDirections /= np.linalg.norm(self.worldTrajectoryDirections, axis=1).reshape([len(self.worldTrajectoryDirections), 1])
        
        if len(self.worldTrajectoryPoints) > 10 :
            np.save("tmp_trajectory_3D.npy", {"trajectoryPointsCameraSpace":self.cameraTrajectoryPoints, "trajectoryPointsWorldSpace":self.worldTrajectoryPoints, 
                                              "trajectoryDirectionsWorldSpace":self.worldTrajectoryDirections, "intrinsics":self.cameraIntrinsics, "extrinsics":self.cameraExtrinsics})
        
    def setShaders(self) :
        self.polyline.setShaders()
        self.initDone = True
        
    def draw(self, pvm) :
        if self.initDone :
            self.polyline.draw(pvm)

class GLBillboard() :
    def __init__(self, img, scale, modelMat=np.eye(4, dtype=np.float32), isFrontoParallel = False, rotateAboutPlaneNormal = None, normalizeToPixelSize = False) :
        self.initDone = False
        self.textureChanged = False
        self.pixelSize = 0.01
        
        self.scale = scale
        self.isFrontoParallel = isFrontoParallel
        self.doRotateAboutPlaneNormal = rotateAboutPlaneNormal != None
        self.rotateAboutPlaneNormal = np.copy(rotateAboutPlaneNormal)
        self.normalizeToPixelSize = normalizeToPixelSize
        self.modelMat = np.copy(modelMat)
        if self.isFrontoParallel or self.doRotateAboutPlaneNormal :
            ## remove any rotations as it won't work if it has rotations
            self.modelMat[:-1, :-1] = np.eye(3, dtype=np.float32)            
        
        self.setTexture(img)
        self.setGeometryAndBuffers()
        
    def __del__(self) :
        del self.tex, self.vertices, self.indices, self.uvs
        
    def setScale(self, scale) :
        self.scale = scale
        self.setGeometryAndBuffers()
        
    def setTexture(self, img) :
        possibleTexSizes = np.array([128, 256, 512, 1204, 2048])
        texSize = possibleTexSizes[np.argwhere(possibleTexSizes-np.max(img.shape[0:2]) > 0).flatten()[0]]
        self.tex = np.zeros([texSize, texSize, 4], np.int8)
        self.tex[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        ## set alpha channel if inout image is just rgb
        if True or img.shape[2] == 3 :
            self.tex[:img.shape[0], :img.shape[1], -1] = np.int8(255)
            
        self.textureChanged = True
        [self.maxV, self.maxU] = np.array(img.shape[:2], np.float32)/np.array(self.tex.shape[:2], np.float32)
        self.aspectRatio = float(img.shape[1])/float(img.shape[0])
        
        if self.normalizeToPixelSize :
            self.scale = self.pixelSize*img.shape[1]
            top, left, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)
#             print(top, left, width, height)
            self.scale = float(img.shape[1])/float(1280)*5.36229266*(width/1280.0)
        
        self.setGeometryAndBuffers()
        
    def setGeometryAndBuffers(self) :
        ## IMAGE PLANE ##
        self.vertices = np.dot(np.array([[self.scale, 0, 0],
                                         [0, self.scale, 0],
                                         [0, 0, 1.0]], np.float32),
                               np.array([[self.aspectRatio/2.0, -0.5, 0.0], [self.aspectRatio/2.0, 0.5, 0.0], [-self.aspectRatio/2.0, -0.5, 0.0],
                                         [self.aspectRatio/2.0, 0.5, 0.0], [-self.aspectRatio/2.0, 0.5, 0.0], [-self.aspectRatio/2.0, -0.5, 0.0]], np.float32).T).T
        self.indices = np.array([0, 1, 2, 3, 4, 5], np.int32)
        self.uvs = np.array([[self.maxU, self.maxV], [self.maxU, 0.0], [0.0, self.maxV],
                             [self.maxU, 0.0], [0.0, 0.0], [0.0, self.maxV]], np.float32)

        self.verticesBuffer = glvbo.VBO(self.vertices, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.indexBuffer = glvbo.VBO(self.indices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        self.uvsBuffer = glvbo.VBO(self.uvs, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        
    def setShaders(self) :
        self.texturedGeometryShadersProgram = compileShaders(VS_IMAGE, FS_IMAGE)
        if np.any(self.texturedGeometryShadersProgram == None) :
            self.initDone = False
            return
        self.initDone = True
                
        self.textureID = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
        
    def draw(self, projectionMat, viewMat, rotDir=None, rotAngle=None) :
        if self.initDone :
            gl.glUseProgram(self.texturedGeometryShadersProgram)
            
            ## find rotation to apply so that billboard is frontoparallel
            rotMat = np.eye(4, dtype=np.float32)
            if self.isFrontoParallel :
                pos, norm = getWorldSpacePosAndNorm(self.modelMat)
                cameraPos, cameraNorm = getWorldSpacePosAndNorm(np.linalg.pinv(viewMat), np.array([[0.0], [0.0], [-1.0], [1.0]]))
                cameraPos, cameraUp = getWorldSpacePosAndNorm(np.linalg.pinv(viewMat), np.array([[0.0], [1.0], [0.0], [1.0]]))
                
                lookAt = cameraPos-pos
                lookAt /= np.linalg.norm(lookAt)
                
                rightVec = np.cross(cameraUp, lookAt)
                upVec = np.cross(lookAt, rightVec)
                
                rotMat = np.array([[rightVec[0], upVec[0], lookAt[0], 0],
                                   [rightVec[1], upVec[1], lookAt[1], 0],
                                   [rightVec[2], upVec[2], lookAt[2], 0],
                                   [0, 0, 0, 1]])
            elif self.doRotateAboutPlaneNormal :
                pos, norm = getWorldSpacePosAndNorm(self.modelMat)
                cameraPos, cameraNorm = getWorldSpacePosAndNorm(np.linalg.pinv(viewMat), np.array([[0.0], [0.0], [-1.0], [1.0]]))
                
                lookAt = cameraPos-pos
                lookAt /= np.linalg.norm(lookAt)
                ## project lookAt onto plane
                lookAt = lookAt - np.dot(lookAt, self.rotateAboutPlaneNormal)*self.rotateAboutPlaneNormal
                lookAt /= np.linalg.norm(lookAt)
                
                upVec = np.copy(self.rotateAboutPlaneNormal)
                rightVec = np.cross(upVec, lookAt)
                rightVec /= np.linalg.norm(rightVec)
                
                rotMat = np.array([[rightVec[0], upVec[0], lookAt[0], 0],
                                   [rightVec[1], upVec[1], lookAt[1], 0],
                                   [rightVec[2], upVec[2], lookAt[2], 0],
                                   [0, 0, 0, 1]])
            
            if rotDir != None and rotAngle != None :
                rotMat = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(rotAngle, lookAt*rotDir)), rotMat)

            ## send mvp
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.texturedGeometryShadersProgram, "m_pvm"), 1, gl.GL_FALSE,
                                  np.dot(projectionMat, np.dot(viewMat, np.dot(self.modelMat, rotMat))).T)
            ## send model
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.texturedGeometryShadersProgram, "m_m"), 1, gl.GL_FALSE, self.modelMat.T)

            if self.textureChanged :
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.tex.shape[1], self.tex.shape[0], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self.tex)
                gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
                gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
                gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
                gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
                gl.glTexParameterf(gl.GL_TEXTURE_2D, tfa.GL_TEXTURE_MAX_ANISOTROPY_EXT, gl.glGetFloatv(tfa.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT))
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                self.textureChanged = False

            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureID)
            gl.glUniform1i(gl.glGetUniformLocation(self.texturedGeometryShadersProgram, "texture_sampler"), 0)

            ## bind the index buffer
            self.indexBuffer.bind()

            ## bind the VBO with vertex data
            self.verticesBuffer.bind()
            gl.glEnableVertexAttribArray(0)
            # tell OpenGL that the VBO contains an array of vertices
            # these vertices contain 3 single precision coordinates
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## bind the VBO with uv data
            self.uvsBuffer.bind()
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## draw points from the VBO
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)

            gl.glUseProgram(0)
            
            
class GLCameraFrustum() :
    def __init__(self, modelMat, billboardImage, scale) :
        self.initFailed = False
        self.drawBillboard = True
        self.modelMat = modelMat
        self.scale = scale
        self.zDir = 0.6
        
        ## translate by zDir and rotate by 180 along x axis so that it faces the camera center
        tMat = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, self.zDir],
                         [0, 0, 0, 1]], np.float32)
        self.imagePlaneBillboard = GLBillboard(billboardImage, scale, np.dot(self.modelMat, tMat))
        
        self.setGeometryAndBuffers()
        
    def __del__(self) :
        del self.indices, self.vertices, self.imagePlaneBillboard
        
    def setScale(self, scale) :
        self.scale = scale
        self.imagePlaneBillboard.setScale(scale)
        self.setGeometryAndBuffers()
                    
    def toggleShowFrustumBillboard(self) :
        self.drawBillboard = not self.drawBillboard
        
    def setImage(self, image) :
        self.imagePlaneBillboard.setTexture(image)
        
    def setShaders(self) :
        self.colorNoShadeShadersProgram = compileShaders(VS_COLOR_NO_SHADE, FS_COLOR_NO_SHADE)
        if np.any(self.colorNoShadeShadersProgram == None) :
            self.initDone = False
            return        
        self.imagePlaneBillboard.setShaders()
        self.initDone = True
        
    def setGeometryAndBuffers(self) :
        self.vertices = np.dot(np.array([[self.scale, 0, 0],
                                         [0, self.scale, 0],
                                         [0, 0, 1.0]], np.float32),
                               np.array([[-self.imagePlaneBillboard.aspectRatio/2.0, -0.5, self.zDir], [-self.imagePlaneBillboard.aspectRatio/2.0, 0.5, self.zDir],
                                         [self.imagePlaneBillboard.aspectRatio/2.0, 0.5, self.zDir], [self.imagePlaneBillboard.aspectRatio/2.0, -0.5, self.zDir], [0.0, 0.0, 0.0],
                                         [0.0, 0.0, self.zDir], [0.0, -1.0, 0.0]], np.float32).T).T
        self.indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 4, 2, 4, 3, 4, 4, 5, 4, 6], np.int32)
        self.vertices = self.vertices[list(self.indices), :].astype(np.float32)
        self.indices = np.arange(len(self.vertices)).astype(np.int32)

        self.indexBuffer = glvbo.VBO(self.indices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        self.verticesBuffer = glvbo.VBO(self.vertices, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        colorArray = np.repeat(np.array([[1, 0, 0]], np.float32), len(self.vertices), 0)
        self.colorBuffer = glvbo.VBO(colorArray, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        
    def draw(self, projectionMat, viewMat) :
        if self.initDone :
            if self.drawBillboard :
                self.imagePlaneBillboard.draw(projectionMat, viewMat)
            
            gl.glUseProgram(self.colorNoShadeShadersProgram)

            ## send mvp
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.colorNoShadeShadersProgram, "m_pvm"), 1, gl.GL_FALSE, np.dot(projectionMat, np.dot(viewMat, self.modelMat)).T)
            ## send camera distance
            gl.glUniform1f(gl.glGetUniformLocation(self.colorNoShadeShadersProgram, "camera_dist"), np.float32(1.0))
    #             print(cameraDist)

            ################ RENDER BODY ################

            ## bind the index buffer
            self.indexBuffer.bind()

            ## bind the VBO with vertex data
            self.verticesBuffer.bind()
            gl.glEnableVertexAttribArray(0)
            # tell OpenGL that the VBO contains an array of vertices
            # these vertices contain 3 single precision coordinates
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## bind the VBO with color data
            self.colorBuffer.bind()
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## draw points from the VBO
            gl.glDrawElements(gl.GL_LINES, len(self.indices), gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)

            gl.glUseProgram(0)

# <codecell>

class GLFilmedObject() :
    def __init__(self, objectLoc, cameraIntrinsics, cameraExtrinsics, isDistorted, distortionCoeff, originalIntrinsics, footprintScale=0.5, footprintAspectRatio=1.5) :
        self.initDone = False
        self.footprintScale = footprintScale
        self.footprintAspectRatio = footprintAspectRatio
        self.objectData = np.load(objectLoc).item()
        print("LOADED", self.objectData[DICT_SEQUENCE_NAME])
        self.cameraIntrinsics = cameraIntrinsics
        self.cameraExtrinsics = cameraExtrinsics
        
        ## rotate by 180 along z axis and translate up by 1
        self.modelMat = np.array([[-1, 0, 0, 2],
                                  [0, -1, 0, -3],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]], np.float32)

        self.forwardDir = np.array([[1.0], [0.0], [0.0], [1.0]]) ## model space
#         frameSubset = [5, -5]
        frameSubset = [30, -50]
        
        ## need a -1 because patches does not contain the empty frame whereas the trajectory points do
        self.trajectoryPoints = np.array([self.objectData[DICT_BBOX_CENTERS][key] for key in sort(self.objectData[DICT_BBOX_CENTERS].keys())])[frameSubset[0]:frameSubset[1]-1, :]
        if isDistorted :
            self.trajectoryPoints = cv2.undistortPoints(self.trajectoryPoints.reshape((1, len(self.trajectoryPoints), 2)), originalIntrinsics, distortionCoeff, P=self.cameraIntrinsics)[0, :, :]
        
        self.trajectory = GLTrajectory(self.trajectoryPoints, cameraIntrinsics, cameraExtrinsics, self.objectData[DICT_REPRESENTATIVE_COLOR])
        
        global tmpTrajectoryCameraSpace
        tmpTrajectoryCameraSpace = np.copy(self.trajectory.cameraTrajectoryPoints)
        
        ## move object to the first image on its original trajectort and orient properly
        trajPointIdx = 0
        objPos, objFDir = getWorldSpacePosAndNorm(self.modelMat, self.forwardDir)
        adjustAngle = np.arccos(np.clip(np.dot(objFDir, self.trajectory.worldTrajectoryDirections[trajPointIdx, :]), -1, 1))
        adjustAxis = np.cross(self.trajectory.worldTrajectoryDirections[trajPointIdx, :], objFDir)
        adjustAxis /= np.linalg.norm(adjustAxis)
        self.modelMat = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis)), self.modelMat)
        self.modelMat[:-1, -1] = self.trajectory.worldTrajectoryPoints[trajPointIdx, :]
        
#         patchesLoc = "/".join(self.objectData[DICT_PATCHES_LOCATION].split("/")[:-1])+"/camera_adjusted_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
        patchesLoc = "/".join(self.objectData[DICT_PATCHES_LOCATION].split("/")[:-1])+"/camera_adjusted_plus_scale_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
        if True and os.path.isfile(patchesLoc) :
            print("using patches from:", patchesLoc)
            self.patches = np.load(patchesLoc).item()
        else :
            print("using patches from:", self.objectData[DICT_PATCHES_LOCATION])
            self.patches = np.load(self.objectData[DICT_PATCHES_LOCATION]).item()
        self.sortedPatchKeys = np.sort(self.patches.keys())[frameSubset[0]:frameSubset[1]]
        
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.0, np.dot(self.modelMat, np.array([[-1, 0, 0, 0],
#                                                                                                       [0, -1, 0, 0],
#                                                                                                       [0, 0, 1, 0],
#                                                                                                       [0, 0, 0, 1]], np.float32)), False, None, True)
        
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.0, self.modelMat, False, np.array([0, 0, 1], np.float32), True)
        self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.0, self.modelMat, True, None, True)
        
        self.setGeometryAndBuffers()
        
        ## find directions from center of the car to the center of the camera
        cameraPos = getWorldSpacePosAndNorm(np.linalg.inv(self.cameraExtrinsics), posOnly=True)
        self.pointToCameraDirectionsWorldSpace = cameraPos.reshape([1, 3]).astype(np.float32) - self.trajectory.worldTrajectoryPoints
        self.pointToCameraDistances = np.linalg.norm(self.pointToCameraDirectionsWorldSpace, axis=1).astype(np.float32)
        self.pointToCameraDirectionsWorldSpace /= self.pointToCameraDistances.reshape([len(self.pointToCameraDistances), 1])
        
        self.cameraToObjectDirectionsObjSpace = np.zeros([len(self.trajectory.worldTrajectoryDirections), 3])
        for i, direction in enumerate(self.trajectory.worldTrajectoryDirections) :
            rotAxis = np.cross(np.array([1, 0, 0]), direction)
            rotAxis /= np.linalg.norm(rotAxis)
            rotAngle = np.arccos(np.dot(direction, np.array([1, 0, 0])))

            M = quaternionTo4x4Rotation(angleAxisToQuaternion(rotAngle, rotAxis))
            ## here it works to get the dir like this because the origin is 0 and I'm rotating about it what I would really have to do is rotate the point translated by the direction and then take diff between it and origin 
            rotatedDir = np.dot(M, np.array([[self.pointToCameraDirectionsWorldSpace[i, 0], self.pointToCameraDirectionsWorldSpace[i, 1], self.pointToCameraDirectionsWorldSpace[i, 2], 1]]).T)
            rotatedDir = rotatedDir[:-1, 0]/rotatedDir[-1, 0]
            rotatedDir /= np.linalg.norm(rotatedDir)
            self.cameraToObjectDirectionsObjSpace[i, :] = -rotatedDir
            
            ## this turns the camera towards the object
            adjustCamPos, adjustCamNorm = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics))
            adjustAxis = np.cross(-self.pointToCameraDirectionsWorldSpace[i, :], adjustCamNorm)
            adjustAxis /= np.linalg.norm(adjustAxis)
            adjustAngle = np.arccos(np.clip(np.dot(adjustCamNorm, -self.pointToCameraDirectionsWorldSpace[i, :]), -1, 1))
            adjustM = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))

            camMat = np.eye(4)
            camMat[:-1, -1] = rotatedDir
            camMat[:-1, :-1] = np.dot(adjustM[:-1, :-1], np.linalg.inv(cameraExtrinsics)[:-1, :-1])

            ## this rotates camera to align with ground plane (and the car itself)
            _, adjustCamRightVec2 = getWorldSpacePosAndNorm(camMat, np.array([[1, 0, 0, 1]], float).T)
            _, adjustCamUpVec2 = getWorldSpacePosAndNorm(camMat, np.array([[0, -1, 0, 1]], float).T)
            _, adjustCamNorm2 = getWorldSpacePosAndNorm(camMat)
            adjustAxis2 = np.copy(adjustCamNorm2)
    #         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, adjustCamRightVec2*np.array([1, 1, 0], float)), -1, 1)) ## aligns camera right vector to ground plane
    #         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, np.array([1, 0, 0], float)), -1, 1)) ## not sure what this does
            trajDir = self.trajectory.cameraTrajectoryPoints[i, :]-self.trajectory.cameraTrajectoryPoints[i+1, :]
            trajDir /= np.linalg.norm(trajDir) ## CAREFUL HERE AS THE NORM CAN BE 0
            adjustAngle2 = np.arccos(np.clip(np.dot(trajDir, np.array([1, 0], float)), -1, 1)) ## align camera space direction to x axis (does it even make sense?)
            if np.cross(trajDir, np.array([1, 0], float)) < 0 :
                adjustAxis2 *= -1.0


            adjustM2 = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle2, adjustAxis2))
            camMat[:-1, :-1] = np.dot(M[:-1, :-1], np.dot(adjustM2[:-1, :-1], camMat[:-1, :-1]))
            
        global tmpDirections
        tmpDirections = np.copy(self.cameraToObjectDirectionsObjSpace)
        
    def __del__(self) :
        del self.objectData, self.cameraIntrinsics, self.cameraExtrinsics, self.trajectory, self.patches, self.sortedPatchKeys
        del self.billboard, self.footprintVertices, self.footprintIndices, self.arrowIndices, self.arrowVertices
        del self.pointToCameraDirectionsWorldSpace, self.pointToCameraDistances, self.cameraToObjectDirectionsObjSpace, self.trajectoryPoints
    
    def getImageFromPatch(self, idx) :
        ## find offset between location of the trajectory point and the patch center and then center the patch on the point by padding img
        patch = self.patches[self.sortedPatchKeys[idx]]
#         ## top_left_pos in patch is wrt the original image size whereas the trajectory points are wrt the undistorted image so compensate
#         patchTopLeft = (patch['top_left_pos']+((np.array([936, 1664])-np.array([720, 1280]))/2))[::-1]
#         pointLocation = np.round(self.trajectory.cameraTrajectoryPoints[idx, :]-patchTopLeft).astype(int)
#         print("TRAJ POINT, PATCH CENTER, TOP LEFT, POINT IN PATCH", self.trajectory.cameraTrajectoryPoints[idx, :], patch['patch_size'][::-1]/2, patchTopLeft, pointLocation)
        img = np.zeros([patch['patch_size'][0], patch['patch_size'][1], 4], dtype=np.int8)
        img[patch['visible_indices'][:, 0], patch['visible_indices'][:, 1], :] = patch['sprite_colors']
        
#         print("PATCH SIZE", patch['patch_size'])
        
        return img[:, :, [2, 1, 0, 3]]
    
    def setShaders(self) :
        self.colorNoShadeShadersProgram = compileShaders(VS_COLOR_NO_SHADE, FS_COLOR_NO_SHADE)
        if np.any(self.colorNoShadeShadersProgram == None) :
            self.initDone = False
            return
        
        self.trajectory.setShaders()
        self.billboard.setShaders()
        self.initDone = True
        
    def setGeometryAndBuffers(self) :
        forwardDirPos = [self.footprintAspectRatio/2.0, 0.5, 0.0]*self.forwardDir[:-1, 0]
        scaleMat = np.array([[self.footprintScale, 0, 0],
                             [0, self.footprintScale, 0],
                             [0, 0, 1.0]], np.float32)
        ## FOOTPRINT BOX ##
        self.footprintVertices = np.dot(scaleMat, np.array([[-self.footprintAspectRatio/2.0, -0.5, 0.0], [-self.footprintAspectRatio/2.0, 0.5, 0.0],
                                                            [self.footprintAspectRatio/2.0, 0.5, 0.0], [self.footprintAspectRatio/2.0, -0.5, 0.0], [0.0, 0.0, 0.0],
                                                            forwardDirPos], np.float32).T).T
        self.footprintIndices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5], np.int32)
        self.footprintVertices = self.footprintVertices[list(self.footprintIndices), :].astype(np.float32)
        self.footprintIndices = np.arange(len(self.footprintVertices)).astype(np.int32)

        self.footprintIndexBuffer = glvbo.VBO(self.footprintIndices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        self.footprintVerticesBuffer = glvbo.VBO(self.footprintVertices, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        colorArray = np.repeat(np.array([[1, 0, 0]], np.float32), len(self.footprintVertices), 0)
        self.footprintColorBuffer = glvbo.VBO(colorArray, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        
        ## FORWARD DIR ARROW ##
        arrowMesh = pyassimp.load("arrowTop.obj").meshes[0]
        self.arrowVertices = arrowMesh.vertices.astype(np.float32)*np.float32(self.footprintScale*5.0)
        forwardDirPos = np.dot(scaleMat, forwardDirPos)
        ## move to position and rotate by 90 along z axis
        tMat = np.array([[0, 1, 0, forwardDirPos[0]],
                         [-1, 0, 0, forwardDirPos[1]],
                         [0, 0, 1, forwardDirPos[2]],
                         [0, 0, 0, 1]], np.float32)
        
        self.arrowVertices = np.dot(tMat, np.concatenate([self.arrowVertices, np.ones([len(self.arrowVertices), 1])], axis=1).T)
        self.arrowVertices = (self.arrowVertices[:-1, :]/self.arrowVertices[-1, :]).T
        self.arrowIndices = arrowMesh.faces.flatten()
        
        colorArray = np.repeat(np.array([[1, 0, 0]], np.float32), len(self.arrowVertices), 0)        
        self.arrowVerticesBuffer = glvbo.VBO(self.arrowVertices.astype(np.float32), gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.arrowColorsBuffer = glvbo.VBO(colorArray.astype(np.float32), gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.arrowIndexBuffer = glvbo.VBO(self.arrowIndices.astype(np.int32), gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        
    def drawFootprint(self, projectionMat, viewMat) :
        gl.glUseProgram(self.colorNoShadeShadersProgram)

        ## send mvp
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.colorNoShadeShadersProgram, "m_pvm"), 1, gl.GL_FALSE, np.dot(projectionMat, np.dot(viewMat, self.modelMat)).T)
        ## send camera distance
        gl.glUniform1f(gl.glGetUniformLocation(self.colorNoShadeShadersProgram, "camera_dist"), np.float32(1.0))
#             print(cameraDist)

        ################ RENDER FOOTPRINT BOX ################

        ## bind the index buffer
        self.footprintIndexBuffer.bind()

        ## bind the VBO with vertex data
        self.footprintVerticesBuffer.bind()
        gl.glEnableVertexAttribArray(0)
        # tell OpenGL that the VBO contains an array of vertices
        # these vertices contain 3 single precision coordinates
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        ## bind the VBO with color data
        self.footprintColorBuffer.bind()
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        ## draw points from the VBO
        gl.glDrawElements(gl.GL_LINES, len(self.footprintIndices), gl.GL_UNSIGNED_INT, None)

        ## clean up
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        
        ################ RENDER FORWARD DIR ARROW ################

        ## bind the index buffer
        self.arrowIndexBuffer.bind()

        ## bind the VBO with vertex data
        self.arrowVerticesBuffer.bind()
        gl.glEnableVertexAttribArray(0)
        # tell OpenGL that the VBO contains an array of vertices
        # these vertices contain 3 single precision coordinates
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        ## bind the VBO with color data
        self.arrowColorsBuffer.bind()
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        ## draw points from the VBO
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.arrowIndices), gl.GL_UNSIGNED_INT, None)

        ## clean up
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)

        gl.glUseProgram(0)
        
    def draw(self, projectionMat, viewMat) :
        if self.initDone :
            top, left, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)
            
            camPos = getWorldSpacePosAndNorm(np.linalg.inv(viewMat), posOnly=True)
            objPos = getWorldSpacePosAndNorm(self.modelMat, posOnly=True)

            cameraToObjDir = objPos-camPos
            cameraToObjDir /= np.linalg.norm(cameraToObjDir)
#                 cameraToObjDir = np.dot(np.linalg.inv(self.modelMat), np.concatenate([objPos+cameraToObjDir, [1]]).reshape([4, 1])).flatten()
#                 cameraToObjDir = cameraToObjDir[:-1]/cameraToObjDir[-1]
            ## in object space from world space
            cameraPosObjSpace = np.dot(np.linalg.inv(self.modelMat), np.concatenate([objPos-cameraToObjDir, [1]]).reshape([4, 1])).flatten()
            cameraPosObjSpace = cameraPosObjSpace[:-1]/cameraPosObjSpace[-1]
            cameraToObjDir = np.zeros(3)-cameraPosObjSpace
            cameraToObjDir /= np.linalg.norm(cameraToObjDir)

            directionAngleDistances = np.abs(np.arccos(np.clip(np.dot(cameraToObjDir.reshape([1, 3]), self.cameraToObjectDirectionsObjSpace.T), -1.0, 1.0))*180.0/np.pi)

#                 print(camPos, objPos, directionAngleDistances.shape, int(np.argmin(directionAngleDistances).flatten()))
            self.frameToUseIdx = int(np.argmin(directionAngleDistances).flatten())
#                 print(self.frameToUseIdx)

            ## some movingDirection is nan because of 2 consecutive points being the same but because of the smoothing it's pretty unlikely
            if self.frameToUseIdx < len(self.trajectory.cameraTrajectoryPoints)-1 :
                movingDirection = self.trajectory.cameraTrajectoryPoints[self.frameToUseIdx+1, :]-self.trajectory.cameraTrajectoryPoints[self.frameToUseIdx, :]
                movingDirection /= np.linalg.norm(movingDirection)
            else :
                movingDirection = self.trajectory.cameraTrajectoryPoints[self.frameToUseIdx, :]-self.trajectory.cameraTrajectoryPoints[self.frameToUseIdx-1, :]
                movingDirection /= np.linalg.norm(movingDirection)
            ## cameraTrajectoryPoints are not in the same aspect ratio as the objPos and objDirPos because the first are defined using the original camera and the second are defined using the opengl camera
            ## so need to compensate for that by (well flipping y first because of opencv convention) removing the distortions due to the viewport aspect ratio
            movingDirection[1] *= -1.0
            movingDirection[0] *= 1.0/(float(width)/float(height))
            movingDirection /= np.linalg.norm(movingDirection)
            
            
            objPosInCameraSpace = np.dot(np.dot(projectionMat, viewMat), np.concatenate([objPos, [1]]).reshape([4, 1]))
            objPosInCameraSpace = objPosInCameraSpace[:-1, 0]/objPosInCameraSpace[-1, 0]
            objPosInClipSpace = np.array([(objPosInCameraSpace[0]+1.0)*width/2.0, (1.0-objPosInCameraSpace[1])*height/2.0])

            objDirPosInWorldSpace = np.dot(self.modelMat, self.forwardDir)
            objDirPosInWorldSpace = objDirPosInWorldSpace[:3, 0]/objDirPosInWorldSpace[3, 0]
            objDirPosInCameraSpace = np.dot(np.dot(projectionMat, viewMat), np.concatenate([objDirPosInWorldSpace, [1]]).reshape([4, 1]))
            objDirPosInCameraSpace = objDirPosInCameraSpace[:-1, 0]/objDirPosInCameraSpace[-1, 0]
            objDirPosInClipSpace = np.array([(objDirPosInCameraSpace[0]+1.0)*width/2.0, (1.0-objDirPosInCameraSpace[1])*height/2.0])

            objMovingDirectionInCameraSpace = objDirPosInCameraSpace[:-1]-objPosInCameraSpace[:-1]#objDirPosInClipSpace-objPosInClipSpace
            objMovingDirectionInCameraSpace /= np.linalg.norm(objMovingDirectionInCameraSpace)
            
            objDirLine = GLPolyline(np.concatenate([np.concatenate([[objPosInCameraSpace[:-1]], [objPosInCameraSpace[:-1]+objMovingDirectionInCameraSpace]]), np.zeros([2, 1])], axis=1).astype(np.float32),
                                    drawColor=array([   0.,  255.,    255.]))
            objDirLine.setShaders()
            movingDirLine = GLPolyline(np.concatenate([np.concatenate([[objPosInCameraSpace[:-1]], [objPosInCameraSpace[:-1]+movingDirection]]), np.zeros([2, 1])], axis=1).astype(np.float32),
                                       drawColor=array([   255.,  255.,    0.]))
            movingDirLine.setShaders()
            
            
            movingDirection[0] *= (float(width)/float(height))
            movingDirection /= np.linalg.norm(movingDirection)
            objMovingDirectionInCameraSpace[0] *= (float(width)/float(height))
            objMovingDirectionInCameraSpace /= np.linalg.norm(objMovingDirectionInCameraSpace)
            
            rotDir = np.cross(objMovingDirectionInCameraSpace, movingDirection)
            rotDir /= np.linalg.norm(rotDir)
            rotAngle = np.arccos(np.clip(np.dot(objMovingDirectionInCameraSpace, movingDirection), -1.0, 1.0))
            

            self.billboard.setTexture(self.getImageFromPatch(self.frameToUseIdx))
            global tmpDirectionAngleDistances
            tmpDirectionAngleDistances = np.copy(directionAngleDistances)
            
            self.billboard.draw(projectionMat, viewMat, rotDir=rotDir, rotAngle=rotAngle)

#             print("(pos, posFDir, camFDir, camProjFDir)", pos, posFDir, camFDir, camProjFDir)
    
            isDepthTestOn = bool(gl.glGetBooleanv(gl.GL_DEPTH_TEST))
            gl.glDisable(gl.GL_DEPTH_TEST)
            self.trajectory.draw(np.dot(projectionMat, viewMat))
            self.drawFootprint(projectionMat, viewMat)
            objDirLine.draw(np.eye(4, dtype=np.float32))
            movingDirLine.draw(np.eye(4, dtype=np.float32))
#             objDirLine.draw(np.dot(projectionMat, viewMat))
#             movingDirLine.draw(np.dot(projectionMat, viewMat))
            if isDepthTestOn :
                gl.glEnable(gl.GL_DEPTH_TEST)

# <codecell>

VIDEO_PLAYBACK_FPS = 15
class GLFilmedScene() :
    def __init__(self, filmedSceneLoc, videoFPS=30, downsampleRate=4, frustumScale=0.5, pointSize=1.0) :
        self.initDone = False
        self.initFailed = False
        self.videoFPS = videoFPS
        self.playbackFrameSkip = videoFPS/VIDEO_PLAYBACK_FPS
        self.downsampleRate = downsampleRate
#         self.frustumScale = frustumScale
        self.pointSize = np.float32(pointSize)
        self.downsampledLoc = filmedSceneLoc+"downsampledSet-"+np.string_(self.downsampleRate)+"x.npy"
        self.distortionCoeff = np.zeros(5)
        
        
#         self.cameraExtrinsics = np.array([[0.910911599349, 0.406799823245, 0.0689489809887, 1.04144313517],
#                                           [0.201452403084, -0.292662295856, -0.934754358041, -0.491083466897],
#                                           [-0.360079140556, 0.865368525191, -0.348540281954, 3.93563077185],
#                                           [0.0, 0.0, 0.0, 1.0]], np.float32)
        
        ## extrinsics after fitting a sligthly different square found by checking the vanishing line and taking into account the aspect ratio of the rectangle I fit (rather than just assume it is a square)
        self.cameraExtrinsics = np.array([[0.820045839796, 0.57100067645, -0.0385103638868, 1.67922756789],
                                          [0.22275752409, -0.380450047102, -0.897572753108, -0.831720502302],
                                          [-0.527165918942, 0.727472328789, -0.439181175316, 6.76268742928],
                                          [0.0, 0.0, 0.0, 1.0]], np.float32)
        
        self.modelMat = np.linalg.inv(self.cameraExtrinsics)
        
        self.cameraIntrinsics = np.array([[702.736053, 0.0, 640.0],
                                          [0.0, 702.736053, 360.0],
                                          [0.0, 0.0, 1.0]])
        originalIntrinsics = np.copy(self.cameraIntrinsics)
        
        self.medianImage = np.array(Image.open(filmedSceneLoc+"median.png"), np.uint8)
        
        if True :
            self.isDistorted = True
            self.distortionParameter = -0.19
            self.distortionRatio = -0.19
            self.distortionCoeff = np.array([self.distortionParameter, self.distortionParameter*self.distortionRatio, 0.0, 0.0, 0.0])

            frameSize = np.array([self.medianImage.shape[1], self.medianImage.shape[0]])

            ## undistort image
            sizeDelta = 0.3
            newFrameSize = (frameSize*(1+sizeDelta)).astype(int)
            newIntrinsics = np.copy(self.cameraIntrinsics)
            newIntrinsics[0, 2] += self.medianImage.shape[1]*sizeDelta/2.0
            newIntrinsics[1, 2] += self.medianImage.shape[0]*sizeDelta/2.0

            map1, map2 = cv2.initUndistortRectifyMap(self.cameraIntrinsics, self.distortionCoeff, None, newIntrinsics, tuple(newFrameSize), cv2.CV_16SC2)
            self.medianImage = cv2.remap(self.medianImage, map1, map2, cv2.INTER_LINEAR)
            self.cameraIntrinsics = np.copy(newIntrinsics)
        else :
            self.isDistorted = False
            
        self.projectImageGridPoints(self.medianImage)
        
        self.filmedFramesLocs = np.sort(glob.glob(filmedSceneLoc+"frame-*.png"))
        if len(self.filmedFramesLocs) > 0 :
            self.currentFrame = 0
            if os.path.isfile(self.downsampledLoc) :
                print("LOADING", self.downsampledLoc); sys.stdout.flush()
                self.allFrames = np.load(self.downsampledLoc)
            else :
                print("WRITING", self.downsampledLoc); sys.stdout.flush()
                if self.isDistorted :
                    firstImg = Image.fromarray(cv2.remap(np.array(Image.open(self.filmedFramesLocs[0])), map1, map2, cv2.INTER_LINEAR).astype(numpy.uint8))
                else :
                    firstImg = Image.open(self.filmedFramesLocs[0])
                firstImg.thumbnail((firstImg.width/self.downsampleRate, firstImg.height/self.downsampleRate), Image.ANTIALIAS)
                firstImg = np.array(firstImg, np.int8)
                self.allFrames = np.zeros([len(self.filmedFramesLocs), firstImg.shape[0], firstImg.shape[1], firstImg.shape[2]], np.int8)
                self.allFrames[0, :, :, :] = firstImg
                for i, imageLoc in enumerate(self.filmedFramesLocs[1:]) :
                    if self.isDistorted :
                        img = Image.fromarray(cv2.remap(np.array(Image.open(imageLoc)), map1, map2, cv2.INTER_LINEAR).astype(numpy.uint8))
                    else :
                        img = Image.open(imageLoc)
                    img.thumbnail((firstImg.shape[1], firstImg.shape[0]), Image.ANTIALIAS)
                    self.allFrames[i, :, :, :] = np.array(img, np.int8)
                np.save(self.downsampledLoc, self.allFrames)
#             [self.maxV, self.maxU], self.aspectRatio = self.setFrame(self.allFrames[self.currentFrame, :, :, :])
            self.aspectRatio = float(self.allFrames[self.currentFrame, :, :, :].shape[1])/float(self.allFrames[self.currentFrame, :, :, :].shape[0])
            self.setGeometryAndBuffers()

            self.playTimer = QtCore.QTimer()
            self.playTimer.setInterval(1000/VIDEO_PLAYBACK_FPS)
            self.playTimer.timeout.connect(self.requestRender)
    #             self.playTimer.start()
            
            self.cameraFrustum = GLCameraFrustum(self.modelMat, self.allFrames[self.currentFrame, :, :, :], frustumScale)
        else :
            self.initFailed = True
            
        self.filmedObjects = []
        for loc in np.sort(glob.glob(filmedSceneLoc+"semantic_sequence-*.npy"))[1:2] :
            self.filmedObjects.append(GLFilmedObject(loc, self.cameraIntrinsics, np.linalg.inv(self.modelMat), self.isDistorted, self.distortionCoeff, originalIntrinsics))
            
            
        self.modifiedScene = GLModifiedScene(self.medianImage, self.filmedObjects[0], self.cameraExtrinsics, self.cameraIntrinsics)
        
        
    ## not sure this actually cleans up properly
    def __del__(self) :
        del self.allFrames, self.filmedFramesLocs, self.filmedObjects, self.cameraFrustum, self.modifiedScene
        
    def projectImageGridPoints(self, img) :
        frameSize = np.array([img.shape[1], img.shape[0]])
        gridDownsample = 1
        self.projectedImageGridPoints = np.indices(frameSize/gridDownsample).reshape([2, np.prod(frameSize/gridDownsample)]).T*gridDownsample
        self.projectedImageGridColors = img[self.projectedImageGridPoints[:, 1], self.projectedImageGridPoints[:, 0], :].astype(np.float32)/np.float32(255.0)
        if True :
#             cameraExtrinsics = np.array(self.modelMat.inverted()[0].data()).reshape([4, 4]).T
            print("README", self.cameraExtrinsics)
            inverseT = np.linalg.inv(np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]]))
            self.projectedImageGridPoints = np.dot(inverseT, np.concatenate([self.projectedImageGridPoints, np.ones([len(self.projectedImageGridPoints), 1], np.float32)], axis=1).T)
            self.projectedImageGridPoints /= self.projectedImageGridPoints[-1, :]
            self.projectedImageGridPoints[-1, :] = 0
            self.projectedImageGridPoints = self.projectedImageGridPoints.T.astype(np.float32)
            
        else :
            ### HACK : USE HOMOGRAPHY INSTEAD OF CAMERA MATRICES ###
            homography = np.array([[11.6261525276, 185.257281938, 818.145590521],
                                   [-24.7005245641, 14.5276400234, 272.499203107],
                                   [-0.197073111956, 0.178268418299, 1.0]])
            self.projectedImageGridPoints = np.dot(np.linalg.inv(homography), np.concatenate([self.projectedImageGridPoints.astype(np.float32), np.ones([len(self.projectedImageGridPoints), 1], np.float32)], axis=1).T)
            self.projectedImageGridPoints /= self.projectedImageGridPoints[-1, :]
            self.projectedImageGridPoints[-1, :] = 0
            self.projectedImageGridPoints = self.projectedImageGridPoints.T.astype(np.float32)
        print("RANGE OF POINTS", np.min(self.projectedImageGridPoints, axis=0), np.max(self.projectedImageGridPoints, axis=0))
    
    def setFrustumScaleDelta(self, scaleDelta) :
        if scaleDelta < 0.0 :
            self.cameraFrustum.setScale(np.max([0.01, scaleDelta+self.cameraFrustum.scale]))
        else :
            self.cameraFrustum.setScale(np.min([50.0, scaleDelta+self.cameraFrustum.scale]))
            
    def toggleShowFrustumBillboard(self) :
        self.cameraFrustum.toggleShowFrustumBillboard()
        
    def setPointSize(self, pointSize) :
        self.pointSize = np.float32(pointSize)
#         self.setGeometryAndBuffers()
        
    def setGeometryAndBuffers(self) :
        if not self.initFailed :
            ## PROJECTED IMAGE POINTS ##
            self.gridIndices = np.arange(len(self.projectedImageGridPoints)).astype(np.int32)

            self.gridIndexBuffer = glvbo.VBO(self.gridIndices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
            self.gridVerticesBuffer = glvbo.VBO(self.projectedImageGridPoints, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
            self.gridColorBuffer = glvbo.VBO(self.projectedImageGridColors, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        
    def requestRender(self) :
        if not self.initFailed :
            self.currentFrame = np.mod(self.currentFrame+self.playbackFrameSkip, len(self.filmedFramesLocs))
#             self.imagePlaneBillboard.setTexture(self.allFrames[self.currentFrame])
            self.cameraFrustum.setImage(self.allFrames[self.currentFrame])
        
    def setShaders(self) :
        self.colorNoShadeShadersProgram = compileShaders(VS_COLOR_NO_SHADE, FS_COLOR_NO_SHADE)
        if np.any(self.colorNoShadeShadersProgram == None) :
            self.initDone = False
            return
        self.cameraFrustum.setShaders()
        for filmedObject in self.filmedObjects :
            filmedObject.setShaders()
            
        self.modifiedScene.setShaders()
        
        self.initDone = True
    
    def isPlayerLookingAtCamera(self, projectionMat, viewMat) :
        ## THIS WORKS BUT IT CAN START PLAYING EVEN WHEN NOT LOOKING AT THE CAMERA SINCE THE ONLY THING THAT IS IMPORTANT IS THAT THE NORMALS ARE SOMEWHAT PARALLEL AND THEY ARE CLOSE BY
        camPos, camNorm = getWorldSpacePosAndNorm(self.modelMat)
        viewPos, viewNorm = getWorldSpacePosAndNorm(np.linalg.pinv(viewMat), np.array([[0.0], [0.0], [-1.0], [1.0]]))
#         print("README", np.linalg.norm(camPos-viewPos), np.arccos(np.dot(camNorm, viewNorm))*180.0/np.pi, camNorm, viewNorm, np.dot(camNorm, viewNorm), np.arccos(np.dot(camNorm, viewNorm)), angle)
        
#         return np.linalg.norm(camPos-viewPos) < 3.0 and np.pi-np.arccos(np.dot(camNorm, viewNorm)) < 25.0*np.pi/180.0 ## I think is wrong because I was using the wrong transforamtion which assumed -1 zDir
        return np.linalg.norm(camPos-viewPos) < 3.0 and np.arccos(np.clip(np.dot(camNorm, viewNorm), -1.0, 1.0)) < 25.0*np.pi/180.0
    
    def doRequestPlayVideo(self, doRequest) :
        if doRequest :
            if not self.playTimer.isActive() :
                print("START PLAYING"); sys.stdout.flush()
                self.playTimer.start()
        else :
            if self.playTimer.isActive() :
                print("STOP PLAYING"); sys.stdout.flush()
                self.playTimer.stop()
        
    def drawProjectedImageGridPoints(self, projectionMat, viewMat) :
        if self.pointSize > 0.01 :
            gl.glUseProgram(self.colorNoShadeShadersProgram)
            gl.glPointSize(self.pointSize)

            ## send mvp
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.colorNoShadeShadersProgram, "m_pvm"), 1, gl.GL_FALSE, np.dot(projectionMat, viewMat).T)
            ## send camera distance
            gl.glUniform1f(gl.glGetUniformLocation(self.colorNoShadeShadersProgram, "camera_dist"), np.float32(1.0))
    #             print(cameraDist)

            ################ RENDER BODY ################

            ## bind the index buffer
            self.gridIndexBuffer.bind()

            ## bind the VBO with vertex data
            self.gridVerticesBuffer.bind()
            gl.glEnableVertexAttribArray(0)
            # tell OpenGL that the VBO contains an array of vertices
            # these vertices contain 3 single precision coordinates
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## bind the VBO with color data
            self.gridColorBuffer.bind()
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## draw points from the VBO
            gl.glDrawElements(gl.GL_POINTS, len(self.gridIndices), gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)

            gl.glUseProgram(0)
        
    def draw(self, projectionMat, viewMat) :
        if self.initDone and not self.initFailed :
            self.doRequestPlayVideo(self.isPlayerLookingAtCamera(projectionMat, viewMat))

            self.drawProjectedImageGridPoints(projectionMat, viewMat)
            self.cameraFrustum.draw(projectionMat, viewMat)
                
            for filmedObject in self.filmedObjects :
                filmedObject.draw(projectionMat, viewMat)
                
            self.modifiedScene.draw(projectionMat, viewMat)

# <codecell>

class GLScene() :
    def __init__(self) :
        self.lightDirection = np.array([[1, 1, 1, 0]], np.float32).T#QtGui.QVector4D(1.0, 1.0, 1.0, 0.0)
        self.lightColor = np.array([1.0, 1.0, 1.0], np.float32)
        self.lightPower = np.float32(1.0)
        
        self.projectionMat = np.eye(4, dtype=np.float32)
        
        ## set view matrix using the qt lookat function
        self.viewMat = QtGui.QMatrix4x4()
        ## (cameraPos, cameraPos + direction, upVec) are in gl coords
        cameraPos = QtGui.QVector3D(0.0, 1.0, 6.0)
        self.viewMat.lookAt(cameraPos, cameraPos+QtGui.QVector3D(0, 0, -1), QtGui.QVector3D(0, 1, 0))
        ## rotate gl coords to match my world coords
        self.viewMat.rotate(-90, 1, 0, 0)
        
        self.viewMat = np.array(self.viewMat.data(), np.float32).reshape([4, 4]).T
        self.width = 1280
        self.height = 720
        
        
        self.shaders_program = None
        
        self.doShowEdges = True
        
        self.axesWidget = AxesWidget()
        self.meshes = []
        self.filmedScenes = []
        
        self.currentObjectViewFrame = -1
        self.doPlaybackObjectViews = False
    
    ## not sure this actually cleans up properly
    def __del__(self) :
        del self.meshes, self.filmedScenes, self.axesWidget, self.lightDirection, self.lightColor, self.lightPower
        
    def setShaderProgram(self, shaders_program) :
        self.shaders_program = shaders_program
        for i in xrange(len(self.meshes)) :
            self.meshes[i].shaders_program = self.shaders_program
        
        if not self.axesWidget.initDone :
            self.axesWidget.setShaders()
        for i in xrange(len(self.filmedScenes)) :
            if not self.filmedScenes[i].initDone :
                self.filmedScenes[i].setShaders()
        
    def setFrustumScaleDelta(self, scaleDelta) :
        for filmedScene in self.filmedScenes :
            filmedScene.setFrustumScaleDelta(scaleDelta)
            
    def toggleShowFrustumBillboard(self) :
        for filmedScene in self.filmedScenes :
            filmedScene.toggleShowFrustumBillboard()
        
    def setPointSizeDelta(self, sizeDelta) :
        for filmedScene in self.filmedScenes :
            if sizeDelta < 0.0 :
                filmedScene.setPointSize(np.max([0.01, sizeDelta+filmedScene.pointSize]))
            else :
                filmedScene.setPointSize(np.min([30.0, sizeDelta+filmedScene.pointSize]))
        
    def setCameraProjectionMat(self, cameraFOV, width, height, near=0.1, far=1000.0) :
        self.width = width
        self.height = height
        self.projectionMat = QtGui.QMatrix4x4()
        self.projectionMat.perspective(cameraFOV, width/float(height), near, far)
        self.projectionMat = np.array(self.projectionMat.data(), np.float32).reshape([4, 4]).T
        
    def translateCamera(self, translation) :
        viewPos, viewDir = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), np.array([[0.0], [0.0], [-1.0], [1.0]]))
        viewPos, viewUp = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), np.array([[0.0], [1.0], [0.0], [1.0]]))
        viewPos, viewRight = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), np.array([[1.0], [0.0], [0.0], [1.0]]))
        
        t = viewDir*translation[0] + viewRight*translation[1] + viewUp*translation[2]
        tMat = np.array([[1, 0, 0, t[0]],
                         [0, 1, 0, t[1]],
                         [0, 0, 1, t[2]],
                         [0, 0, 0, 1]], np.float32)
        self.viewMat = np.dot(self.viewMat, np.linalg.pinv(tMat))

    def rotateCamera(self, quaternion, centerPoint) :        
        self.viewMat = np.linalg.inv(rotateAboutPoint(np.linalg.inv(self.viewMat), quaternion, centerPoint))
        
    ## this probably makes more sense in some other class but it's easier here for now
    def goToCamera(self) :
        if len(self.filmedScenes) > 0 :
            print("PREVIOUS:\n", self.viewMat)
            print("EXTRINSICS:\n", self.filmedScenes[0].cameraExtrinsics)
            
            self.viewMat = np.copy(self.filmedScenes[0].cameraExtrinsics)
            ## flip z and y axis because of opencv vs opengl coord systems
            self.viewMat[2, :] *= -1
            self.viewMat[1, :] *= -1
            print("VIEW:\n", self.viewMat)
            
            cameraIntrinsics = np.copy(self.filmedScenes[0].cameraIntrinsics)
            ## changing signs for the same reason as above for the viewMat
            cameraIntrinsics[:, 2] *= -1
            cameraIntrinsics[:, 1] *= -1
            near = 0.1
            far = 100.0
            projectionMat = np.zeros([4, 4])
            projectionMat[:2, :-1] = cameraIntrinsics[:2, :]
            projectionMat[-1, :-1] = cameraIntrinsics[-1, :]
            projectionMat[2, 2] = near + far
            projectionMat[2, 3] = near*far
            
            ## HACK hardcoded for havana
            left = 0.0
            right = float(self.filmedScenes[0].medianImage.shape[1])
            bottom = float(self.filmedScenes[0].medianImage.shape[0])
            top = 0.0
            print("FSDFSAD", left, right, bottom, top)
            
            projectionMat = np.dot(np.array([[2/(right-left), 0, 0, -(right+left)/(right-left)],
                                             [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
                                             [0, 0, -2/(far-near), -(far+near)/(far-near)],
                                             [0, 0, 0, 1]]), np.copy(projectionMat))
            
            print("PROJ:\n", self.projectionMat)
            self.projectionMat = np.copy(projectionMat)
            
            
            print("PROJ_NEW:\n", self.projectionMat)
            
            sys.stdout.flush()
            ## returns new fov
            return np.arctan2(1.0, projectionMat[1, 1])*2.0*180.0/np.pi
            
    ## this probably makes more sense in some other class but it's easier here for now
    def playbackObjectViews(self) :
        self.currentObjectViewFrame = np.mod(self.currentObjectViewFrame+1, len(self.filmedScenes[0].filmedObjects[0].cameraToObjectDirectionsObjSpace))
#         self.currentObjectViewFrame = 0
        
        ## find distance from obj to current camera
        viewPos = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), posOnly=True)
        objPos = getWorldSpacePosAndNorm(self.filmedScenes[0].filmedObjects[0].modelMat, posOnly=True)
        viewToObjDir = objPos-viewPos
        distanceToObj = np.linalg.norm(viewToObjDir)
        viewToObjDir /= distanceToObj

        ## desired direction in object space
        desiredViewDir = self.filmedScenes[0].filmedObjects[0].cameraToObjectDirectionsObjSpace[self.currentObjectViewFrame, :]
        ## direction into world space
        ## from object to world space using modelMat
        desiredViewDirPos = np.dot(self.filmedScenes[0].filmedObjects[0].modelMat, np.concatenate([np.zeros(3) - desiredViewDir, [1]]).reshape([4, 1])).flatten()
        desiredViewDirPos = desiredViewDirPos[:-1]/desiredViewDirPos[-1]
        desiredViewDir = objPos-desiredViewDirPos
        desiredViewDir /= np.linalg.norm(desiredViewDir)

        camMat = np.eye(4, dtype=np.float32)
        camMat[:-1, -1] = objPos-desiredViewDir*distanceToObj

        ## rotate camera to give it the desired direction
        viewPos, viewDir = getWorldSpacePosAndNorm(camMat)
        axis = np.cross(desiredViewDir, viewDir)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(desiredViewDir, viewDir), -1.0, 1.0))
        camMat = rotateAboutPoint(camMat, angleAxisToQuaternion(angle, axis), viewPos)

        ## now rotate along viewDir to make sure that the up vector is pointing up (i.e. camera is parallel to ground plane)
        desiredPlaneNorm = np.array([0.0, 0.0, 1.0])
        _, viewDir = getWorldSpacePosAndNorm(camMat)
        _, viewUp = getWorldSpacePosAndNorm(camMat, np.array([[0.0], [-1.0], [0.0], [1.0]]))
        ## project desiredPlaneNorm (i.e. norm of ground plane) onto camera image plane (described by the view direction or normal)
        projNorm = desiredPlaneNorm - np.dot(desiredPlaneNorm, viewDir)*viewDir
        projNorm /= np.linalg.norm(projNorm)

        adjustAngle = np.arccos(np.clip(np.dot(projNorm, viewUp), -1, 1))
        adjustAxis = np.cross(projNorm, viewUp)
        adjustAxis /= np.linalg.norm(adjustAxis)
        camMat = rotateAboutPoint(camMat, angleAxisToQuaternion(adjustAngle, adjustAxis), viewPos)

        self.viewMat = np.linalg.inv(camMat)
        self.viewMat[1:3] *= -1
        
#         print(self.currentObjectViewFrame)
        
    def draw(self) :
        if self.doPlaybackObjectViews :
            self.playbackObjectViews()
        if np.all(self.shaders_program != None) :
            gl.glUseProgram(self.shaders_program)

            ## scene specific parameters
            gl.glUniform3fv(gl.glGetUniformLocation(self.shaders_program, "l_color"), 1, self.lightColor)
            gl.glUniform1f(gl.glGetUniformLocation(self.shaders_program, "l_power"), self.lightPower)

            if gl.glGetUniformLocation(self.shaders_program, "l_dir") != -1 :
                lightDirection = np.dot(self.viewMat, self.lightDirection)
                lightDirection /= np.float32(np.linalg.norm(lightDirection).flatten())
                gl.glUniform3fv(gl.glGetUniformLocation(self.shaders_program, "l_dir"), 1, lightDirection.flatten()[:3])

            if gl.glGetUniformLocation(self.shaders_program, "l_pos_world") != -1 :
                ## send light position (i.e. camera position)
                lightPos = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), posOnly=True)
                gl.glUniform3fv(gl.glGetUniformLocation(self.shaders_program, "l_pos_world"), 1, lightPos.astype(np.float32))

            if gl.glGetUniformLocation(self.shaders_program, "show_edges") != -1 :
                gl.glUniform1i(gl.glGetUniformLocation(self.shaders_program, "show_edges"), self.doShowEdges)
                
            ## send viewlightDirection
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shaders_program, "m_v"), 1, gl.GL_FALSE, self.viewMat.T)
            gl.glUseProgram(0)
            
            ## draw meshes
            for i in xrange(len(self.meshes)) :
                self.meshes[i].draw(self.projectionMat, self.viewMat)
                
            ## draw filmed scenes
            for i in xrange(len(self.filmedScenes)) :
                self.filmedScenes[i].draw(self.projectionMat, self.viewMat)
                
#             cameraPos = np.array([self.cameraPos.x(), self.cameraPos.y(), self.cameraPos.z()], np.float32)
#             cameraDist = np.float32(self.cameraPos.length())
            cameraPos = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), posOnly=True)
            cameraDist = np.float32(np.linalg.norm(cameraPos))
            self.axesWidget.draw(cameraDist, np.dot(self.projectionMat, self.viewMat))
        
    def loadSceneFromFile(self, sceneLoc) :
#         self.addMeshesFromFile(sceneLoc)
        self.filmedScenes.append(GLFilmedScene("../data/havana/"))
        
    def addMeshesFromFile(self, fileLoc) :
        meshAsset = pyassimp.load(fileLoc)
        for mesh in meshAsset.meshes :
            newMesh = GLMesh(mesh, self.shaders_program)
            if not newMesh.isInvalidMesh :
                self.meshes.append(newMesh)
        
                print("Loaded mesh:", len(mesh.vertices), "vertices,", len(mesh.faces), "faces"); sys.stdout.flush()
            else :
                del newMesh
        
        pyassimp.release(meshAsset)

# <codecell>

class GLWidget(QGLWidget):
    # default window size
    width, height = 600.0, 600.0
    
    def __init__(self, fmt, parent=None):
        super(GLWidget, self).__init__(fmt, parent=parent)
        
        self.scene = GLScene()
        
        self.initDone = False
        self.sceneChanged = False
        self.shadersChanged = False
        self.doShowEdges = True
        
        self.cameraHorizontalAngle = np.pi
        self.cameraVerticalAngle = 0#-np.pi/6
        self.cameraFOV = 45.0
        self.cameraSpeed = 10.0
        
        self.indexBuffers = []
        self.verticesBuffers = []
        self.uvsBuffers = []
        self.barycentricsBuffers = []
        self.normalsBuffers = []
        
        self.setScene("../data/suzanne.obj")
        self.setShaders(VS_HEAD_LIGHT, FS_HEAD_LIGHT)
#         self.setViewAndProjectionMats()
        self.scene.setCameraProjectionMat(self.cameraFOV, self.width, self.height)
        
    def setCameraFOV(self, cameraFOV) :
        self.cameraFOV = cameraFOV
        self.scene.setCameraProjectionMat(self.cameraFOV, self.width, self.height)
        
    def setFrustumScaleDelta(self, scaleDelta) :
        self.scene.setFrustumScaleDelta(scaleDelta)
        
    def setPointSizeDelta(self, sizeDelta) :
        self.scene.setPointSizeDelta(sizeDelta)
    
    def setScene(self, sceneLoc) :
        self.scene.loadSceneFromFile(sceneLoc)

        self.sceneChanged = True
        self.glInit()
    
    def setShaders(self, vs, fs) :
        self.vs = vs
        self.fs = fs
        self.shadersChanged = True
        self.glInit()
        
    def setShowEdges(self, doShowEdges) :
        self.scene.doShowEdges = doShowEdges
    
    def cleanup(self) :
        del self.scene

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        
        self.initDone = True
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # background color
        gl.glClearColor(0.2, 0.2, 0.2, 0.2)

        if self.shadersChanged :
            self.shaders_program = compileShaders(self.vs, self.fs)
            self.scene.setShaderProgram(self.shaders_program)
            if np.any(self.shaders_program == None) :
                self.initDone = False
                

    def paintGL(self):
        """Paint the scene."""
        
        if self.initDone :
            # clear the buffer
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.scene.draw()
            
            if False and self.scene.doPlaybackObjectViews :
                imgBuffer = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                global glImage
                glImage = Image.frombytes(mode="RGB", size=(self.width, self.height), data=imgBuffer)
                glImage = glImage.transpose(Image.FLIP_TOP_BOTTOM)
                glImage.save("/home/ilisescu/PhD/animation_images/adjusted-rotation-adjust-scale-adjust/frame-{0:05}.png".format(self.scene.currentObjectViewFrame))

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        self.width, self.height = width, height
        self.scene.setCameraProjectionMat(self.cameraFOV, self.width, self.height)
        
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

# define a Qt window with an OpenGL widget inside it
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.doShowEdges = True
        self.maxDeltaTime = 0.0
        
        self.changingOrientation = False
        self.prevPoint = None
        self.mouseSpeed = 0.001
        self.mouseDiffPos = QtCore.QPointF(0, 0)
        self.doMoveForwards = 0.0
        self.doMoveSideways = 0.0
        self.doMoveUpwards = 0.0
        self.doPivotHorizontally = 0.0
        self.doPivotVertically = 0.0
        self.doRoll = 0.0
        
        self.doChangeFOV = False
        self.doChangeFrustumScale = False
        self.doChangePointSize = False
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.requestRender)
        self.lastRenderTime = time.time()
        self.playTimer.start()
        
        self.setWindowTitle("3D Looping")
        self.resize(1280, 720)
        
        self.setFocus()
        
    def requestRender(self) :
        currentTime = time.time()
        deltaTime = currentTime - self.lastRenderTime

        if self.doMoveForwards != 0.0 or self.doMoveSideways != 0.0 or self.doMoveUpwards != 0.0 :
            self.glWidget.scene.translateCamera(np.array([self.doMoveForwards*deltaTime*self.glWidget.cameraSpeed,
                                                          self.doMoveSideways*deltaTime*self.glWidget.cameraSpeed,
                                                          self.doMoveUpwards*deltaTime*self.glWidget.cameraSpeed]))
            
        if self.doRoll != 0.0 :
            angle = (self.doRoll*deltaTime*self.glWidget.cameraSpeed*5)*np.pi/180.0
            cameraPos, axis = getWorldSpacePosAndNorm(np.linalg.pinv(self.glWidget.scene.viewMat))
            self.glWidget.scene.rotateCamera(angleAxisToQuaternion(angle, axis), cameraPos)
            
        if self.doPivotHorizontally != 0.0 :
            angle = (self.doPivotHorizontally*deltaTime*self.glWidget.cameraSpeed*5)*np.pi/180.0
            axis = np.array([0, 0, 1], np.float32)
            self.glWidget.scene.rotateCamera(angleAxisToQuaternion(angle, axis), np.zeros(3))
            
        if self.doPivotVertically != 0.0 :
            angle = (self.doPivotVertically*deltaTime*self.glWidget.cameraSpeed*5)*np.pi/180.0
            _, axis = getWorldSpacePosAndNorm(np.linalg.pinv(self.glWidget.scene.viewMat), np.array([[1.0], [0.0], [0.0], [1.0]]))
            self.glWidget.scene.rotateCamera(angleAxisToQuaternion(angle, axis), np.zeros(3))
        
        self.glWidget.updateGL()
        
        cameraPos = getWorldSpacePosAndNorm(np.linalg.pinv(self.glWidget.scene.viewMat), posOnly=True)
        
        self.infoLabel.setText("Rendering at {0} FPS, FOV: {1}; {2}; render time[ms]: {3}, using frame {4}".format(int(1.0/(deltaTime)), self.glWidget.cameraFOV,
                                                                                                                   cameraPos, (time.time()-currentTime)*1000.0,
                                                                                                                   self.glWidget.scene.filmedScenes[0].filmedObjects[0].frameToUseIdx)+
                               "\nMove: Arrows/WASD --- Rise: R/F --- Roll: Q/E --- Pivot H: Z/X --- Pivot V: PageUp/Down --- FOV: V --- Frustum: U --- Point: P --- GoToCam: C --- "+
                               "Show Frustum Billboard: Space --- Speed: M Wheel --- Playback Obj Views: K")
        self.lastRenderTime = np.copy(currentTime)
        if self.maxDeltaTime < deltaTime :
            self.maxDeltaTime = deltaTime
            print("MAX DELTA", self.maxDeltaTime); sys.stdout.flush()
#         self.playTimer.stop()
        
    def mousePressed(self, event):
        if event.button() == QtCore.Qt.LeftButton :
            self.changingOrientation = True
            self.prevPoint = event.posF()
                
    def mouseMoved(self, event):
        if self.changingOrientation and self.prevPoint != None :
            try :
                prevPos = np.array([self.prevPoint.x(), self.prevPoint.y(), 0.0])
                currentPos = np.array([event.posF().x(), event.posF().y(), 0.0])
                angle = (np.linalg.norm(prevPos-currentPos)*self.mouseSpeed*100)*np.pi/180.0
                if angle > 0.0 :
                    axis = prevPos-currentPos
                    axis /= np.linalg.norm(axis)
                    axis = -np.array([axis[1], axis[0], axis[2]])
                    cameraPos, axis = getWorldSpacePosAndNorm(np.linalg.pinv(self.glWidget.scene.viewMat), np.array([[axis[0]], [axis[1]], [axis[2]], [1.0]]))
                    self.glWidget.scene.rotateCamera(angleAxisToQuaternion(angle, axis), cameraPos)

                self.prevPoint = event.posF()
            except :
                print("ROTATION", axis, np.linalg.norm(prevPos-currentPos)*self.mouseSpeed*100)
                print(self.glWidget.scene.viewMat)
                print("ERROR:", sys.exc_info()[0])
                raise
            
    def mouseReleased(self, event):
        if event.button() == QtCore.Qt.LeftButton :
            self.changingOrientation = False
            self.prevPoint = None
            self.mouseDiffPos = QtCore.QPointF(0, 0)
        
    def wheelEvent(self, e) :
        if self.doChangeFrustumScale :
            self.glWidget.setFrustumScaleDelta(0.0001*e.delta())
        elif self.doChangeFOV :
            if e.delta() < 0.0 :
                self.glWidget.setCameraFOV(np.max([10.0, 0.005*e.delta()+self.glWidget.cameraFOV]))
            else :
                self.glWidget.setCameraFOV(np.min([170.0, 0.005*e.delta()+self.glWidget.cameraFOV]))
        elif self.doChangePointSize :
            self.glWidget.setPointSizeDelta(0.005*e.delta())
        else :
            if e.delta() < 0.0 :
                self.glWidget.cameraSpeed = np.max([0.1, 0.001*e.delta()+self.glWidget.cameraSpeed])
            else :
                self.glWidget.cameraSpeed = np.min([30.0, 0.001*e.delta()+self.glWidget.cameraSpeed])
        
    def eventFilter(self, obj, event) :
        if obj == self.glWidget and event.type() == QtCore.QEvent.Type.MouseMove :
            self.mouseMoved(event)
            return True
        elif obj == self.glWidget and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.mousePressed(event)
            return True
        elif obj == self.glWidget and event.type() == QtCore.QEvent.Type.MouseButtonRelease :
            self.mouseReleased(event)
            return True
        return QtGui.QWidget.eventFilter(self, obj, event)
    
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_C :
            newFOV = self.glWidget.scene.goToCamera()
            if newFOV != None :
                self.glWidget.cameraFOV = np.copy(newFOV)
        if e.key() == QtCore.Qt.Key_Space :
            self.glWidget.scene.toggleShowFrustumBillboard()
        if e.key() == QtCore.Qt.Key_K :
            self.glWidget.scene.doPlaybackObjectViews = not self.glWidget.scene.doPlaybackObjectViews
        
        ## Move
        if e.key() == QtCore.Qt.Key_W or e.key() == QtCore.Qt.Key_Up :
            self.doMoveForwards += 1.0
        if e.key() == QtCore.Qt.Key_S or e.key() == QtCore.Qt.Key_Down :
            self.doMoveForwards -= 1.0
        if e.key() == QtCore.Qt.Key_D or e.key() == QtCore.Qt.Key_Right :
            self.doMoveSideways += 1.0
        if e.key() == QtCore.Qt.Key_A or e.key() == QtCore.Qt.Key_Left :
            self.doMoveSideways -= 1.0            
        ## Rise
        if e.key() == QtCore.Qt.Key_R :
            self.doMoveUpwards += 1.0
        if e.key() == QtCore.Qt.Key_F :
            self.doMoveUpwards -= 1.0
        ## Roll
        if e.key() == QtCore.Qt.Key_Q :
            self.doRoll -= 1.0
        if e.key() == QtCore.Qt.Key_E :
            self.doRoll += 1.0
        ## Pivot
        if e.key() == QtCore.Qt.Key_Z :
            self.doPivotHorizontally += 1.0
        if e.key() == QtCore.Qt.Key_X :
            self.doPivotHorizontally -= 1.0
        if e.key() == QtCore.Qt.Key_PageDown :
            self.doPivotVertically -= 1.0
        if e.key() == QtCore.Qt.Key_PageUp :
            self.doPivotVertically += 1.0
            
        if e.key() == QtCore.Qt.Key_V :
            self.doChangeFOV = True
        if e.key() == QtCore.Qt.Key_U :
            self.doChangeFrustumScale = True
        if e.key() == QtCore.Qt.Key_P :
            self.doChangePointSize = True
    
    def keyReleaseEvent(self, e) :
        ## Move
        if e.key() == QtCore.Qt.Key_W or e.key() == QtCore.Qt.Key_Up :
            self.doMoveForwards -= 1.0
        if e.key() == QtCore.Qt.Key_S or e.key() == QtCore.Qt.Key_Down :
            self.doMoveForwards += 1.0
        if e.key() == QtCore.Qt.Key_D or e.key() == QtCore.Qt.Key_Right :
            self.doMoveSideways -= 1.0
        if e.key() == QtCore.Qt.Key_A or e.key() == QtCore.Qt.Key_Left :
            self.doMoveSideways += 1.0
        ## Rise
        if e.key() == QtCore.Qt.Key_R :
            self.doMoveUpwards -= 1.0
        if e.key() == QtCore.Qt.Key_F :
            self.doMoveUpwards += 1.0
        ## Roll
        if e.key() == QtCore.Qt.Key_Q :
            self.doRoll += 1.0
        if e.key() == QtCore.Qt.Key_E :
            self.doRoll -= 1.0
        ## Pivot
        if e.key() == QtCore.Qt.Key_Z :
            self.doPivotHorizontally -= 1.0
        if e.key() == QtCore.Qt.Key_X :
            self.doPivotHorizontally += 1.0
        if e.key() == QtCore.Qt.Key_PageDown :
            self.doPivotVertically += 1.0
        if e.key() == QtCore.Qt.Key_PageUp :
            self.doPivotVertically -= 1.0
            
        if e.key() == QtCore.Qt.Key_V :
            self.doChangeFOV = False
        if e.key() == QtCore.Qt.Key_U :
            self.doChangeFrustumScale = False
        if e.key() == QtCore.Qt.Key_P :
            self.doChangePointSize = False
            
    def closeEvent(self, event) :
        self.playTimer.stop()
        self.glWidget.cleanup()
    
    def changeScene(self) :
        sceneLoc = QtGui.QFileDialog.getOpenFileName(self, "Load Scene", os.path.expanduser("~")+"/PhD/data/", "OBJ files (*.obj);;PLY files (*.ply)")[0]
        if sceneLoc != "" :
            self.glWidget.setScene(sceneLoc)
        
    def useHeadLight(self) :
        self.glWidget.setShaders(VS_HEAD_LIGHT, FS_HEAD_LIGHT)
    
    def useDirLight(self) :
        self.glWidget.setShaders(VS_DIR_LIGHT, FS_DIR_LIGHT)
        
    def toggleEdges(self) :
        self.doShowEdges = not self.doShowEdges
        self.glWidget.setShowEdges(self.doShowEdges)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        # initialize the GL widget
        fmt = QGLFormat()
        fmt.setSampleBuffers(True)
        fmt.setSamples(8)
        fmt.setAlpha(True)
        self.glWidget = GLWidget(fmt=fmt, parent=self)
        self.glWidget.setMinimumSize(self.glWidget.width, self.glWidget.height)
        self.glWidget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.glWidget.installEventFilter(self)
         
        self.infoLabel = QtGui.QLabel("Info")
#         self.infoLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignHCenter)
        
        self.changeSceneButton = QtGui.QPushButton("Change Scene")
        self.changeSceneButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.useHeadLightButton = QtGui.QPushButton("Use Head Light")
        self.useHeadLightButton.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.useDirLightButton = QtGui.QPushButton("Use Directional Light")
        self.useDirLightButton.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.toggleEdgesButton = QtGui.QPushButton("Toggle Edges")
        self.toggleEdgesButton.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        
        ## SIGNALS ##
        
        self.changeSceneButton.clicked.connect(self.changeScene)
        self.useHeadLightButton.clicked.connect(self.useHeadLight)
        self.useDirLightButton.clicked.connect(self.useDirLight)
        self.toggleEdgesButton.clicked.connect(self.toggleEdges)
        
        ## LAYOUTS ##
        
        controlsLayout = QtGui.QGridLayout()
        idx = 0
        controlsLayout.addWidget(self.changeSceneButton, idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.setColumnStretch(0, 10)
        controlsLayout.addWidget(self.useHeadLightButton, idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.useDirLightButton, idx, 2, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.toggleEdgesButton, idx, 3, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.glWidget)
        mainLayout.addWidget(self.infoLabel)
        mainLayout.addLayout(controlsLayout)
        
        self.setLayout(mainLayout)

# <codecell>

class GLModifiedScene() :
    def __init__(self, bgImage, filmedObject, cameraExtrinsics, cameraIntrinsics) :
        self.initDone = False
        
        self.bgImage = bgImage
        self.viewport = np.array([0, 0, self.bgImage.shape[1], self.bgImage.shape[0]], np.float32)
        self.cameraExtrinsics = np.copy(cameraExtrinsics)
        self.cameraIntrinsics = np.copy(cameraIntrinsics)
        
        self.screenHeightRatio = 0.4
        ## screen height in clip space is 2 so need to multiply this ratio by 2
        self.renderBillboard = GLBillboard(self.bgImage, self.screenHeightRatio*2.0, np.eye(4, dtype=np.float32), False, None, False)
        
        self.trajectory = GLTrajectory(filmedObject.trajectoryPoints, cameraIntrinsics, cameraExtrinsics, filmedObject.objectData[DICT_REPRESENTATIVE_COLOR], False)
        
        ## eventually I can use these to control the car
        self.moveDirection = np.array([[-1.0, 0.0]], dtype=np.float32)
        self.position = np.array([[932.0, 538.0]], dtype=np.float32)
        
        self.moveDirectionIndicatorCameraSpace = GLTrajectory(np.concatenate([self.position, self.position+self.moveDirection*100.0]), cameraIntrinsics, cameraExtrinsics, doDrawProjectedPoints=False, doSmoothing=False)
        self.moveDirectionIndicatorWorldSpace = GLTrajectory(np.concatenate([self.position, self.position+self.moveDirection*100.0]), cameraIntrinsics, cameraExtrinsics, doSmoothing=False)
        
    def __del__(self) :
        del self.bgImage, self.renderBillboard
        
    def setShaders(self) :
        self.renderBillboard.setShaders()
        self.trajectory.setShaders()
        self.moveDirectionIndicatorCameraSpace.setShaders()
        self.moveDirectionIndicatorWorldSpace.setShaders()
        self.initDone = True
        
    def draw(self, projectionMat, viewMat) :
        if self.initDone :
            top, left, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)
            viewportAspectRatio = float(width)/float(height)
            
            isDepthTestOn = bool(gl.glGetBooleanv(gl.GL_DEPTH_TEST))
            gl.glDisable(gl.GL_DEPTH_TEST)
            
            ## change modelMat to scale x axis (y axis is fine as it's 1) so that aspect ratio is correct, by removing the viewport aspect ratio
            ## also move the correctly scaled billboard to sit in the top right corner
            tMat = np.array([[1.0/viewportAspectRatio, 0, 0, 1.0-self.screenHeightRatio*self.renderBillboard.aspectRatio/viewportAspectRatio],
                             [0, 1, 0, 1.0-self.screenHeightRatio],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], np.float32)
            self.renderBillboard.modelMat = tMat
            self.renderBillboard.draw(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
            
            ## align trajectory to billboard by setting modelMat of trajectory to pvm = m = tMat*scaleMat
            self.trajectory.draw(np.dot(tMat, np.array([[self.renderBillboard.scale, 0, 0, 0],
                                                        [0, -self.renderBillboard.scale, 0, 0],
                                                        [0, 0, self.renderBillboard.scale, 0],
                                                        [0, 0, 0, 1]], np.float32)))
            
            ## render the indicators for where the object should be (both camera and world space)
            self.moveDirectionIndicatorCameraSpace.draw(np.dot(tMat, np.array([[self.renderBillboard.scale, 0, 0, 0],
                                                                               [0, -self.renderBillboard.scale, 0, 0],
                                                                               [0, 0, self.renderBillboard.scale, 0],
                                                                               [0, 0, 0, 1]], np.float32)))
            
            self.moveDirectionIndicatorWorldSpace.draw(np.dot(projectionMat, viewMat))
            
            
            
    
            if isDepthTestOn :
                gl.glEnable(gl.GL_DEPTH_TEST)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

# [-0.90505216  0.42530058]
# print(np.array([-0.90343641, 0.42872211])/np.array([1.4828125, 1.45694444])/np.linalg.norm(np.array([-0.90343641, 0.42872211])/np.array([1.4828125, 1.45694444])))
# print(1280/720.0, 1898/1050.0)
# print(832*2, 468*2)
# print(1664/936.0, 1280/720.0, 1898/1050.0)
figure();
xlim([0, 1664])
ylim([936, 0])
plot(np.array([(newFrameSize/2)[0], (newFrameSize/2)[0]+jack[0]*50]), np.array([(newFrameSize/2)[1], (newFrameSize/2)[1]+jack[1]*50]))

figure();
xlim([0, 1898])
ylim([1050, 0])
plot(np.array([(np.array([1898, 1050.0])/2)[0], (np.array([1898, 1050.0])/2)[0]+jack[0]*50]), np.array([(np.array([1898, 1050.0])/2)[1], (np.array([1898, 1050.0])/2)[1]+jack[1]*50]))

# <codecell>

# print(tmpTrajectoryCameraSpace-trajectoryPointsCameraSpace)
figure(); plot(tmpTrajectoryCameraSpace[:, 0], tmpTrajectoryCameraSpace[:, 1])
plot(trajectoryPointsCameraSpace[:, 0], trajectoryPointsCameraSpace[:, 1])

# <codecell>

print(jack)

# <codecell>

## UNDISTORT EACH IMAGE WHERE A SPRITE IS PRESENT AND SAVE CORRECTED PATCHES ##

## load camera data
trajectorySmoothness = 5
data3D = np.load("tmp_trajectory_3D.npy").item()
print(data3D.keys())
cameraExtrinsics = data3D['extrinsics']
cameraIntrinsics = data3D['intrinsics']
distortionParameter = -0.19
distortionRatio = -0.19
distortionCoeff = np.array([distortionParameter, distortionParameter*distortionRatio, 0.0, 0.0, 0.0])
originalIntrinsics = np.array([[702.736053, 0.0, 640.0],
                               [0.0, 702.736053, 360.0],
                               [0.0, 0.0, 1.0]])

objectData = np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()
print("LOADED", objectData[DICT_SEQUENCE_NAME])
patches = np.load(objectData[DICT_PATCHES_LOCATION]).item()
sortedPatchKeys = np.sort(patches.keys())


## get trajectory points and directions in both camera and world space
frameSubset = [30, -50]
# frameSubset = [0, 0]
trajectoryPointsCameraSpace = np.array([objectData[DICT_BBOX_CENTERS][key] for key in sort(objectData[DICT_BBOX_CENTERS].keys())])[frameSubset[0]:frameSubset[1]-1, :]
## undistort points
trajectoryPointsCameraSpace = cv2.undistortPoints(trajectoryPointsCameraSpace.reshape((1, len(trajectoryPointsCameraSpace), 2)), originalIntrinsics, distortionCoeff, P=cameraIntrinsics)[0, :, :]

inverseT = np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]))
trajectoryPointsWorldSpace = np.dot(inverseT, np.concatenate([trajectoryPointsCameraSpace, np.ones([len(trajectoryPointsCameraSpace), 1], np.float32)], axis=1).T)
trajectoryPointsWorldSpace /= trajectoryPointsWorldSpace[-1, :]
trajectoryPointsWorldSpace[-1, :] = 0
trajectoryPointsWorldSpace = trajectoryPointsWorldSpace.T.astype(np.float32)
        
## smooth trajectory
trajectoryPointsWorldSpace = np.array([spimg.filters.gaussian_filter1d(trajectoryPointsWorldSpace[:, 0], trajectorySmoothness, axis=0),
                                       spimg.filters.gaussian_filter1d(trajectoryPointsWorldSpace[:, 1], trajectorySmoothness, axis=0), 
                                       spimg.filters.gaussian_filter1d(trajectoryPointsWorldSpace[:, 2], trajectorySmoothness, axis=0)]).T.astype(np.float32)

## reproject points into image space after smoothing
T = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])
trajectoryPointsCameraSpace = np.dot(T, np.concatenate([trajectoryPointsWorldSpace[:, :-1], np.ones([len(trajectoryPointsWorldSpace), 1])], axis=1).T)
trajectoryPointsCameraSpace = (trajectoryPointsCameraSpace[:-1, :]/trajectoryPointsCameraSpace[-1, :]).T


trajectoryDirectionsWorldSpace = np.array([trajectoryPointsWorldSpace[i, :]-trajectoryPointsWorldSpace[j, :] for i, j in zip(xrange(1, len(trajectoryPointsWorldSpace)),
                                                                                                                             xrange(0, len(trajectoryPointsWorldSpace)-1))])
trajectoryDirectionsWorldSpace /= np.linalg.norm(trajectoryDirectionsWorldSpace, axis=1).reshape([len(trajectoryDirectionsWorldSpace), 1])
## use direction of second to last point as the direction for the last point
trajectoryDirectionsWorldSpace = np.concatenate([trajectoryDirectionsWorldSpace, trajectoryDirectionsWorldSpace[-1, :].reshape([1, trajectoryDirectionsWorldSpace.shape[1]])], axis=0)
print("Trajectory points and directions", trajectoryPointsCameraSpace.shape, trajectoryPointsWorldSpace.shape, trajectoryDirectionsWorldSpace.shape)


## find directions from center of the car to the center of the camera
cameraPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
pointToCameraDirectionsWorldSpace = cameraPos.reshape([1, 3]) - trajectoryPointsWorldSpace
pointToCameraDistances = np.linalg.norm(pointToCameraDirectionsWorldSpace, axis=1)
pointToCameraDirectionsWorldSpace /= pointToCameraDistances.reshape([len(pointToCameraDistances), 1])

spacing = 98#1
subset = spacing+1#400
# for each point in the trajectory
preloadedPatches = {}
for idx, direction in enumerate(trajectoryDirectionsWorldSpace[:subset:spacing, :]) :
    i = idx*spacing
    rotAxis = np.cross(np.array([1, 0, 0]), direction)
    rotAxis /= np.linalg.norm(rotAxis)
    rotAngle = np.arccos(np.dot(direction, np.array([1, 0, 0])))
    
    
    ################ figure out how to turn the camera to look at the object ################
    ## undo rotation of car wrt camera
    M = quaternionTo4x4Rotation(angleAxisToQuaternion(rotAngle, rotAxis))
    rotatedDir = np.dot(M, np.array([[pointToCameraDirectionsWorldSpace[i, 0], pointToCameraDirectionsWorldSpace[i, 1], pointToCameraDirectionsWorldSpace[i, 2], 1]]).T)
    rotatedDir = rotatedDir[:-1, 0]/rotatedDir[-1, 0]
    rotatedDir /= np.linalg.norm(rotatedDir)
    
    ## this turns the camera towards the object
    adjustCamPos, adjustCamNorm = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics))
    adjustAxis = np.cross(-pointToCameraDirectionsWorldSpace[i, :], adjustCamNorm)
    adjustAxis /= np.linalg.norm(adjustAxis)
    adjustAngle = np.arccos(np.clip(np.dot(adjustCamNorm, -pointToCameraDirectionsWorldSpace[i, :]), -1, 1))
    adjustM = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))

    camMat = np.eye(4)
    camMat[:-1, -1] = rotatedDir
    camMat[:-1, :-1] = np.dot(adjustM[:-1, :-1], np.linalg.inv(cameraExtrinsics)[:-1, :-1])

    ## this rotates camera to align with ground plane (and the car itself)
    _, adjustCamRightVec2 = getWorldSpacePosAndNorm(camMat, np.array([[1, 0, 0, 1]], float).T)
    _, adjustCamUpVec2 = getWorldSpacePosAndNorm(camMat, np.array([[0, -1, 0, 1]], float).T)
    _, adjustCamNorm2 = getWorldSpacePosAndNorm(camMat)
    adjustAxis2 = np.copy(adjustCamNorm2)
#         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, adjustCamRightVec2-np.dot(adjustCamRightVec2, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0])), -1, 1)) ## aligns camera right vector to ground plane
#         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, np.array([1, 0, 0], float)), -1, 1)) ## not sure what this does
    if i < len(trajectoryPointsCameraSpace)-1 :
        trajDir = trajectoryPointsCameraSpace[i, :]-trajectoryPointsCameraSpace[i+1, :]
    else :
        trajDir = trajectoryPointsCameraSpace[i-1, :]-trajectoryPointsCameraSpace[i, :]
#     print(i, np.linalg.norm(trajDir))
    trajDir /= np.linalg.norm(trajDir)
    adjustAngle2 = np.arccos(np.clip(np.dot(trajDir, np.array([1, 0], float)), -1, 1)) ## align camera space direction to x axis (does it even make sense?)
    if np.cross(trajDir, np.array([1, 0], float)) < 0 :
        adjustAxis2 *= -1.0


    adjustM2 = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle2, adjustAxis2))
    camMat[:-1, :-1] = np.dot(M[:-1, :-1], np.dot(adjustM2[:-1, :-1], camMat[:-1, :-1]))
    
    #########################################################################################
    
    
    ################ rotate the camera to look at the car ################
    camPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
    rotatedCamTransform = rotateAboutPoint(np.linalg.inv(cameraExtrinsics), angleAxisToQuaternion(adjustAngle, adjustAxis), camPos)
#     rotatedCamTransform = rotateAboutPoint(rotatedCamTransform, angleAxisToQuaternion(adjustAngle2, adjustAxis2), camPos)


    _, camDir = getWorldSpacePosAndNorm(rotatedCamTransform, np.array([[0.0], [0.0], [1.0], [1.0]]))
    desiredDist = np.linalg.norm(camPos)#1.0
    t = camDir*(np.linalg.norm(trajectoryPointsWorldSpace[i, :]-camPos)-desiredDist)
    tMat = np.array([[1, 0, 0, t[0]],
                     [0, 1, 0, t[1]],
                     [0, 0, 1, t[2]],
                     [0, 0, 0, 1]])
    ################################### can use this to do the scale normalization thing ###################################
    tMat = np.eye(4)
    rotatedExtrinsics = np.dot(np.linalg.inv(rotatedCamTransform), np.linalg.pinv(tMat))
#     print("dist after tMat", np.linalg.norm(trajectoryPointsWorldSpace[i, :]-getWorldSpacePosAndNorm(np.linalg.inv(rotatedExtrinsics), posOnly=True)))
    
    #########################################################################################
    
    frameName = "frame-{0:05}.png".format(sortedPatchKeys[i+frameSubset[0]]+1)
    frameImg = np.array(Image.open("/home/ilisescu/PhD/data/havana/"+frameName)).astype(np.uint8)
#     figure(); imshow(frameImg)
#     scatter(objectData[DICT_BBOX_CENTERS][sortedPatchKeys[i]][0], objectData[DICT_BBOX_CENTERS][sortedPatchKeys[i]][1])

    frameSize = np.array([frameImg.shape[1], frameImg.shape[0]])
    print(i, "dist", np.linalg.norm(trajectoryPointsWorldSpace[i, :]-camPos), desiredDist, frameName)

    ## undistort image
    sizeDelta = 0.3
    newFrameSize = (frameSize*(1+sizeDelta)).astype(int)

    map1, map2 = cv2.initUndistortRectifyMap(originalIntrinsics, distortionCoeff, None, cameraIntrinsics, tuple(newFrameSize), cv2.CV_16SC2)
    undistortedUncropped = cv2.remap(frameImg, map1, map2, cv2.INTER_LINEAR)
    figure(); imshow(undistortedUncropped)
    scatter(trajectoryPointsCameraSpace[i, 0], trajectoryPointsCameraSpace[i, 1])
    jack = trajectoryPointsCameraSpace[i+1, :]-trajectoryPointsCameraSpace[i, :]
    jack /= np.linalg.norm(jack)
    plot(np.array([trajectoryPointsCameraSpace[i, 0], trajectoryPointsCameraSpace[i, 0]+jack[0]*50]), np.array([trajectoryPointsCameraSpace[i, 1], trajectoryPointsCameraSpace[i, 1]+jack[1]*50]))

    ## get grid points into world space and back into image space using the rotate extrinsics
    gridDownsample = 1
    imageGridPoints = np.indices(newFrameSize/gridDownsample).reshape([2, np.prod(newFrameSize/gridDownsample)]).T*gridDownsample
    
    ## figure out mapping between original camera matrix and the new one that looks at the car
    rotatedToWorld = np.linalg.inv(np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, [0, 1, 3]]))
    worldToOriginal = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, :])

    rotatedGridWorldSpace = np.dot(rotatedToWorld, np.concatenate([imageGridPoints, np.ones([len(imageGridPoints), 1], np.float64)], axis=1).T)
    rotatedGridWorldSpace /= rotatedGridWorldSpace[-1, :]
    rotatedGridWorldSpace[-1, :] = 0
    rotatedGridWorldSpace = rotatedGridWorldSpace.T.astype(np.float64)

    rotatedGridInOriginalCamera = np.dot(worldToOriginal, np.concatenate([rotatedGridWorldSpace, np.ones([len(rotatedGridWorldSpace), 1], np.float64)], axis=1).T)
    rotatedGridInOriginalCamera = (rotatedGridInOriginalCamera[:-1, :]/rotatedGridInOriginalCamera[-1, :]).T
    rotatedGridInOriginalCamera = rotatedGridInOriginalCamera.T.reshape([2, (newFrameSize/gridDownsample)[0], (newFrameSize/gridDownsample)[1]]).T.astype(np.float32)
    mapPoints1, mapPoints2 = cv2.convertMaps(rotatedGridInOriginalCamera, None, cv2.CV_16SC2)
    rotatedFrameImg = cv2.remap(undistortedUncropped, mapPoints1, mapPoints2, cv2.INTER_LINEAR)
    figure(); imshow(rotatedFrameImg)
    scatter((newFrameSize/2)[0], (newFrameSize/2)[1])
    plot(np.array([(newFrameSize/2)[0], (newFrameSize/2)[0]+jack[0]*50]), np.array([(newFrameSize/2)[1], (newFrameSize/2)[1]+jack[1]*50]))
    
    rotatedFrameAlpha = cv2.remap(cv2.remap(np.array(Image.open("/home/ilisescu/PhD/data/havana/blue_car1-maskedFlow-blended/"+frameName)).astype(np.uint8),
                                            map1, map2, cv2.INTER_LINEAR), mapPoints1, mapPoints2, cv2.INTER_LINEAR)
#     figure(); imshow(rotatedFrameAlpha)
    
    ## find patchsize and top left such that the center of the image is in the center of the image
    visiblePixels = np.argwhere(rotatedFrameAlpha[:, :, -1] != 0)
    imgCenter = np.array(rotatedFrameAlpha.shape[:2])/2
    halfSize = np.max(np.array([imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter]), axis=0)
    
    topLeft = imgCenter-halfSize #np.min(visiblePixels, axis=0)
    patchSize = halfSize*2 + 1 #np.max(visiblePixels, axis=0) - topLeft + 1
    
    colors = np.concatenate([rotatedFrameImg[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)
#     print(colors.shape)

    preloadedPatches[sortedPatchKeys[i]] = {'top_left_pos':topLeft, 'sprite_colors':colors[:, [2, 1, 0, 3]],
                                            'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}

# <codecell>

# 142 [-0.90505211 -0.42530069]
print(jack)
sortedPatchKeys[i+frameSubset[0]]+1
i+frameSubset[0]
sortedPatchKeys[i+frameSubset[0]]

# <codecell>

# figure(); plot(np.linalg.norm(trajectoryPointsWorldSpace-camPos, axis=1))
np.linalg.norm(trajectoryPointsWorldSpace-camPos, axis=1)

# <codecell>

# np.save("/home/ilisescu/PhD/data/havana/camera_adjusted_plus_scale_preloaded_patches-blue_car1.npy", preloadedPatches)
# print(trajectoryDirectionsWorldSpace)

# <codecell>

# print(len(preloadedPatches))
# figure(); imshow(rotatedFrameAlpha)
# print(np.min(visiblePixels, axis=0))
# print(len(sortedPatchKeys))
# print(patches[sortedPatchKeys[i]])
figure(); imshow(rotatedFrameImg)
# print(i)

# <codecell>

print(preloadedPatches.keys())
patch = preloadedPatches[sortedPatchKeys[193]]
img = np.zeros([patch['patch_size'][0], patch['patch_size'][1], 4], dtype=np.int8)
img[patch['visible_indices'][:, 0], patch['visible_indices'][:, 1], :] = patch['sprite_colors']
img = img[:, :, [2, 1, 0, 3]]
figure(); imshow(img.astype(np.uint8))
scatter((patch['patch_size']/2)[1], (patch['patch_size']/2)[0])

# <codecell>

print(patch['patch_size']/2)

# <codecell>

print(imgCenter, halfSize)
print(np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0), "\n")

print(topLeft, topLeft+patchSize)
print(np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0), "\n")

print(np.array([imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter]))
print(imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter)
print(np.max(np.array([imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter]), axis=0))

# <codecell>

## check if the trajectory point projects to the center of the image using the rotatedExtrinsics
print(trajectoryPointsCameraSpace[i, :])
print(objectData[DICT_BBOX_CENTERS][sortedPatchKeys[frameIdx]])
trajPoint = np.concatenate([trajectoryPointsWorldSpace[i, :], [1]]).reshape([1, 4])
print(trajectoryPointsWorldSpace[i, :], trajPoint)
projTrajPoint = np.dot(np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, :]), np.concatenate([trajectoryPointsWorldSpace[i, :], [1]]).reshape([1, 4]).T)
projTrajPoint /= projTrajPoint[-1, 0]
print(np.array(rotatedFrameAlpha.shape[:2])[::-1]/2, projTrajPoint.flatten())

# <codecell>

frustumEdges = np.array(np.array([[0, 0, 0, 1],
                                  [.25, .25, 1, 1],
                                  [.25, -.25, 1, 1],
                                  [-.25, -.25, 1, 1],
                                  [-.25, .25, 1, 1]]))
## load 3D data
data3D = np.load("tmp_trajectory_3D.npy").item()
print(data3D.keys())
cameraExtrinsics = data3D['extrinsics']
cameraIntrinsics = data3D['intrinsics']
### NEED TO SAVE THIS IN THE 3D DATA BUT WHATEVS, FOR NOW IT'S HARDCODED FOR BLUE_CAR1
# frameSubset = [30, -50]
# frameSubset = [0, -1]

# trajectoryPointsCameraSpace = data3D['trajectoryPointsCameraSpace']
# trajectoryPointsWorldSpace = data3D['trajectoryPointsWorldSpace']
# trajectoryDirectionsWorldSpace = data3D['trajectoryDirectionsWorldSpace']

## set up figure
fig = figure()
ax = fig.add_subplot(111, aspect='equal', projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
cols = cm.jet(np.arange(len(trajectoryPointsWorldSpace), dtype=float)/len(trajectoryPointsWorldSpace))

## plot car footprint and look direction as [1, 0, 0]
if True :
    footprint = np.array([[0.8, 0.5, 0.0],
                          [0.8, -0.5, 0.0],
                          [-0.8, -0.5, 0.0],
                          [-0.8, 0.5, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.8, 0.0, 0.0]]).T*0.5
    for i, j in zip([0, 1, 2, 3, 4], [1, 2, 3, 0, 5]) :
        ax.plot(np.array([footprint[0, i], footprint[0, j]]), np.array([footprint[1, i], footprint[1, j]]), np.array([footprint[2, i], footprint[2, j]]), c="magenta")

## plot normalized directions
if True :
    for i, direction in enumerate(trajectoryDirectionsWorldSpace) :
        ax.plot(np.array([0, direction[0]]), np.array([0, direction[1]]), np.zeros(2), c=cols[i, :])
    
## plot trajectory
if True :
    ax.plot(trajectoryPointsWorldSpace[:, 0], trajectoryPointsWorldSpace[:, 1], np.zeros(len(trajectoryPointsWorldSpace)), c="cyan")

## find directions from center of the car to the center of the camera
cameraPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
pointToCameraDirectionsWorldSpace = cameraPos.reshape([1, 3]) - trajectoryPointsWorldSpace
pointToCameraDistances = np.linalg.norm(pointToCameraDirectionsWorldSpace, axis=1)
pointToCameraDirectionsWorldSpace /= pointToCameraDistances.reshape([len(pointToCameraDistances), 1])

## plot pointToCameraDistances
if False :
    for i, [direction, position] in enumerate(zip(pointToCameraDirectionsWorldSpace[:, :], trajectoryPointsWorldSpace[:, :])) :
        ax.plot(np.array([position[0], direction[0]*pointToCameraDistances[i]+position[0]]),
                np.array([position[1], direction[1]*pointToCameraDistances[i]+position[1]]),
                np.array([position[2], direction[2]*pointToCameraDistances[i]+position[2]]), c=cols[i, :])
        

## find rotation to align trajectory directions to [1, 0, 0] and use it to rotate the pointToCameraDirections
spacing = 13
subset = 400
scaledFrustumEdges = np.copy(frustumEdges.T)
scaledFrustumEdges[0:2, :] *= .04
scaledFrustumEdges[2, :] *= .05
for idx, direction in enumerate(trajectoryDirectionsWorldSpace[:subset:spacing, :]) :
    i = idx*spacing
    lastI = np.copy(i)
    rotAxis = np.cross(np.array([1, 0, 0]), direction)
    rotAxis /= np.linalg.norm(rotAxis)
    rotAngle = np.arccos(np.dot(direction, np.array([1, 0, 0])))
    
    M = quaternionTo4x4Rotation(angleAxisToQuaternion(rotAngle, rotAxis))
    rotatedDir = np.dot(M, np.array([[pointToCameraDirectionsWorldSpace[i, 0], pointToCameraDirectionsWorldSpace[i, 1], pointToCameraDirectionsWorldSpace[i, 2], 1]]).T)
    rotatedDir = rotatedDir[:-1, 0]/rotatedDir[-1, 0]
    rotatedDir /= np.linalg.norm(rotatedDir)
    if False :
        ax.plot(np.array([0, rotatedDir[0]]), np.array([0, rotatedDir[1]]), np.array([0, rotatedDir[2]]), c=cols[i, :])
    
    if True :
        ## this turns the camera towards the object
        adjustCamPos, adjustCamNorm = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics))
        adjustAxis = np.cross(-pointToCameraDirectionsWorldSpace[i, :], adjustCamNorm)
        adjustAxis /= np.linalg.norm(adjustAxis)
        adjustAngle = np.arccos(np.clip(np.dot(adjustCamNorm, -pointToCameraDirectionsWorldSpace[i, :]), -1, 1))
        adjustM = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))
        print(i, -rotatedDir, pointToCameraDirectionsWorldSpace[i, :], direction)
        
        camMat = np.eye(4)
        camMat[:-1, -1] = rotatedDir
        camMat[:-1, :-1] = np.dot(adjustM[:-1, :-1], np.linalg.inv(cameraExtrinsics)[:-1, :-1])
#         camMat[:-1, :-1] = np.dot(M[:-1, :-1], np.dot(adjustM2[:-1, :-1], np.dot(adjustM[:-1, :-1], np.linalg.inv(cameraExtrinsics)[:-1, :-1])))
        
        ## this rotates camera to align with ground plane (and the car itself)
        _, adjustCamRightVec2 = getWorldSpacePosAndNorm(camMat, np.array([[1, 0, 0, 1]], float).T)
        _, adjustCamUpVec2 = getWorldSpacePosAndNorm(camMat, np.array([[0, -1, 0, 1]], float).T)
        _, adjustCamNorm2 = getWorldSpacePosAndNorm(camMat)
        adjustAxis2 = np.copy(adjustCamNorm2)
#         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, adjustCamRightVec2-np.dot(adjustCamRightVec2, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0])), -1, 1)) ## aligns camera right vector to ground plane
#         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, np.array([1, 0, 0], float)), -1, 1)) ## not sure what this does
        trajDir = trajectoryPointsCameraSpace[i, :]-trajectoryPointsCameraSpace[i+1, :]
        trajDir /= np.linalg.norm(trajDir)
        adjustAngle2 = np.arccos(np.clip(np.dot(trajDir, np.array([1, 0], float)), -1, 1)) ## align camera space direction to x axis (does it even make sense?)
        if np.cross(trajDir, np.array([1, 0], float)) < 0 :
            adjustAxis2 *= -1.0
        
        
        adjustM2 = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle2, adjustAxis2))
        camMat[:-1, :-1] = np.dot(M[:-1, :-1], np.dot(adjustM2[:-1, :-1], camMat[:-1, :-1]))
        
        camFrustum = np.dot(camMat, np.concatenate([scaledFrustumEdges, np.array([[0, 0, 1, 1]]).T], axis=1))
        camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
        for idxI, idxJ in zip([0, 0, 0, 0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 2, 3, 4, 1, 5]) :
            ax.plot(np.array([camFrustum[0, idxI], camFrustum[0, idxJ]]), np.array([camFrustum[1, idxI], camFrustum[1, idxJ]]), np.array([camFrustum[2, idxI], camFrustum[2, idxJ]]), c=cols[i, :], linewidth=.5)
            
        camPos, camUp = getWorldSpacePosAndNorm(camMat, np.array([[0.0, -1.0, 0.0, 1.0]]).T)
        camUp *=0.05
        ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]), c="green")
        
        _, bob = getWorldSpacePosAndNorm(camMat, np.array([[1, 0, 0, 1]], float).T)
#         print(adjustAngle2, bob, adjustCamRightVec2, adjustCamRightVec2-np.dot(adjustCamRightVec2, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0]),
#               np.arccos(np.clip(np.dot(bob, bob-np.dot(bob, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0])), -1, 1)))
    
#     ax.scatter(-tmpDirections[lastI, 0], -tmpDirections[lastI, 1], -tmpDirections[lastI, 2])
        
        
## plot captured camera frustum
if True :
    camFrustum = np.dot(np.linalg.inv(cameraExtrinsics), frustumEdges.T)
    camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
    camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
    for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
        ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="blue")
    ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]))
        
        
## draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
x=np.cos(u)*np.sin(v)
y=np.sin(u)*np.sin(v)
z=np.cos(v)
ax.plot_wireframe(x, y, z, color="r", linewidth=.2)

# <codecell>

13 [-0.12638843 -0.88260246 -0.45281218] [-0.08635512 -0.88741416  0.45281217] [-0.99897742 -0.04521232  0.        ]
13 [-0.12638853 -0.8826024  -0.45281225] [-0.08635522 -0.88741413  0.45281225] [-0.99897742 -0.04521232  0.        ]

# <codecell>

TRALALA 0 [-0.32530311 -0.89164722  0.31487012] [-0.28411783 -0.90561243 -0.31487011]

# <codecell>

objectData = np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()
patches = np.load(objectData[DICT_PATCHES_LOCATION]).item()
sortedPatchKeys = np.sort(patches.keys())

frameIdx = lastI+frameSubset[0]
patch = patches[sortedPatchKeys[frameIdx]]
img = np.zeros([patch['patch_size'][0], patch['patch_size'][1], 4], dtype=np.int8)
img[patch['visible_indices'][:, 0], patch['visible_indices'][:, 1], :] = patch['sprite_colors']
img = img[:, :, [2, 1, 0, 3]]

# figure(); imshow(img[:, :, :-1].astype(np.uint8))

camPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
rotatedCamTransform = rotateAboutPoint(np.linalg.inv(cameraExtrinsics), angleAxisToQuaternion(adjustAngle, adjustAxis), camPos)
rotatedCamTransform = rotateAboutPoint(rotatedCamTransform, angleAxisToQuaternion(adjustAngle2, adjustAxis2), camPos)


_, camDir = getWorldSpacePosAndNorm(rotatedCamTransform, np.array([[0.0], [0.0], [1.0], [1.0]]))
t = camDir*1.3
tMat = np.array([[1, 0, 0, t[0]],
                 [0, 1, 0, t[1]],
                 [0, 0, 1, t[2]],
                 [0, 0, 0, 1]])
################################### can use this to do the scale normalization thing ###################################
tMat = np.eye(4)
rotatedExtrinsics = np.dot(np.linalg.inv(rotatedCamTransform), np.linalg.pinv(tMat))

print(np.linalg.inv(rotatedExtrinsics))
print(np.linalg.inv(cameraExtrinsics))

camFrustum = np.dot(np.linalg.inv(rotatedExtrinsics), frustumEdges.T)
camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(rotatedExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
    ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="magenta")
ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]), c="magenta")

# <codecell>

distortionParameter = -0.19
distortionRatio = -0.19
distortionCoeff = np.array([distortionParameter, distortionParameter*distortionRatio, 0.0, 0.0, 0.0])

        
originalIntrinsics = np.array([[702.736053, 0.0, 640.0],
                               [0.0, 702.736053, 360.0],
                               [0.0, 0.0, 1.0]])

frameImg = np.array(Image.open("/home/ilisescu/PhD/data/havana/frame-{0:05}.png".format(sortedPatchKeys[frameIdx]+1))).astype(np.uint8)
figure(); imshow(frameImg)
scatter(objectData[DICT_BBOX_CENTERS][sortedPatchKeys[frameIdx]][0], objectData[DICT_BBOX_CENTERS][sortedPatchKeys[frameIdx]][1])

frameSize = np.array([frameImg.shape[1], frameImg.shape[0]])

## undistort image
sizeDelta = 0.3
newFrameSize = (frameSize*(1+sizeDelta)).astype(int)

map1, map2 = cv2.initUndistortRectifyMap(originalIntrinsics, distortionCoeff, None, cameraIntrinsics, tuple(newFrameSize), cv2.CV_16SC2)
undistortedUncropped = cv2.remap(frameImg, map1, map2, cv2.INTER_LINEAR)
figure(); imshow(undistortedUncropped)
scatter(trajectoryPointsCameraSpace[lastI, 0], trajectoryPointsCameraSpace[lastI, 1])

## get grid points into world space and back into image space using the rotate extrinsics
gridDownsample = 1
imageGridPoints = np.indices(newFrameSize/gridDownsample).reshape([2, np.prod(newFrameSize/gridDownsample)]).T*gridDownsample

if False :
    print("using old way")
    inverseT = np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]))
    projectedImageGridPoints = np.dot(inverseT, np.concatenate([imageGridPoints, np.ones([len(imageGridPoints), 1], np.float64)], axis=1).T)
    projectedImageGridPoints /= projectedImageGridPoints[-1, :]

    projectedImageGridPoints[-1, :] = 0
    projectedImageGridPoints = projectedImageGridPoints.T.astype(np.float64)

    T = np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, :])
    rotatedGridPoints = np.dot(T, np.concatenate([projectedImageGridPoints, np.ones([len(projectedImageGridPoints), 1], np.float64)], axis=1).T)
    mapPoints1, mapPoints2 = cv2.convertMaps((rotatedGridPoints[:-1, :]/rotatedGridPoints[-1, :]).T.reshape([2, (newFrameSize/gridDownsample)[0], (newFrameSize/gridDownsample)[1]]).T.astype(np.float32), None, cv2.CV_16SC2)
    rotatedGridPoints = np.round((rotatedGridPoints[:-1, :]/rotatedGridPoints[-1, :]).T).astype(int)


    validCoords = np.all(np.concatenate([rotatedGridPoints >= 0,
                                         (rotatedGridPoints[:, 0] < newFrameSize[0]).reshape([len(rotatedGridPoints), 1]),
                                         (rotatedGridPoints[:, 1] < newFrameSize[1]).reshape([len(rotatedGridPoints), 1])], axis=1), axis=1)

    rotatedFrameImg = np.zeros(undistortedUncropped.shape, np.uint8)
    rotatedFrameImg[rotatedGridPoints[validCoords, 1], rotatedGridPoints[validCoords, 0], :] = undistortedUncropped[imageGridPoints[validCoords, 1], imageGridPoints[validCoords, 0], :]
else :
    print("using opencv")
    rotatedToWorld = np.linalg.inv(np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, [0, 1, 3]]))
    worldToOriginal = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, :])

    rotatedGridWorldSpace = np.dot(rotatedToWorld, np.concatenate([imageGridPoints, np.ones([len(imageGridPoints), 1], np.float64)], axis=1).T)
    rotatedGridWorldSpace /= rotatedGridWorldSpace[-1, :]
    rotatedGridWorldSpace[-1, :] = 0
    rotatedGridWorldSpace = rotatedGridWorldSpace.T.astype(np.float64)

    rotatedGridInOriginalCamera = np.dot(worldToOriginal, np.concatenate([rotatedGridWorldSpace, np.ones([len(rotatedGridWorldSpace), 1], np.float64)], axis=1).T)
    rotatedGridInOriginalCamera = (rotatedGridInOriginalCamera[:-1, :]/rotatedGridInOriginalCamera[-1, :]).T
    rotatedGridInOriginalCamera = rotatedGridInOriginalCamera.T.reshape([2, (newFrameSize/gridDownsample)[0], (newFrameSize/gridDownsample)[1]]).T.astype(np.float32)
    mapPoints1, mapPoints2 = cv2.convertMaps(rotatedGridInOriginalCamera, None, cv2.CV_16SC2)
    rotatedFrameImg = cv2.remap(undistortedUncropped, mapPoints1, mapPoints2, cv2.INTER_LINEAR)
figure(); imshow(rotatedFrameImg)

scatter((newFrameSize/2)[0], (newFrameSize/2)[1])

# <codecell>

modelMat = QtGui.QMatrix4x4(0.910911599349, 0.406799823245, 0.0689489809887, 1.04144313517,
                            0.201452403084, -0.292662295856, -0.934754358041, -0.491083466897,
                            -0.360079140556, 0.865368525191, -0.348540281954, 3.93563077185,
                            0.0, 0.0, 0.0, 1.0).inverted()[0]

cameraIntrinsics = np.array([[702.736053, 0.0, 640.0],
                             [0.0, 702.736053, 360.0],
                             [0.0, 0.0, 1.0]])
cameraExtrinsics = np.array(modelMat.inverted()[0].data()).reshape([4, 4]).T
print("README", cameraExtrinsics)

frameSize = np.array([1280, 720])
gridDownsample = 1
projectedImageGridPoints = np.indices(frameSize/gridDownsample).reshape([2, np.prod(frameSize/gridDownsample)]).T*gridDownsample
# projectedImageGridColors = img[projectedImageGridPoints[:, 1], projectedImageGridPoints[:, 0], :].astype(np.float32)/np.float32(255.0)
tmp = np.copy(projectedImageGridPoints)

inverseT = np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]))
projectedImageGridPoints = np.dot(inverseT, np.concatenate([projectedImageGridPoints, np.ones([len(projectedImageGridPoints), 1], np.float64)], axis=1).T)
projectedImageGridPoints /= projectedImageGridPoints[-1, :]
projectedImageGridPoints[-1, :] = 0
projectedImageGridPoints = projectedImageGridPoints.T.astype(np.float64)

# <codecell>

## USED TO FIGURE OUT HOW TO ALIGN A CAMERA TO THE GROUND PLANE
fig = figure()
ax = fig.add_subplot(111, aspect='equal', projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

camFrustum = np.dot(np.linalg.inv(cameraExtrinsics), frustumEdges.T)
camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
_, camNorm = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), np.array([[0.0, 0.0, 1.0, 1.0]]).T)
for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
    ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="blue")
ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]))

frustumCenter = np.average(camFrustum[:, 1:], axis=1)
ax.plot(np.array([frustumCenter[0], frustumCenter[0]+camUp[0]]), np.array([frustumCenter[1], frustumCenter[1]+camUp[1]]), np.array([frustumCenter[2], frustumCenter[2]+camUp[2]]), c="magenta")
planeNorm = np.array([0.0, 0.0, 1.0])
ax.plot(np.array([frustumCenter[0], frustumCenter[0]+planeNorm[0]]), np.array([frustumCenter[1], frustumCenter[1]+planeNorm[1]]), np.array([frustumCenter[2], frustumCenter[2]+planeNorm[2]]), c="cyan")
ax.plot(np.array([frustumCenter[0], frustumCenter[0]+camNorm[0]]), np.array([frustumCenter[1], frustumCenter[1]+camNorm[1]]), np.array([frustumCenter[2], frustumCenter[2]+camNorm[2]]), c="blue")

projNorm = planeNorm - np.dot(planeNorm, camNorm)*camNorm
projNorm /= np.linalg.norm(projNorm)
ax.plot(np.array([frustumCenter[0], frustumCenter[0]+projNorm[0]]), np.array([frustumCenter[1], frustumCenter[1]+projNorm[1]]), np.array([frustumCenter[2], frustumCenter[2]+projNorm[2]]), c="black")

adjustAngle3 = np.arccos(np.clip(np.dot(projNorm, camUp), -1, 1))
adjustAxis3 = np.cross(projNorm, camUp)
adjustAxis3 /= np.linalg.norm(adjustAxis3)
rotatedCamTransform = rotateAboutPoint(np.linalg.inv(cameraExtrinsics), angleAxisToQuaternion(adjustAngle3, adjustAxis3), camPos)
camFrustum = np.dot(rotatedCamTransform, frustumEdges.T)
camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(rotatedExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
    ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="magenta")
ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]), c="magenta")

