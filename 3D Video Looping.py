
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab')
from __future__ import print_function
from matplotlib import cm
import sys
import numpy as np
import time
import os
import glob
import pyassimp
import opengm
import cv2
from scipy import ndimage as spimg
from scipy import special
from scipy.spatial import ConvexHull
from sklearn import linear_model

from skimage import measure
from scipy.optimize import minimize

sys.path.append('CMT tracker/')
import CMT
import triangulate
from ICPmatching import icp

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



DICT_FILMED_OBJECT_NAME = 'filmed_object_name'
DICT_TRAJECTORY_POINTS = 'trajectory_points'
DICT_NEEDS_UNDISTORT = 'do_undistort_trajectory_points'
DICT_OBJECT_BILLBOARD_ORIENTATION = 'object_color_billboard_orientation_angle'
DICT_OBJECT_BILLBOARD_SCALE = 'object_color_bilboard_scale'
DICT_TRACK_LOCATION='track_points_location'

DICT_FILMED_SCENE_BASE_LOC = 'filmed_scene_base_location'
DICT_CAMERA_EXTRINSICS = 'camera_extrinsics'
DICT_CAMERA_INTRINSICS = 'camera_intrinsics'
DICT_DISTORTION_PARAMETER = 'distortion_parameter'
DICT_DISTORTION_RATIO = 'distortion_ratio'
DICT_DOWNSAMPLED_FRAMES_RATE = 'downsampled_frames_rate'
DICT_COMMENTS = "comments_and_info"
DICT_GROUND_MESH_POINTS = 'camera_ground_plane_mesh_points'
DICT_GROUND_MESH_SEGS_EXTRUDE = 'ground_plane_mesh_segments_to_extrude'


# In[2]:

## compute euclidean distance assuming f is an array where each row is a flattened image (1xN array, N=W*H*Channels)
## euclidean distance defined as the length of the the displacement vector:
## len(q-p) = sqrt(len(q)^2+len(p)^2 - 2*dot(p, q)) where p and q are two images in vector format and 1xN size
def ssd(f) :
    ## gives sum over squared intensity values for each image
    ff = np.sum(f*f, axis=1)
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
    d = np.reshape(ff, [len(ff),1])+ff.T - 2*np.dot(f, f.T)
    return d

def ssd2(f1, f2) :
    ## gives sum over squared intensity values for each image
    ff1 = np.sum(f1*f1, axis=1)
    ff2 = np.sum(f2*f2, axis=1)
#     print ff1.shape
#     print ff2.shape
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
#     print "askdfh", np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1).shape, np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0).shape
    d = np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1)+np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0) - 2*np.dot(f1, f2.T)
    return d


# In[3]:

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

# Vertex shader
VS_PROJECTIVE = """
#version 330

layout(location = 0) in vec3 position_model;

uniform mat4 m_pvm;
uniform mat4 m_proj_mat;

out Data {
    vec4 vertex_proj;
} DataOut;

void main () {
    gl_Position = m_pvm * vec4(position_model, 1.0);
    DataOut.vertex_proj = m_proj_mat * vec4(position_model, 1.0);
}
"""

# Fragment shader
FS_PROJECTIVE = """
#version 330

uniform sampler2D texture_sampler;

out vec4 out_color;

in Data {
    vec4 vertex_proj;
} DataIn;

void main() {
    vec2 uv_model;
    uv_model.x = (DataIn.vertex_proj.x/DataIn.vertex_proj.w+1.0)/2.0;
    uv_model.y = (-DataIn.vertex_proj.y/DataIn.vertex_proj.w+1.0)/2.0;
    if(uv_model.x >= 0.0 && uv_model.x <= 1.0 && uv_model.y >= 0.0 && uv_model.y <= 1.0) {
        out_color = texture(texture_sampler, uv_model);
    }
    //else {
    //    out_color = vec4(1, 1, 1, 0);
    //}
}
"""


# In[4]:

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


# In[5]:

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


# In[67]:

def undistortImage(distortionParameter, distortionRatio, image, cameraIntrinsics, doUncrop=True, interpolation=cv2.INTER_LANCZOS4) :
    distortionCoeff = np.array([distortionParameter, distortionParameter*distortionRatio, 0.0, 0.0, 0.0])

    frameSize = np.array([image.shape[1], image.shape[0]])

    ## undistort image
    if doUncrop :
        ## here I was just making the image I project the undistorted pixels to bigger
#         sizeDelta = 0.3
#         newFrameSize = (frameSize*(1+sizeDelta)).astype(int)
#         newIntrinsics = np.copy(cameraIntrinsics)
#         newIntrinsics[0, 2] += image.shape[1]*sizeDelta/2.0
#         newIntrinsics[1, 2] += image.shape[0]*sizeDelta/2.0
        ## here I instead use opencv to figure out the best new camera matrix that includes all possible pixels
        newIntrinsics = cv2.getOptimalNewCameraMatrix(cameraIntrinsics, distortionCoeff, tuple(frameSize), 1)[0]
        ## the above, changes the focal length to see the full scene, but I want to keep focal length and have a bigger image instead, so I change the intrinsics to get the original focal length but bigger image
        scale = cameraIntrinsics[0, 0]/newIntrinsics[0, 0]
        newFrameSize = np.ceil(np.copy(frameSize)*scale).astype(int)
        newIntrinsics[0, 0] *= scale
        newIntrinsics[1, 1] *= scale
        newIntrinsics[:-1, -1] = newFrameSize/2.0
    else :
        newIntrinsics = np.copy(cameraIntrinsics)
        newFrameSize = np.copy(frameSize)
    
    map1, map2 = cv2.initUndistortRectifyMap(cameraIntrinsics, distortionCoeff, None, newIntrinsics, tuple(newFrameSize), cv2.CV_16SC2)
    undistortedImage = cv2.remap(image, map1, map2, interpolation)
    return undistortedImage, newIntrinsics, distortionCoeff, map1, map2

def line2lineIntersection(line1, line2) :
    """x1, y1, x2, y2 = line1
       x3, y3, x4, y4 = line2"""
    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if denominator != 0 :
        Px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denominator
        Py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denominator
        return np.array([Px, Py])
    else :
        raise Exception("lines are parallel")

def isABetweenBandC(a, b, c):
    distAB = np.linalg.norm(a-b)
    distAC = np.linalg.norm(a-c)
    distBC = np.linalg.norm(b-c)
    return np.abs(distAB+distAC-distBC) < 1e-10
        
def cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, imageShape) :
    """ return viewMat, projectionMat """
    
    viewMat = np.copy(cameraExtrinsics)
    ## flip z and y axis because of opencv vs opengl coord systems
    viewMat[2, :] *= -1
    viewMat[1, :] *= -1

    K = np.copy(cameraIntrinsics)
    ## changing signs for the same reason as above for the viewMat
    K[:, 2] *= -1
    K[:, 1] *= -1
    near = 0.1
    far = 100.0
    projectionMat = np.zeros([4, 4])
    projectionMat[:2, :-1] = K[:2, :]
    projectionMat[-1, :-1] = K[-1, :]
    projectionMat[2, 2] = near + far
    projectionMat[2, 3] = near*far

    left = 0.0
    right = float(imageShape[1])
    bottom = float(imageShape[0])
    top = 0.0

    projectionMat = np.dot(np.array([[2/(right-left), 0, 0, -(right+left)/(right-left)],
                                     [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
                                     [0, 0, -2/(far-near), -(far+near)/(far-near)],
                                     [0, 0, 0, 1]]), np.copy(projectionMat))
    ## returns new fov
#     return np.arctan2(1.0, projectionMat[1, 1])*2.0*180.0/np.pi
    return viewMat, projectionMat

def worldToScreenSpace(viewMat, projectionMat, worldSpacePoint, viewportWidth, viewportHeight) :
    """worldSpacePoint can be either a vector of length 3 or it can be a matrix Nx3"""
    if len(worldSpacePoint.shape) == 1 :
        worldSpacePoints = np.reshape(worldSpacePoint, [1, 3])
    else :
        worldSpacePoints = worldSpacePoint
    
    screenSpacePoints = np.dot(np.dot(projectionMat, viewMat), np.hstack([worldSpacePoints, np.ones([len(worldSpacePoints), 1])]).T)
    screenSpacePoints = screenSpacePoints[:-1, :]/screenSpacePoints[-1, :]
    screenSpacePoints = screenSpacePoints.T
    
    ## from clip space to screen space
    screenSpacePoints = np.hstack([((screenSpacePoints[:, 0]+1.0)*viewportWidth/2.0)[:, np.newaxis], 
                                   ((1.0-screenSpacePoints[:, 1])*viewportHeight/2.0)[:, np.newaxis]])
    if len(screenSpacePoints) == 1 :
        return screenSpacePoints.flatten()
    else :
        return screenSpacePoints
        

def triangulate2DPolygon(poly2D, doReturnIndices=True) :
    pts = [(point[0], point[1]) for point in poly2D]
    availableIndices = np.ones(len(pts), dtype=bool)
    tris = []
    plist = pts[::-1] if triangulate.IsClockwise(pts) else pts[:]
    while len(plist) >= 3:
        a = triangulate.GetEar(plist, np.arange(len(pts), dtype=int), availableIndices, doReturnIndices)
        if a == []:
            break
        if doReturnIndices :
            if triangulate.IsClockwise(pts) :
                tris.append([len(pts)-1-a[0], len(pts)-1-a[1], len(pts)-1-a[2]])
            else :
                tris.append(list(a))
        else :
            tris.append(a)
            
    return tris

def extrudeSegment(points, height, viewLoc) :
    inputVertices = np.vstack([points, points[::-1, :]+np.array([0.0, 0.0, height])])
    outputIndices = [0, 1, 3, 1, 2, 3]
    
    ## check that the triangle is front facing --> https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/single-vs-double-sided-triangle-backface-culling
    vertices = inputVertices[outputIndices[:3], :]
    N = np.cross(vertices[1, :]-vertices[0, :], vertices[2, :]-vertices[0, :])
    N /= np.linalg.norm(N)
    viewDir = vertices[0, :]-viewLoc
    viewDir /= np.linalg.norm(viewDir)
    
    ## it is back-facing so need to reverse dir
    if np.dot(viewDir, N) > 0 :
        outputIndices[0], outputIndices[1] = outputIndices[1], outputIndices[0]
        outputIndices[3], outputIndices[4] = outputIndices[4], outputIndices[3]
    return inputVertices[outputIndices, :], outputIndices

def isPoint2DInTriangle2D(point2D, triangle2D) :
    ## formulas from here http://mathworld.wolfram.com/TriangleInterior.html
    v = point2D
    v0 = triangle2D[0, :]
    v1 = triangle2D[1, :]-v0
    v2 = triangle2D[2, :]-v0
    
    vv2 = np.hstack([v.reshape([2, 1]), v2.reshape([2, 1])])
    v0v2 = np.hstack([v0.reshape([2, 1]), v2.reshape([2, 1])])
    v1v2 = np.hstack([v1.reshape([2, 1]), v2.reshape([2, 1])])
    vv1 = np.hstack([v.reshape([2, 1]), v1.reshape([2, 1])])
    v0v1 = np.hstack([v0.reshape([2, 1]), v1.reshape([2, 1])])
    
    a = (np.linalg.det(vv2)-np.linalg.det(v0v2))/np.linalg.det(v1v2)
    b = -(np.linalg.det(vv1)-np.linalg.det(v0v1))/np.linalg.det(v1v2)
    
    return a > 0 and b > 0 and (a+b) < 1

def getGridPointsInPolygon2D(polygon2D, gridSpacing) :
    triangles = np.array(triangulate2DPolygon(polygon2D, False))
    minBounds = ((np.min(polygon2D, axis=0)>0)*2-1)*np.ceil(np.abs(np.min(polygon2D, axis=0)))
    maxBounds = ((np.max(polygon2D, axis=0)>0)*2-1)*np.ceil(np.abs(np.max(polygon2D, axis=0)))

    gridPoints = np.mgrid[minBounds[0]:maxBounds[0]+gridSpacing:gridSpacing, minBounds[1]:maxBounds[1]+gridSpacing:gridSpacing]
    gridPoints = gridPoints.reshape([2, gridPoints.shape[1]*gridPoints.shape[2]]).T
    
    validPoints = []
    for pointIdx, point in enumerate(gridPoints) :
        for triangle in triangles :
            if isPoint2DInTriangle2D(point, triangle) :
                validPoints.append(pointIdx)
                break
    return gridPoints[validPoints, :]


# In[7]:

def smoothTrajectory(filterSize, trajectory) :
    sigma = filterSize/5.0 #1.2 #2.5
    coeff = special.binom(filterSize*2, range(0, filterSize*2 +1))
    coeff /= np.sum(coeff)
    neighbourIdxs = np.arange(-filterSize, filterSize+1)
#     print(coeff, neighbourIdxs)
    
#     smoothed = np.array([np.convolve(trajectory[:, 0], coeff, mode='valid'),
#                          np.convolve(trajectory[:, 1], coeff, mode='valid')]).T.astype(np.float32)
    smoothed = np.zeros_like(trajectory)
    
    for i, point in enumerate(trajectory) :
        validIdxs = np.all(np.array([i+neighbourIdxs >= 0, i+neighbourIdxs < len(trajectory)]), axis=0)
        closenessToEdge = filterSize*2+1-len(np.argwhere(validIdxs).flatten())
        filterCoeffs = coeff**np.exp(closenessToEdge/sigma)
        filterCoeffs /= np.sum(filterCoeffs)
#         print(i, point, i+neighbourIdxs[validIdxs], closenessToEdge, np.exp(closenessToEdge/sigma), np.round(filterCoeffs, decimals=2)[validIdxs])
#         print(np.sum(trajectory[i+neighbourIdxs[validIdxs]]*filterCoeffs[validIdxs].reshape([len(np.argwhere(validIdxs).flatten()), 1]), axis=0))
        smoothed[i, :] = np.sum(trajectory[i+neighbourIdxs[validIdxs]]*filterCoeffs[validIdxs].reshape([len(np.argwhere(validIdxs).flatten()), 1]), axis=0)
    
    return smoothed#[filterSize:-filterSize]

# smoothedTrajectory = smoothTrajectory(15, trajectoryPoints)

# figure()
# imshow(medianImage)
# xlim([0, medianImage.shape[1]])
# ylim([medianImage.shape[0], 0])
# # xlim([510, 1510])
# # ylim([760, 460])
# scatter(trajectoryPoints[:, 0], trajectoryPoints[:, 1], marker='o', color='blue', facecolors='none')
# scatter(smoothedTrajectory[:, 0], smoothedTrajectory[:, 1], marker='x', color='red')


# In[8]:

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.setMouseTracking(True)
        
        self.image = None
        self.qImage = None
        self.bbox = None
        self.bboxColor = [0, 255, 255, 255]
        
    def setImage(self, image) :
        if np.all(image != None) :
            self.image = np.ascontiguousarray(image.copy())
            if self.width() != self.image.shape[1] or self.height() != self.image.shape[0] :
                self.setFixedSize(self.image.shape[1], self.image.shape[0])
            self.qImage = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QtGui.QImage.Format_RGB888);
            self.update()
        else :
            self.image = None
            self.qImage = None
            
    def setBBox(self, bbox, bboxColor) :
        if np.all(bbox != None) :
            self.bbox = bbox
            self.bboxColor = bboxColor
            self.update()
        else :
            self.bbox = None
            self.bboxColor = [0, 255, 255, 255]
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        if np.all(self.qImage != None) :
            painter.drawImage(QtCore.QPoint(0, 0), self.qImage)
            
        if np.all(self.bbox != None) :
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(self.bboxColor[0], self.bboxColor[1], self.bboxColor[2], self.bboxColor[3]), 3, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
            
            painter.drawRect(QtCore.QRectF(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]))
            
            
            painter.drawPoint(QtCore.QPointF(self.bbox[0]+self.bbox[2]/2.0, self.bbox[1]+self.bbox[3]/2.0))
            
        painter.end()


# In[9]:

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
        arrowSpacing = 0.0 #arrowLength*0.2
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
            ## add a bunch of points at the beginning and end of worldTrjectoryPoints so that the beginning and end points don't deviate massively from the original value
#             extraPoints = 15
#             self.worldTrajectoryPoints = np.concatenate([self.worldTrajectoryPoints[0, :].reshape([1, 3]).repeat(15, axis=0),
#                                                          self.worldTrajectoryPoints,
#                                                          self.worldTrajectoryPoints[-1, :].reshape([1, 3]).repeat(15, axis=0)], axis=0)
            
#             ## smooth trajectory
#             self.worldTrajectoryPoints = np.array([spimg.filters.gaussian_filter1d(self.worldTrajectoryPoints[:, 0], self.trajectorySmoothness, axis=0, mode='nearest'),
#                                                    spimg.filters.gaussian_filter1d(self.worldTrajectoryPoints[:, 1], self.trajectorySmoothness, axis=0, mode='nearest'),
#                                                    spimg.filters.gaussian_filter1d(self.worldTrajectoryPoints[:, 2], self.trajectorySmoothness, axis=0, mode='nearest')]).T.astype(np.float32)[extraPoints:-extraPoints, :]
            
            self.worldTrajectoryPoints = smoothTrajectory(15, self.worldTrajectoryPoints)

            ## reproject points into image space after smoothing
            T = np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]])
            self.cameraTrajectoryPoints = np.dot(T, np.concatenate([self.worldTrajectoryPoints[:, :-1], np.ones([len(self.worldTrajectoryPoints), 1])], axis=1).T)
            self.cameraTrajectoryPoints = (self.cameraTrajectoryPoints[:-1, :]/self.cameraTrajectoryPoints[-1, :]).T
            
#             print(self.camera)

        self.worldTrajectoryDirections = np.array([self.worldTrajectoryPoints[i, :]-self.worldTrajectoryPoints[j, :] for i, j in zip(xrange(1, len(self.worldTrajectoryPoints)),
                                                                                                                                     xrange(0, len(self.worldTrajectoryPoints)-1))])
        self.worldTrajectoryDirections = np.vstack([self.worldTrajectoryDirections, self.worldTrajectoryDirections[-1, :].reshape([1, self.worldTrajectoryDirections.shape[-1]])])
        self.worldTrajectoryDirections /= np.linalg.norm(self.worldTrajectoryDirections, axis=1).reshape([len(self.worldTrajectoryDirections), 1])
        for i in xrange(len(self.worldTrajectoryDirections)) :
            if np.linalg.norm(self.worldTrajectoryDirections[i, :]) != 1.0 and i > 0 :
                self.worldTrajectoryDirections[i, :] = self.worldTrajectoryDirections[i-1, :]
        
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
        possibleTexSizes = np.array([128, 256, 512, 1024, 2048])
        texSize = possibleTexSizes[np.argwhere(possibleTexSizes-np.max(img.shape[0:2]) >= 0).flatten()[0]]
        self.tex = np.zeros([texSize, texSize, 4], np.int8)
        self.tex[:img.shape[0], :img.shape[1], :img.shape[2]] = img
        ## set alpha channel if inout image is just rgb
        if False or img.shape[2] == 3 :
            self.tex[:img.shape[0], :img.shape[1], -1] = np.int8(255)
            
        self.textureChanged = True
        [self.maxV, self.maxU] = np.array(img.shape[:2], np.float32)/np.array(self.tex.shape[:2], np.float32)
        self.aspectRatio = float(img.shape[1])/float(img.shape[0])
        
        ## sets scale of the billboard so that it compensates for the size of the texture it shows in such a way that pixels have the same size in the same viewport
        ## (without this, the scale of the billboard is fixed so that bigger textures are pasted onto the same billboard and therefore look smaller than smaller textures)
        if self.normalizeToPixelSize :
#             self.scale = self.pixelSize*img.shape[1]
            top, left, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)
#             print(top, left, width, height)
            ## the idea here is that I want the ratio of the patch width to the width of the image it comes from, to be the same as the ratio of the viewport width to the width of the same image
            ## I know that the height is always 1 and the width is defined wrt it which is why I'm computing this wrt the width
            self.scale = float(img.shape[1])/float(1280)*5.36229266*(width/1280.0)
            #self.scale = 1.0#float(img.shape[1])/float(1280)*1.0*(width/1280.0)
#             print("POOP", img.shape[0], height, height/float(img.shape[0]), self.scale, 2.0*self.scale, )
        
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


# In[10]:

# tmpLoc = "/media/ilisescu/Data1/PhD/data/theme_park_sunny/filmed_object-person2.npy"
# tmpData = np.load(tmpLoc).item()
# tmpData[DICT_TRACK_LOCATION] = "/media/ilisescu/Data1/PhD/data/theme_park_sunny/{0}-track.txt".format("person2")
# np.save(tmpLoc, tmpData)


# In[11]:

# f = open("/home/ilisescu/PhD/data/havana/{0}-track.txt".format("blue_car1"), 'r')
# lines = f.readlines()
# vals = [np.array(i.split(" ")).astype(float) for i in lines]
# vals = [(int(i[-1]), i[0:2]) for i in vals]
# tmp = dict(vals)
# patches = np.load("/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-blue_car1.npy").item()
# trajectoryPoints = np.array([tmp[key] for key in np.sort(patches.keys())])

# filmedObjectData = {DICT_FILMED_OBJECT_NAME : "blue_car1",
#                     DICT_TRACK_LOCATION : "/home/ilisescu/PhD/data/havana/{0}-track.txt".format("blue_car1"),
#                     DICT_TRAJECTORY_POINTS : trajectoryPoints,
#                     DICT_NEEDS_UNDISTORT : False,
#                     DICT_CAMERA_INTRINSICS : np.array([[702.736053, 0.0, 640.0],
#                                                        [0.0, 702.736053, 360.0],
#                                                        [0.0, 0.0, 1.0]]),
#                     DICT_PATCHES_LOCATION : "/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-blue_car1.npy",
#                     DICT_REPRESENTATIVE_COLOR : np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()[DICT_REPRESENTATIVE_COLOR]}
# np.save("/home/ilisescu/PhD/data/havana/filmed_object-blue_car1.npy", filmedObjectData)


# f = open("/home/ilisescu/PhD/data/havana/{0}-track.txt".format("red_car1"), 'r')
# lines = f.readlines()
# vals = [np.array(i.split(" ")).astype(float) for i in lines]
# vals = [(int(i[-1]), i[0:2]) for i in vals]
# tmp = dict(vals)
# patches = np.load("/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-red_car1.npy").item()
# trajectoryPoints = np.array([tmp[key] for key in np.sort(patches.keys())])

# filmedObjectData = {DICT_FILMED_OBJECT_NAME : "red_car1",
#                     DICT_TRACK_LOCATION : "/home/ilisescu/PhD/data/havana/{0}-track.txt".format("red_car1"),
#                     DICT_TRAJECTORY_POINTS : trajectoryPoints,
#                     DICT_NEEDS_UNDISTORT : False,
#                     DICT_CAMERA_INTRINSICS : np.array([[702.736053, 0.0, 640.0],
#                                                        [0.0, 702.736053, 360.0],
#                                                        [0.0, 0.0, 1.0]]),
#                     DICT_PATCHES_LOCATION : "/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-red_car1.npy",
#                     DICT_REPRESENTATIVE_COLOR : np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-red_car1.npy").item()[DICT_REPRESENTATIVE_COLOR]}
# np.save("/home/ilisescu/PhD/data/havana/filmed_object-red_car1.npy", filmedObjectData)

# f = open("/home/ilisescu/PhD/data/havana/{0}-track.txt".format("white_bus1"), 'r')
# lines = f.readlines()
# vals = [np.array(i.split(" ")).astype(float) for i in lines]
# vals = [(int(i[-1]), i[0:2]) for i in vals]
# tmp = dict(vals)
# trajectoryPoints = np.array([tmp[key] for key in np.sort(tmp.keys())])

# filmedObjectData = {DICT_FILMED_OBJECT_NAME : "white_bus1",
#                     DICT_TRACK_LOCATION : "/home/ilisescu/PhD/data/havana/{0}-track.txt".format("white_bus1"),
#                     DICT_TRAJECTORY_POINTS : trajectoryPoints,
#                     DICT_NEEDS_UNDISTORT : False,
#                     DICT_CAMERA_INTRINSICS : np.array([[702.736053, 0.0, 640.0],
#                                                        [0.0, 702.736053, 360.0],
#                                                        [0.0, 0.0, 1.0]]),
#                     DICT_PATCHES_LOCATION : "/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-red_car1.npy",
#                     DICT_REPRESENTATIVE_COLOR : np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-white_bus1.npy").item()[DICT_REPRESENTATIVE_COLOR]}
# np.save("/home/ilisescu/PhD/data/havana/filmed_object-white_bus1.npy", filmedObjectData)


# In[12]:

# f = open("/media/ilisescu/Data1/PhD/data/theme_park_sunny/{0}-track.txt".format("person2"), 'r')
# lines = f.readlines()
# vals = [np.array(i.split(" ")).astype(float) for i in lines]
# vals = [(int(i[-1]), i[0:2]) for i in vals]
# tmp = dict(vals)

# if False :
#     patches = np.load("/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-red_car1.npy").item()
#     trajectoryPoints = np.array([tmp[key] for key in np.sort(patches.keys()) if key in tmp.keys()])
# else :
#     trajectoryPoints = np.array([tmp[key] for key in np.sort(tmp.keys())])

# filmedObjectData = {DICT_FILMED_OBJECT_NAME : "person2",
#                     DICT_TRACK_LOCATION : "/media/ilisescu/Data1/PhD/data/theme_park_sunny/{0}-track.txt".format("person2"),
#                     DICT_TRAJECTORY_POINTS : trajectoryPoints,
#                     DICT_NEEDS_UNDISTORT : False,
#                     DICT_CAMERA_INTRINSICS : np.array([[1275.186144, 0.0, 480.0],
#                                                        [0.0, 1275.186144, 270.0],
#                                                        [0.0, 0.0, 1.0]]),
#                     DICT_PATCHES_LOCATION : "/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-red_car1.npy",
#                     DICT_REPRESENTATIVE_COLOR : np.load("/media/ilisescu/Data1/PhD/data/theme_park_sunny/semantic_sequence-person2.npy").item()[DICT_REPRESENTATIVE_COLOR]}
# np.save("/media/ilisescu/Data1/PhD/data/theme_park_sunny/filmed_object-person2.npy", filmedObjectData)


# In[13]:

class GLFilmedObject() :
    def __init__(self, objectLoc, cameraIntrinsics, cameraExtrinsics, isDistorted, distortionCoeff, originalIntrinsics, footprintScale=0.3, footprintAspectRatio=1.75) :
        self.initDone = False
        self.frameToUseIdx = 0
        self.footprintAspectRatio = footprintAspectRatio
        self.filmedObjectData = np.load(objectLoc).item()
        print("LOADED", self.filmedObjectData[DICT_FILMED_OBJECT_NAME])
        self.footprintScale = self.filmedObjectData[DICT_OBJECT_BILLBOARD_SCALE]*footprintScale
        self.cameraIntrinsics = cameraIntrinsics
        self.cameraExtrinsics = cameraExtrinsics
        self.previouslyUsedFrame = 0
        
        ## rotate by 180 along z axis and translate up by 1
        self.modelMat = np.array([[-1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]], np.float32)

        self.forwardDir = np.array([[1.0], [0.0], [0.0], [1.0]]) ## model space

        self.trajectoryPoints = self.filmedObjectData[DICT_TRAJECTORY_POINTS]
        if self.filmedObjectData[DICT_NEEDS_UNDISTORT] :
            self.trajectoryPoints = cv2.undistortPoints(self.trajectoryPoints.reshape((1, len(self.trajectoryPoints), 2)),
                                                        self.filmedObjectData[DICT_CAMERA_INTRINSICS], distortionCoeff, P=self.cameraIntrinsics)[0, :, :]
        else :
            self.trajectoryPoints = self.trajectoryPoints + cameraIntrinsics[:2, -1] - self.filmedObjectData[DICT_CAMERA_INTRINSICS][:2, -1]
        
        self.trajectory = GLTrajectory(self.trajectoryPoints, cameraIntrinsics, cameraExtrinsics, self.filmedObjectData[DICT_REPRESENTATIVE_COLOR], doSmoothing = False)
        
#         global tmpTrajectoryCameraSpace
#         tmpTrajectoryCameraSpace = np.copy(self.trajectory.cameraTrajectoryPoints)
        
        ## patches defined previously before getting rid of the havana hardcodes (last one is the one I was using last and likely using now too)
#         patchesLoc = "/camera_adjusted_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
#         patchesLoc = "/camera_adjusted_plus_scale_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
#         patchesLoc = "/camera_adjusted_using_billboard_homography_scale-based-on-world-billboard_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
#         patchesLoc = "/camera_adjusted_using_billboard_homography_scale-based-on-patch_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
#         patchesLoc = "/camera_adjusted_using_billboard_homography_scale-test_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
#         patchesLoc = "/camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
#         patchesLoc = "/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-{0}.npy".format(self.objectData[DICT_SEQUENCE_NAME])
        
        self.patches = np.load(self.filmedObjectData[DICT_PATCHES_LOCATION]).item()
        self.sortedPatchKeys = np.sort(self.patches.keys())
        print(self.sortedPatchKeys, len(self.sortedPatchKeys), len(self.trajectoryPoints))
        
        
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.0, np.dot(self.modelMat, np.array([[-1, 0, 0, 0],
#                                                                                                       [0, -1, 0, 0],
#                                                                                                       [0, 0, 1, 0],
#                                                                                                       [0, 0, 0, 1]], np.float32)), False, None, True)
        
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.0, self.modelMat, False, np.array([0, 0, 1], np.float32), True)
        ## this is for the latest setup before I switched to using homographies for adjusting patches
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.0, self.modelMat, True, None, True)
    
        ## this is for the patches adjusted using homographies
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 0.72, np.dot(self.modelMat, np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi, np.array([0.0, 1.0, 0.0]))),
#                                                                                                     quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2, np.array([1.0, 0.0, 0.0]))))), False, None, False)
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.52, np.dot(self.modelMat, np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi, np.array([0.0, 1.0, 0.0]))),
#                                                                                                     quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2, np.array([1.0, 0.0, 0.0]))))), False, None, False)
#         self.billboard = GLBillboard(self.getImageFromPatch(-1), 1.23, np.dot(self.modelMat, np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi, np.array([0.0, 1.0, 0.0]))),
#                                                                                                     quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2, np.array([1.0, 0.0, 0.0]))))), False, None, False)
        self.billboard = GLBillboard(self.getImageFromPatch(-1), self.filmedObjectData[DICT_OBJECT_BILLBOARD_SCALE],
                                     np.dot(self.modelMat, np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi, np.array([0.0, 1.0, 0.0]))),
                                                                  quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2, np.array([1.0, 0.0, 0.0]))))), False, None, False)
        
        
        ## move object to the first image on its original trajectory and orient properly
        if True :
            self.setObjectOnTrajectory(0)
        
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
            
            ### THIS STUFF FROM HERE ON I"M NOT REALLY SURE ABOUT
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
            if i == len(self.trajectory.worldTrajectoryDirections)-1 :
                trajDir = self.trajectory.cameraTrajectoryPoints[i-1, :]-self.trajectory.cameraTrajectoryPoints[i, :]
            else :
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
        del self.filmedObjectData, self.cameraIntrinsics, self.cameraExtrinsics, self.trajectory, self.patches, self.sortedPatchKeys
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
#         print(idx)
        img[patch['visible_indices'][:, 0], patch['visible_indices'][:, 1], :] = patch['sprite_colors']
        
#         print("PATCH SIZE", patch['patch_size'])
        
        return img[:, :, [2, 1, 0, 3]]
    
    def setObjectOnTrajectory(self, trajPointIdx) :
        self.setObjectPosAndDir(self.trajectory.worldTrajectoryPoints[trajPointIdx, :], self.trajectory.worldTrajectoryDirections[trajPointIdx, :])
            
    def setObjectPosAndDir(self, positionWorld, directionWorld) :
        objPos, objFDir = getWorldSpacePosAndNorm(self.modelMat, self.forwardDir)
        adjustAngle = np.arccos(np.clip(np.dot(objFDir, directionWorld), -1, 1))
        modelMat = np.copy(self.modelMat)
        if np.abs(adjustAngle) > 1e-06 :
#             print(adjustAngle, np.cross(directionWorld, objFDir))
            adjustAxis = np.cross(directionWorld, objFDir)
            adjustAxis /= np.linalg.norm(adjustAxis)
            modelMat = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis)), self.modelMat)
        modelMat[:-1, -1] = positionWorld
        
        self.setObjectModelMat(modelMat)
        
    def setObjectModelMat(self, modelMat) :
        self.modelMat = modelMat
#         if not self.billboard.isFrontoParallel and not self.billboard.doRotateAboutPlaneNormal :
# #             self.billboard.modelMat = np.dot(self.modelMat, np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi, np.array([0.0, 1.0, 0.0]))),
# #                                                                    quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2, np.array([1.0, 0.0, 0.0])))))
#             self.billboard.modelMat = self.modelMat
    
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
        
    def draw(self, projectionMat, viewMat, doDrawFootprint=True, doDrawTrajectory=True, doDrawColors=True, doDrawMisc=True) :
        if self.initDone :
            ## find which frame to show
            distances = viewToObjectDirAngleDistance(self, viewMat)
            tmp = self.frameToUseIdx
            self.frameToUseIdx = int(np.argmin(distances).flatten())
            if tmp != self.frameToUseIdx :
                self.previouslyUsedFrame = tmp

            top, left, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)

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
            
            ## these are actually in clip space as the camera space would only be after applying viewMat
            objPos = getWorldSpacePosAndNorm(self.modelMat, posOnly=True)
            objPosInCameraSpace = np.dot(np.dot(projectionMat, viewMat), np.concatenate([objPos, [1]]).reshape([4, 1]))
            objPosInCameraSpace = objPosInCameraSpace[:-1, 0]/objPosInCameraSpace[-1, 0]
            ## these are actually in screen space from clip space (after projection mat applied)
            objPosInClipSpace = np.array([(objPosInCameraSpace[0]+1.0)*width/2.0, (1.0-objPosInCameraSpace[1])*height/2.0])

            objDirPosInWorldSpace = np.dot(self.modelMat, self.forwardDir)
            objDirPosInWorldSpace = objDirPosInWorldSpace[:3, 0]/objDirPosInWorldSpace[3, 0]
            objDirPosInCameraSpace = np.dot(np.dot(projectionMat, viewMat), np.concatenate([objDirPosInWorldSpace, [1]]).reshape([4, 1]))
            objDirPosInCameraSpace = objDirPosInCameraSpace[:-1, 0]/objDirPosInCameraSpace[-1, 0]
            objDirPosInClipSpace = np.array([(objDirPosInCameraSpace[0]+1.0)*width/2.0, (1.0-objDirPosInCameraSpace[1])*height/2.0])

            objMovingDirectionInCameraSpace = objDirPosInCameraSpace[:-1]-objPosInCameraSpace[:-1]#objDirPosInClipSpace-objPosInClipSpace
            objMovingDirectionInCameraSpace /= np.linalg.norm(objMovingDirectionInCameraSpace)
            
            ## moving direction in camera space of requested direction for the filmed object
            objDirLine = GLPolyline(np.concatenate([np.concatenate([[objPosInCameraSpace[:-1]], [objPosInCameraSpace[:-1]+objMovingDirectionInCameraSpace]]), np.zeros([2, 1])], axis=1).astype(np.float32),
                                    drawColor=np.array([   0.,  0.,    255.]))
            objDirLine.setShaders()
            ## moving direction in camera space of the tracked object in the frame currently visualized
            movingDirLine = GLPolyline(np.concatenate([np.concatenate([[objPosInCameraSpace[:-1]], [objPosInCameraSpace[:-1]+movingDirection]]), np.zeros([2, 1])], axis=1).astype(np.float32),
                                       drawColor=np.array([   0.,  255.,    0.]))
            movingDirLine.setShaders()
            
            ## put moving direction back into its original aspect ratio
            movingDirection[0] *= (float(width)/float(height))
            movingDirection /= np.linalg.norm(movingDirection)
            objMovingDirectionInCameraSpace[0] *= (float(width)/float(height))
            objMovingDirectionInCameraSpace /= np.linalg.norm(objMovingDirectionInCameraSpace)
            
            ## THIS SOLVES (THE FUNNY PROBLEM WHERE THE CAR IS POINTING DOWNWARDS WHEN I SHOW A FRAME FROM THE LOWER LEFT CORNER) ONLY IF BILLBOARD IS FRONTOPARALLEL
            rotDir = np.cross(objMovingDirectionInCameraSpace, movingDirection)
            rotDir /= np.linalg.norm(rotDir)
            rotAngle = np.arccos(np.clip(np.dot(objMovingDirectionInCameraSpace, movingDirection), -1.0, 1.0))
#             print("ROTATION", rotAngle*180.0/np.pi, rotDir)
            

            self.billboard.setTexture(self.getImageFromPatch(self.frameToUseIdx))
            global tmpDirectionAngleDistances
            tmpDirectionAngleDistances = np.copy(distances)
    
            isDepthTestOn = bool(gl.glGetBooleanv(gl.GL_DEPTH_TEST))
            gl.glDisable(gl.GL_DEPTH_TEST)
            
            if doDrawColors :
                if self.billboard.isFrontoParallel :
                    self.billboard.draw(projectionMat, viewMat, rotDir=rotDir, rotAngle=rotAngle)
                else :
                    if not self.billboard.isFrontoParallel and not self.billboard.doRotateAboutPlaneNormal :
#                         print(self.frameToUseIdx, self.filmedObjectData[DICT_OBJECT_BILLBOARD_ORIENTATION][self.frameToUseIdx]*180/np.pi, self.modelMat)
                        self.billboard.modelMat = np.dot(self.modelMat, np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(self.filmedObjectData[DICT_OBJECT_BILLBOARD_ORIENTATION][self.frameToUseIdx], 
                                                                                                                             np.array([0.0, 0.0, 1.0]))),
                                                                               quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2, np.array([1.0, 0.0, 0.0])))))
#                         self.billboard.modelMat = np.dot(np.eye(4, dtype=np.float32), quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2, np.array([1.0, 0.0, 0.0]))))
                    
                    self.billboard.draw(projectionMat, viewMat)#, rotDir=rotDir, rotAngle=rotAngle)

#             print("(pos, posFDir, camFDir, camProjFDir)", pos, posFDir, camFDir, camProjFDir)
            
            if doDrawTrajectory :
                self.trajectory.draw(np.dot(projectionMat, viewMat))
                
            if doDrawFootprint :
                self.drawFootprint(projectionMat, viewMat)
            
            if doDrawMisc :
                ## drawing direction line and such 
                objDirLine.draw(np.eye(4, dtype=np.float32))
                movingDirLine.draw(np.eye(4, dtype=np.float32))

                objDirLine.draw(np.dot(projectionMat, viewMat))
                movingDirLine.draw(np.dot(projectionMat, viewMat))

            if isDepthTestOn :
                gl.glEnable(gl.GL_DEPTH_TEST)
                
def viewToObjectDirAngleDistance(filmedObject, viewMat, overrideObjectMat = None) :
    if np.all(overrideObjectMat != None) :
        objectMat = overrideObjectMat
    else :
        objectMat = filmedObject.modelMat
    camPos = getWorldSpacePosAndNorm(np.linalg.inv(viewMat), posOnly=True)
    objPos = getWorldSpacePosAndNorm(objectMat, posOnly=True)
            
    cameraToObjDir = objPos-camPos
    cameraToObjDir /= np.linalg.norm(cameraToObjDir)
#     cameraToObjDir = np.dot(np.linalg.inv(objectMat), np.concatenate([objPos+cameraToObjDir, [1]]).reshape([4, 1])).flatten()
#     cameraToObjDir = cameraToObjDir[:-1]/cameraToObjDir[-1]
    ## in object space from world space
    cameraPosObjSpace = np.dot(np.linalg.inv(objectMat), np.concatenate([objPos-cameraToObjDir, [1]]).reshape([4, 1])).flatten()
    cameraPosObjSpace = cameraPosObjSpace[:-1]/cameraPosObjSpace[-1]
    cameraToObjDir = np.zeros(3)-cameraPosObjSpace
    cameraToObjDir /= np.linalg.norm(cameraToObjDir)

    return np.abs(np.arccos(np.clip(np.dot(cameraToObjDir.reshape([1, 3]), filmedObject.cameraToObjectDirectionsObjSpace.T), -1.0, 1.0))*180.0/np.pi).flatten()


# In[14]:

# filmedSceneData = {DICT_FILMED_SCENE_BASE_LOC : "/home/ilisescu/PhD/data/havana/",
#                    DICT_CAMERA_EXTRINSICS : np.array([[0.820045839796, 0.57100067645, -0.0385103638868, 1.67922756789],
#                                                       [0.22275752409, -0.380450047102, -0.897572753108, -0.831720502302],
#                                                       [-0.527165918942, 0.727472328789, -0.439181175316, 6.76268742928],
#                                                       [0.0, 0.0, 0.0, 1.0]], np.float32),
#                    DICT_CAMERA_INTRINSICS : np.array([[702.736053, 0.0, 640.0],
#                                                       [0.0, 702.736053, 360.0],
#                                                       [0.0, 0.0, 1.0]]),
#                    DICT_DISTORTION_PARAMETER : -0.19,
#                    DICT_DISTORTION_RATIO : -0.19,
#                    DICT_DOWNSAMPLED_FRAMES_RATE : 4,
#                    DICT_COMMENTS : "the extrinsics are fit using a square defined by checking the vanishing line and taking into account the aspect ratio of the rectangle I fit (rather that assuming it is square)"}

# np.save("/home/ilisescu/PhD/data/havana/filmed_scene-havana.npy", filmedSceneData)


# In[15]:

# tmp = np.array([[0.820045839796, 0.57100067645, -0.0385103638868, 1.67922756789],
#                 [0.22275752409, -0.380450047102, -0.897572753108, -0.831720502302],
#                 [-0.527165918942, 0.727472328789, -0.439181175316, 6.76268742928],
#                 [0.0, 0.0, 0.0, 1.0]], np.float32)
# print(np.linalg.inv(tmp))
# tmp = np.array([[0.837692028578, 0.146962583737, 0.525998159919, -1.31690405468],
#                 [0.546141040765, -0.222903435553, -0.807492428454, 1.23665978954],
#                 [-0.00142437669339, 0.963699152952, -0.266986729541, 0.164402319713],
#                 [0.0, 0.0, 0.0, 1.0]], np.float32)
# print(np.linalg.inv(tmp))


# In[16]:

# filmedSceneData = {DICT_FILMED_SCENE_BASE_LOC : "/media/ilisescu/Data1/PhD/data/theme_park_sunny/",
#                    DICT_CAMERA_EXTRINSICS : np.array([[0.733156815131, 0.679921283997, 0.013716121742, 0.455885725879],
#                                                       [0.0618491087665, -0.0465791408423, -0.996998029779, 0.265190959012],
#                                                       [-0.677241295383, 0.73180423011, -0.0762023399981, 2.70396742177],
#                                                       [0.0, 0.0, 0.0, 1.0]], np.float32),
#                    DICT_CAMERA_INTRINSICS : np.array([[1275.186144, 0.0, 480.0],
#                                                       [0.0, 1275.186144, 270.0],
#                                                       [0.0, 0.0, 1.0]]),
#                    DICT_DISTORTION_PARAMETER : -0.3,
#                    DICT_DISTORTION_RATIO : -0.19,
#                    DICT_DOWNSAMPLED_FRAMES_RATE : 4,
#                    DICT_COMMENTS : "the extrinsics are fit using a square defined by checking the vanishing line and taking into account the aspect ratio of the rectangle I fit (rather that assuming it is square)"}

# np.save("/media/ilisescu/Data1/PhD/data/theme_park_sunny/filmed_scene-theme_park_sunny.npy", filmedSceneData)


# In[17]:

# # filmedSceneLoc = "/media/ilisescu/Data1/PhD/data/theme_park_sunny/filmed_scene-theme_park_sunny.npy"
# filmedSceneLoc = "/home/ilisescu/PhD/data/havana/filmed_scene-havana.npy"
# filmedSceneData = np.load(filmedSceneLoc).item()

# cameraExtrinsics = filmedSceneData[DICT_CAMERA_EXTRINSICS]

# cameraIntrinsics = filmedSceneData[DICT_CAMERA_INTRINSICS]
# originalIntrinsics = np.copy(cameraIntrinsics)

# medianImage = np.array(Image.open(filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+"median.png"), np.uint8)

# if DICT_DISTORTION_PARAMETER in filmedSceneData.keys() and DICT_DISTORTION_RATIO in filmedSceneData.keys() :
#     medianImage, cameraIntrinsics, distortionCoeff, map1, map2 = undistortImage(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO], medianImage, cameraIntrinsics)
    
# figure(); imshow(medianImage)
# T = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])
# worldSquareToUse = np.array([[-4, -2], [-4, 3], [1, 3], [1, -2]], float)
# cameraSquareToUse = np.dot(T, np.concatenate([worldSquareToUse, np.ones([4, 1])], axis=1).T)
# cameraSquareToUse = (cameraSquareToUse[:-1, :]/cameraSquareToUse[-1, :]).T
# scatter(cameraSquareToUse[:, 0], cameraSquareToUse[:, 1])

# texSize = 1024
# homography = cv2.findHomography(cameraSquareToUse, np.array([[0, texSize], [0, 0], [texSize, 0], [texSize, texSize]], float))[0]
# tmp = cv2.warpPerspective(medianImage, homography, (texSize, texSize))
# figure(); imshow(tmp)#; xlim([0, texSize]), ylim([0, texSize])


# In[18]:

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(medianImage)

# clickedPoints = np.empty([0, 2])

# def onclick(event):
# #     print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
# #           (event.button, event.x, event.y, event.xdata, event.ydata))
#     global clickedPoints
#     if event.xdata != None and event.ydata != None :
#         clickedPoints = np.concatenate([clickedPoints, np.array([[event.xdata, event.ydata]])])
#         cla()
#         gca().imshow(medianImage)
#         xlim([0, medianImage.shape[1]-1]); ylim([medianImage.shape[0]-1, 0])
#         gca().plot(clickedPoints[:, 0], clickedPoints[:, 1])
#         show()
#         print(clickedPoints); sys.stdout.flush()

# cid = fig.canvas.mpl_connect('button_press_event', onclick)


# In[19]:

## theme park
# cameraGroundPoints = np.array([[2.25268817, 434.08064516],
#                                [640.0625, 350.75],
#                                [1018.875, 411.9375],
#                                [1023., 574.875],
#                                [1.375, 574.875]])
# segmentsToExtrude = [0, 1, 2, 4]

## havana
# cameraGroundPoints = np.array([[17.30645161290323, 928.4677419354839],
#                                [295.85349462365593, 783.5981182795698],
#                                [125.68212365591393, 345.3790322580645],
#                                [230.23185483870964, 332.0322580645161],
#                                [342.5672043010752, 414.33736559139777],
#                                [520.5241935483871, 365.3991935483871],
#                                [479.3716397849462, 344.26680107526875],
#                                [544.9932795698925, 325.35887096774195],
#                                [887.5604838709678, 384.307123655914],
#                                [1106.6700268817206, 263.07392473118273],
#                                [1364.7076612903227, 289.76747311827955],
#                                [1531.5423387096776, 457.7143817204301],
#                                [1648.3266129032259, 468.8366935483871],
#                                [1652.7755376344087, 928.1881720430106]])
# segmentsToExtrude = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# filmedSceneData[DICT_GROUND_MESH_POINTS] = cameraGroundPoints
# filmedSceneData[DICT_GROUND_MESH_SEGS_EXTRUDE] = segmentsToExtrude
# np.save(filmedSceneLoc, filmedSceneData)


# In[20]:

# ## make texture out of medianImage
# possibleTexSizes = np.array([128, 256, 512, 1024, 2048])
# texSize = possibleTexSizes[np.argwhere(possibleTexSizes-np.max(medianImage.shape[0:2]) >= 0).flatten()[0]]
# medianTex = np.zeros([texSize, texSize, 4], np.int8)
# medianTex[:medianImage.shape[0], :medianImage.shape[1], :medianImage.shape[2]] = medianImage
# ## set alpha channel if inout image is just rgb
# medianTex[:medianImage.shape[0], :medianImage.shape[1], -1] = np.int8(255)
# [maxV, maxU] = np.array(medianImage.shape[:2], np.float32)/np.array(medianTex.shape[:2], np.float32)

# cameraGroundPoints = np.array([[2.25268817, 434.08064516],
#                                [640.0625, 350.75],
#                                [1018.875, 411.9375],
#                                [1023., 574.875],
#                                [1.375, 574.875]])
# worldGroundPoints = np.dot(np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])), np.concatenate([cameraGroundPoints, np.ones([len(cameraGroundPoints), 1])], axis=1).T)
# worldGroundPoints = np.vstack([worldGroundPoints[:-1, :]/worldGroundPoints[-1, :], np.zeros([1, len(cameraGroundPoints)])]).T
# ## now figure out vertices, indices and uvs but for this specific case --> there must be an automatic way to figure this out though
# vertices = worldGroundPoints[[0, 2, 1, 0, 3, 2, 0, 4, 3], :].astype(np.float32)
# indices = np.arange(len(vertices), dtype=np.int32)
# ## texture coords origin is bottom left which is why the stuff below does height-ycoord for all camera space points
# texCoords = (np.array([[0, medianImage.shape[0]]])+(np.array([[1, -1]])*cameraGroundPoints))/texSize
# ## the image is placed at the top of texture so the origin of the image is at (0, maxV) while texCoords assumes it's at (0, 0)
# texCoords += np.array([[0.0, maxV]])
# uvs = texCoords[[0, 2, 1, 0, 3, 2, 0, 4, 3], :].astype(np.float32)


# In[21]:

# viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, medianImage.shape[0:2])
# print(worldGroundPoints)
# print(np.dot(np.dot(projectionMat, viewMat), np.hstack([worldGroundPoints, np.ones([len(worldGroundPoints), 1])]).T)/
#       np.dot(np.dot(projectionMat, viewMat), np.hstack([worldGroundPoints, np.ones([len(worldGroundPoints), 1])]).T)[-1, :])


# In[22]:

# figure(); imshow(medianImage)
# plot(cameraGroundPoints[[3, 4, 0, 3, 2, 0, 1, 2], 0], cameraGroundPoints[[3, 4, 0, 3, 2, 0, 1, 2], 1], c="yellow")


# In[23]:

class GLProjectiveTextureMesh() :
    def __init__(self, vertices, indices, texture, projectionMVP, modelMat=np.eye(4, dtype=np.float32)) :
        self.initDone = False
        self.textureChanged = True
        
        self.modelMat = np.copy(modelMat).astype(np.float32)
        self.projectionMVP = np.copy(projectionMVP).astype(np.float32)
        self.vertices = np.copy(vertices).astype(np.float32)
        self.indices = np.copy(indices).astype(np.int32)
#         self.uvzs = np.copy(uvzs).astype(np.float32)
        self.tex = np.copy(texture).astype(np.int8)
        
        self.setGeometryAndBuffers()
        
    def __del__(self) :
        del self.tex, self.vertices, self.indices #, self.uvzs
        
    def setGeometryAndBuffers(self) :
        self.verticesBuffer = glvbo.VBO(self.vertices, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.indexBuffer = glvbo.VBO(self.indices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
#         self.uvzsBuffer = glvbo.VBO(self.uvzs, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        
    def setShaders(self) :
        self.projectiveTextureShadersProgram = compileShaders(VS_PROJECTIVE, FS_PROJECTIVE)
        if np.any(self.projectiveTextureShadersProgram == None) :
            self.initDone = False
            return
        self.initDone = True
                
        self.textureID = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
        
    def draw(self, projectionMat, viewMat) :
        if self.initDone :
            gl.glUseProgram(self.projectiveTextureShadersProgram)

            ## send mvp
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.projectiveTextureShadersProgram, "m_pvm"), 1, gl.GL_FALSE,
                                  np.dot(projectionMat, np.dot(viewMat, self.modelMat)).T)
            
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.projectiveTextureShadersProgram, "m_proj_mat"), 1, gl.GL_FALSE,
                                  self.projectionMVP.T)

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
            gl.glUniform1i(gl.glGetUniformLocation(self.projectiveTextureShadersProgram, "texture_sampler"), 0)

            ## bind the index buffer
            self.indexBuffer.bind()

            ## bind the VBO with vertex data
            self.verticesBuffer.bind()
            gl.glEnableVertexAttribArray(0)
            # tell OpenGL that the VBO contains an array of vertices
            # these vertices contain 3 single precision coordinates
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## bind the VBO with uv data
#             self.uvzsBuffer.bind()
#             gl.glEnableVertexAttribArray(1)
#             gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

            ## draw points from the VBO
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)

            gl.glUseProgram(0)


# In[24]:

# T = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])
# ## this has to be a square or else I'll get problems further down
# worldSquareToUse = np.array([[-4, -2], [-4, 3], [1, 3], [1, -2]], float)
# cameraSquareToUse = np.dot(T, np.concatenate([worldSquareToUse, np.ones([4, 1])], axis=1).T)
# cameraSquareToUse = (cameraSquareToUse[:-1, :]/cameraSquareToUse[-1, :]).T
# texSize = 2000
# ## the cameraSquare needs the y coords flipped
# ## the world square should be np.array([[0, texSize], [0, 0], [texSize, 0], [texSize, texSize]], float) but if I don't flip it vertically, the coordinate systems don't match
# homography = cv2.findHomography(cameraSquareToUse, np.array([[0, texSize], [0, 0], [texSize, 0], [texSize, texSize]], float))[0]
# tmp = cv2.warpPerspective(medianImage, homography, (texSize, texSize))
# figure(); imshow(tmp)


# In[25]:

VIDEO_PLAYBACK_FPS = 15
class GLFilmedScene() :
    def __init__(self, filmedSceneLoc, videoFPS=30, downsampleRate=4, frustumScale=0.5, pointSize=4.0) :
        self.initDone = False
        self.initFailed = False
        self.doRenderModifiedScene = True
        self.gridPointToPlaceObjectAt = -1
        self.videoFPS = videoFPS
        self.playbackFrameSkip = videoFPS/VIDEO_PLAYBACK_FPS
        self.downsampleRate = downsampleRate
        self.pointSize = np.float32(pointSize)
        self.distortionCoeff = np.zeros(5)
        self.currentFilmedObject = -1
        self.modifiedScene = None
        self.costOnGrid = None
        
        ## loading dictionary containing all necessary data
        self.filmedSceneData = np.load(filmedSceneLoc).item()
        
        self.cameraExtrinsics = self.filmedSceneData[DICT_CAMERA_EXTRINSICS]
        self.modelMat = np.linalg.inv(self.cameraExtrinsics)
        
        self.cameraIntrinsics = self.filmedSceneData[DICT_CAMERA_INTRINSICS]
        originalIntrinsics = np.copy(self.cameraIntrinsics)
        
#         self.medianImage = np.array(Image.open(self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+"frame-01499.png"), np.uint8)
        self.medianImage = np.array(Image.open(self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+"median.png"), np.uint8)
        
        if DICT_DISTORTION_PARAMETER in self.filmedSceneData.keys() and DICT_DISTORTION_RATIO in self.filmedSceneData.keys() :
            self.isDistorted = True
            self.medianImage, self.cameraIntrinsics, self.distortionCoeff, map1, map2 = undistortImage(self.filmedSceneData[DICT_DISTORTION_PARAMETER], self.filmedSceneData[DICT_DISTORTION_RATIO],
                                                                                                       self.medianImage, self.cameraIntrinsics)
        else :
            self.isDistorted = False
            
#         self.projectImageGridPoints(self.medianImage)
        
        self.filmedFramesLocs = np.sort(glob.glob(self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+"frame-*.png"))
        if len(self.filmedFramesLocs) > 0 :
            self.currentFrame = 0
            downsampledLoc = self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+"downsampledSet-"+np.string_(self.downsampleRate)+"x.npy"
            if DICT_DOWNSAMPLED_FRAMES_RATE not in self.filmedSceneData.keys() or self.filmedSceneData[DICT_DOWNSAMPLED_FRAMES_RATE] != self.downsampleRate :
                self.filmedSceneData[DICT_DOWNSAMPLED_FRAMES_RATE] = self.downsampleRate
                np.save(filmedSceneLoc, self.filmedSceneData)
                
            if os.path.isfile(downsampledLoc) :
                print("LOADING", downsampledLoc); sys.stdout.flush()
                self.allFrames = np.load(downsampledLoc)
            else :
                print("WRITING", downsampledLoc); sys.stdout.flush()
                if self.isDistorted :
                    firstImg = Image.fromarray(cv2.remap(np.array(Image.open(self.filmedFramesLocs[0])), map1, map2, cv2.INTER_LINEAR).astype(np.uint8))
                else :
                    firstImg = Image.open(self.filmedFramesLocs[0])
                firstImg.thumbnail((firstImg.width/self.downsampleRate, firstImg.height/self.downsampleRate), Image.ANTIALIAS)
                firstImg = np.array(firstImg, np.int8)
                self.allFrames = np.zeros([len(self.filmedFramesLocs), firstImg.shape[0], firstImg.shape[1], firstImg.shape[2]], np.int8)
                self.allFrames[0, :, :, :] = firstImg
                for i, imageLoc in enumerate(self.filmedFramesLocs[1:3000]) :
                    if self.isDistorted :
                        img = Image.fromarray(cv2.remap(np.array(Image.open(imageLoc)), map1, map2, cv2.INTER_LINEAR).astype(np.uint8))
                    else :
                        img = Image.open(imageLoc)
                    img.thumbnail((firstImg.shape[1], firstImg.shape[0]), Image.ANTIALIAS)
                    self.allFrames[i, :, :, :] = np.array(img, np.int8)
                np.save(downsampledLoc, self.allFrames)
#             [self.maxV, self.maxU], self.aspectRatio = self.setFrame(self.allFrames[self.currentFrame, :, :, :])
            self.aspectRatio = float(self.allFrames[self.currentFrame, :, :, :].shape[1])/float(self.allFrames[self.currentFrame, :, :, :].shape[0])
#             self.setGeometryAndBuffers()

            self.playTimer = QtCore.QTimer()
            self.playTimer.setInterval(1000/VIDEO_PLAYBACK_FPS)
            self.playTimer.timeout.connect(self.requestRender)
    #             self.playTimer.start()
             
            self.cameraFrustum = GLCameraFrustum(self.modelMat, self.allFrames[self.currentFrame, :, :, :], frustumScale)
        else :
            self.initFailed = True
            
        self.filmedObjects = []
#         for loc in np.sort(glob.glob(filmedSceneLoc+"semantic_sequence-*.npy"))[7:8] :
        for loc in np.sort(glob.glob(self.filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+"filmed_object-*.npy")) :
            self.filmedObjects.append(GLFilmedObject(loc, self.cameraIntrinsics, np.linalg.inv(self.modelMat), self.isDistorted, self.distortionCoeff, originalIntrinsics))
            
        self.showFilmedObject(0)
        
#         ## PROJECTED GROUND PLANE ONTO BILLBOARD
#         T = np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]])
#         ## this has to be a square or else I'll get problems further down
#         worldSquareToUse = np.array([[-4, -2], [-4, 3], [1, 3], [1, -2]], float)
#         cameraSquareToUse = np.dot(T, np.concatenate([worldSquareToUse, np.ones([4, 1])], axis=1).T)
#         cameraSquareToUse = (cameraSquareToUse[:-1, :]/cameraSquareToUse[-1, :]).T
#         texSize = 2000
#         ## the cameraSquare needs the y coords flipped
#         ## the world square should be np.array([[0, texSize], [0, 0], [texSize, 0], [texSize, texSize]], float) but if I don't flip it vertically, the coordinate systems don't match
#         homography = cv2.findHomography(cameraSquareToUse, np.array([[0, texSize], [0, 0], [texSize, 0], [texSize, texSize]], float))[0]
#         tmp = cv2.warpPerspective(self.medianImage, homography, (texSize, texSize))
#         position = line2lineIntersection(worldSquareToUse[[0, 2], :].flatten(), worldSquareToUse[[1, 3], :].flatten())
#         self.sceneMesh = GLBillboard(tmp, np.max(worldSquareToUse[:, 0])-np.min(worldSquareToUse[:, 0]), np.array([[1, 0, 0, position[0]], [0, 1, 0, position[1]], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
#                                      False, None, False)

        ## TEXTURED MESH FOR GROUND PLANE
#         cameraGroundPoints = np.array([[2.25268817, 434.08064516],
#                                        [640.0625, 350.75],
#                                        [1018.875, 411.9375],
#                                        [1023., 574.875],
#                                        [1.375, 574.875]])
        cameraGroundPoints = self.filmedSceneData[DICT_GROUND_MESH_POINTS]
        worldGroundPoints = np.dot(np.linalg.inv(np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]])), np.concatenate([cameraGroundPoints, np.ones([len(cameraGroundPoints), 1])], axis=1).T)
        worldGroundPoints = np.vstack([worldGroundPoints[:-1, :]/worldGroundPoints[-1, :], np.zeros([1, len(cameraGroundPoints)])]).T
        ## now triangulate the points 
        triangleIndices = np.array(triangulate2DPolygon(worldGroundPoints[:, :-1])).flatten()
        vertices = worldGroundPoints[triangleIndices, :].astype(np.float32)
        indices = np.arange(len(vertices), dtype=np.int32)
        projectionViewMat, projectionProjectionMat = cvCameraToOpenGL(self.cameraExtrinsics, self.cameraIntrinsics, self.medianImage.shape[0:2])
        
        ## extrude some worldGroundPoints up to get some walls
        segmentsToExtrude = self.filmedSceneData[DICT_GROUND_MESH_SEGS_EXTRUDE]
        segmentsToExtrudeIndices = [[segment, (segment+1) % len(worldGroundPoints)] for segment in segmentsToExtrude]
        for segment in segmentsToExtrudeIndices :
            newVertices, newIndices = extrudeSegment(worldGroundPoints[segment, :], 5.0, np.linalg.inv(self.cameraExtrinsics)[:-1, -1])
            vertices = np.concatenate([vertices, newVertices.astype(np.float32)], axis=0)
        indices = np.arange(len(vertices), dtype=np.int32)
        
        self.sceneMesh = GLProjectiveTextureMesh(vertices, indices, np.concatenate([self.medianImage, np.ones([self.medianImage.shape[0], self.medianImage.shape[1], 1])*255], axis=-1).astype(np.int8),
                                                 np.dot(projectionProjectionMat, projectionViewMat))
        
    ## not sure this actually cleans up properly
    def __del__(self) :
        del self.allFrames, self.filmedFramesLocs, self.filmedObjects, self.cameraFrustum, self.modifiedScene, self.costOnGrid, self.filmedSceneData, self.sceneMesh
        
    def showFilmedObject(self, idx) :
        if idx >= 0 and idx < len(self.filmedObjects) and idx != self.currentFilmedObject :
            self.currentFilmedObject = idx
            if np.all(self.modifiedScene != None) : 
                del self.modifiedScene
                self.modifiedScene = None
            if np.all(self.costOnGrid != None) :
                del self.costOnGrid
                self.costOnGrid = None
                
            self.modifiedScene = GLModifiedScene(self.medianImage, self.filmedObjects[self.currentFilmedObject], self.cameraExtrinsics, self.cameraIntrinsics)
            
            cameraGroundPoints = self.filmedSceneData[DICT_GROUND_MESH_POINTS]
            worldGroundPoints = np.dot(np.linalg.inv(np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]])), np.concatenate([cameraGroundPoints, np.ones([len(cameraGroundPoints), 1])], axis=1).T)
            worldGroundPoints = np.vstack([worldGroundPoints[:-1, :]/worldGroundPoints[-1, :], np.zeros([1, len(cameraGroundPoints)])]).T
            
            self.costOnGrid = GLCostOnGrid(self.filmedObjects[self.currentFilmedObject], worldGroundPlanePoints=worldGroundPoints)
            
            if self.initDone :
                self.modifiedScene.setShaders()
                self.costOnGrid.setShaders()
                
            return True
        else :
            return False
        
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

    def placeObjectOnNextCostGridPointAtBestOrientation(self, doGoForward) :
        if doGoForward :
            self.gridPointToPlaceObjectAt = np.mod(self.gridPointToPlaceObjectAt+1, len(self.costOnGrid.gridPoints))
        else :
            self.gridPointToPlaceObjectAt = self.gridPointToPlaceObjectAt-1
            if self.gridPointToPlaceObjectAt < 0 :
                self.gridPointToPlaceObjectAt = len(self.costOnGrid.gridPoints) - 1
                
        objPos, bestMatchesCost, bestOrientationMatchCost, bestOrientationMatchFrameIdx, bestOrientationModelMat = self.costOnGrid.getCostsAtGridPoint(self.gridPointToPlaceObjectAt,
                                                                                                                                                       self.filmedObjects[self.currentFilmedObject])
        self.filmedObjects[self.currentFilmedObject].setObjectModelMat(bestOrientationModelMat)
        
    def setGeometryAndBuffers(self) :
        if False and not self.initFailed :
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
        
        if np.all(self.modifiedScene != None) : 
            self.modifiedScene.setShaders()
        if np.all(self.costOnGrid != None) : 
            self.costOnGrid.setShaders()
            
        self.sceneMesh.setShaders()
        
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
#             tmp = time.time()
            self.doRequestPlayVideo(self.isPlayerLookingAtCamera(projectionMat, viewMat) and self.cameraFrustum.drawBillboard)
#             print("requestePlayVideo", time.time()-tmp)
            
#             tmp = time.time()
            self.cameraFrustum.draw(projectionMat, viewMat)
#             gl.glFinish()
#             print("cameraFrustum.draw", time.time()-tmp)
                
# #             tmp = time.time()
#             if np.all(self.costOnGrid != None) :
#                 self.costOnGrid.draw(projectionMat, viewMat)
# #             gl.glFinish()
# #             print("costOnGrid.draw", time.time()-tmp)

#             tmp = time.time()
#             self.drawProjectedImageGridPoints(projectionMat, viewMat)
            self.sceneMesh.draw(projectionMat, viewMat)
#             gl.glFinish()
#             print("drawProjectedImageGridPoints", time.time()-tmp)
                
#             for filmedObject in self.filmedObjects :
            if self.currentFilmedObject != -1 :
#                 tmp = time.time()
                self.filmedObjects[self.currentFilmedObject].draw(projectionMat, viewMat, doDrawMisc=False)
#                 gl.glFinish()
#                 print("filmedObject.draw", time.time()-tmp)
            
            if np.all(self.modifiedScene != None) and self.doRenderModifiedScene :
#                 tmp = time.time()
                self.modifiedScene.draw(projectionMat, viewMat)
#                 gl.glFinish()
#                 print("modifiedScene.draw", time.time()-tmp)


# In[26]:

class GLScene() :
    def __init__(self) :
        self.lightDirection = np.array([[1, 1, 1, 0]], np.float32).T#QtGui.QVector4D(1.0, 1.0, 1.0, 0.0)
        self.lightColor = np.array([1.0, 1.0, 1.0], np.float32)
        self.lightPower = np.float32(1.0)
        
        self.projectionMat = np.eye(4, dtype=np.float32)
        
        ## set view matrix using the qt lookat function
        self.viewMat = QtGui.QMatrix4x4()
#         cameraPos = QtGui.QVector3D(0.0, 1.0, 6.0)
        ## (cameraPos, cameraPos + direction, upVec) are in gl coords
#         self.viewMat.lookAt(cameraPos, cameraPos+QtGui.QVector3D(0, 0, -1), QtGui.QVector3D(0, 1, 0))

        cameraPos = QtGui.QVector3D(5.0, 6.0, 11.0)
        self.viewMat.lookAt(cameraPos, QtGui.QVector3D(4, 0, 2), QtGui.QVector3D(0, 1, 0))
        
#         cameraPos = QtGui.QVector3D(0.0, 20.0, 0.0)
#         self.viewMat.lookAt(cameraPos, QtGui.QVector3D(0, 0, 0), QtGui.QVector3D(1, 0, 0))
        
        
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
        self.doPlaybackObjectTrajectory = False
        self.playbackLastTime = time.time()
    
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
        
    def moveCameraInRange(self, x, y, rangeWidth = 1.0, rangeHeight = 1.0) :
        """ x and y input arguments must be in range [-1, 1] """
        
#         print("MOVING CAMERA", x, y)
        viewMat, projectionMat = cvCameraToOpenGL(self.filmedScenes[0].cameraExtrinsics, self.filmedScenes[0].cameraIntrinsics, self.filmedScenes[0].medianImage.shape[:2])
        
        self.projectionMat = projectionMat
        
        viewPos, viewUp = getWorldSpacePosAndNorm(np.linalg.pinv(viewMat), np.array([[0.0], [1.0], [0.0], [1.0]]))
        viewPos, viewRight = getWorldSpacePosAndNorm(np.linalg.pinv(viewMat), np.array([[1.0], [0.0], [0.0], [1.0]]))
        
        t = viewRight*x*rangeWidth/2.0 + viewUp*y*rangeHeight/2.0
        tMat = np.array([[1, 0, 0, t[0]],
                         [0, 1, 0, t[1]],
                         [0, 0, 1, t[2]],
                         [0, 0, 0, 1]], np.float32)
        self.viewMat = np.dot(viewMat, np.linalg.pinv(tMat))
        
        
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
    def playbackObjectAnimation(self) :
        self.currentObjectViewFrame = np.mod(self.currentObjectViewFrame+1, len(self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace))
#         self.currentObjectViewFrame = 191

        if self.doPlaybackObjectViews :
            self.playbackObjectViews()
        elif self.doPlaybackObjectTrajectory :
            self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].setObjectOnTrajectory(self.currentObjectViewFrame)
            
            objPos = getWorldSpacePosAndNorm(self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].modelMat, posOnly=True)
            objDirPosInWorldSpace = np.dot(self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].modelMat,
                                           self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].forwardDir).flatten()[:-1]
            self.filmedScenes[0].modifiedScene.updateObjectMovementIndicators(objPos, (objDirPosInWorldSpace-objPos)/np.linalg.norm(objDirPosInWorldSpace-objPos))
            
            
    def playbackObjectViews(self) :
        ## find distance from obj to current camera
        viewPos = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), posOnly=True)
        objPos = getWorldSpacePosAndNorm(self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].modelMat, posOnly=True)
        viewToObjDir = objPos-viewPos
        distanceToObj = np.linalg.norm(viewToObjDir)
        viewToObjDir /= distanceToObj

        ## desired direction in object space
        desiredViewDir = self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace[self.currentObjectViewFrame, :]
        ## direction into world space
        ## from object to world space using modelMat
        desiredViewDirPos = np.dot(self.filmedScenes[0].filmedObjects[self.filmedScenes[0].currentFilmedObject].modelMat, np.concatenate([np.zeros(3) - desiredViewDir, [1]]).reshape([4, 1])).flatten()
        desiredViewDirPos = desiredViewDirPos[:-1]/desiredViewDirPos[-1]
#         print("README", np.linalg.norm(desiredViewDir))
        desiredViewDir = objPos-desiredViewDirPos
#         print("README2", np.linalg.norm(desiredViewDir))
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
        if self.doPlaybackObjectViews or self.doPlaybackObjectTrajectory :
            if time.time() - self.playbackLastTime > 0.05 :
                self.playbackLastTime = time.time()
                self.playbackObjectAnimation()
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
#                 tmp = time.time()
                self.filmedScenes[i].draw(self.projectionMat, self.viewMat)
#                 print("all", time.time()-tmp)
#                 print("\n")
                
#             cameraPos = np.array([self.cameraPos.x(), self.cameraPos.y(), self.cameraPos.z()], np.float32)
#             cameraDist = np.float32(self.cameraPos.length())
            cameraPos = getWorldSpacePosAndNorm(np.linalg.pinv(self.viewMat), posOnly=True)
            cameraDist = np.float32(np.linalg.norm(cameraPos))
            self.axesWidget.draw(cameraDist, np.dot(self.projectionMat, self.viewMat))
        
    def loadSceneFromFile(self, sceneLoc) :
        if sceneLoc.split(".")[-1] == "npy" :
            if len(self.filmedScenes) == 1 :
                del self.filmedScenes[0]
            self.filmedScenes.append(GLFilmedScene(sceneLoc))
        elif sceneLoc.split(".")[-1] == "obj" or sceneLoc.split(".")[-1] == "ply" :
            self.addMeshesFromFile(sceneLoc)
        
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


# In[27]:

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
        
#         self.setScene("../data/suzanne.obj")
#         self.setScene("../data/havana/filmed_scene-havana.npy")
        self.setScene("/media/ilisescu/Data1/PhD/data/theme_park_sunny/filmed_scene-theme_park_sunny.npy")
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
#             tmp = time.time()
            self.scene.draw()
#             print("all scene", time.time()-tmp)
#             print("\n")
            
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
        self.doPlayControl = False
        
        self.doChangeFOV = False
        self.doChangeFrustumScale = False
        self.doChangePointSize = False
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.requestRender)
        self.lastRenderTime = time.time()
        self.playTimer.start()
        
        self.setWindowTitle("3D Looping")
        self.resize(1850, 720)
        self.glWidget.setMinimumSize(1280, 720)
#         self.glWidget.setMinimumSize(self.glWidget.scene.filmedScenes[0].medianImage.shape[1], self.glWidget.scene.filmedScenes[0].medianImage.shape[0])
#         self.glWidget.setFixedSize(self.glWidget.scene.filmedScenes[0].medianImage.shape[1], self.glWidget.scene.filmedScenes[0].medianImage.shape[0])
        
        self.setFocus()
        
    def requestRender(self) :
        currentTime = time.time()
        deltaTime = currentTime - self.lastRenderTime
        
        # self.faceDetectionWidget.trackInFrame()

        if self.doPlayControl :
            self.glWidget.scene.filmedScenes[0].modifiedScene.controlFilmedObject(self.doMoveForwards, self.doMoveSideways, deltaTime)
        else :
            if self.doMoveForwards != 0.0 or self.doMoveSideways != 0.0 :
                self.glWidget.scene.translateCamera(np.array([self.doMoveForwards*deltaTime*self.glWidget.cameraSpeed,
                                                              self.doMoveSideways*deltaTime*self.glWidget.cameraSpeed,
                                                              0.0]))
        
        if self.doMoveUpwards :
            self.glWidget.scene.translateCamera(np.array([0.0,
                                                          0.0,
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
        
#         tmp = time.time()
        self.glWidget.updateGL()
#         print("updateGL", time.time()-tmp)
        
#         tmp = time.time()
        if True :
            cameraPos = getWorldSpacePosAndNorm(np.linalg.pinv(self.glWidget.scene.viewMat), posOnly=True)
            previouslyUsedFrame = self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].previouslyUsedFrame
            usedFrameIdx = self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].frameToUseIdx
            previousCost, usedFrameCost = viewToObjectDirAngleDistance(self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject],
                                                                       self.glWidget.scene.filmedScenes[0].modifiedScene.cameraViewMat)[[previouslyUsedFrame, usedFrameIdx]]

            camPos = getWorldSpacePosAndNorm(np.linalg.inv(self.glWidget.scene.filmedScenes[0].modifiedScene.cameraViewMat), posOnly=True)
            objPos = getWorldSpacePosAndNorm(self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].modelMat, posOnly=True)
            cameraToObjDir = objPos-camPos
            cameraToObjDir /= np.linalg.norm(cameraToObjDir)
            ## in object space from world space
            cameraPosObjSpace = np.dot(np.linalg.inv(self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].modelMat),
                                       np.concatenate([objPos-cameraToObjDir, [1]]).reshape([4, 1])).flatten()
            cameraPosObjSpace = cameraPosObjSpace[:-1]/cameraPosObjSpace[-1]
            cameraToObjDir = np.zeros(3)-cameraPosObjSpace
            cameraToObjDir /= np.linalg.norm(cameraToObjDir)
            thetaCam = np.arctan2(cameraToObjDir[1], cameraToObjDir[0])*180.0/np.pi ## theta is angle on xy plane (i.e. longitude)
            phiCam = np.arccos(cameraToObjDir[2])*180.0/np.pi ## phi is vertical angle (i.e. latitude)

            thetaObjPrev = np.arctan2(self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace[previouslyUsedFrame, 1],
                                      self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace[previouslyUsedFrame, 0])*180.0/np.pi
            phiObjPrev = np.arccos(self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace[previouslyUsedFrame, 2])*180.0/np.pi

            thetaObjCurr = np.arctan2(self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace[usedFrameIdx, 1],
                                      self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace[usedFrameIdx, 0])*180.0/np.pi
            phiObjCurr = np.arccos(self.glWidget.scene.filmedScenes[0].filmedObjects[self.glWidget.scene.filmedScenes[0].currentFilmedObject].cameraToObjectDirectionsObjSpace[usedFrameIdx, 2])*180.0/np.pi
    #         print("computations", time.time()-tmp)
    #         print("\n")

            self.infoLabel.setText("{5} --- Rendering at {0} FPS, FOV: {1}; {2}; render time[ms]: {3}, using frame {4} [{6}; {7}({8})]".format(int(1.0/(deltaTime)), self.glWidget.cameraFOV,
                                                                                                                               cameraPos, (time.time()-currentTime)*1000.0, usedFrameIdx,
                                                                                                                               [" VIEW CONTROL", " OBJECT CONTROL"][bool(self.doPlayControl)],
                                                                                                                               usedFrameCost, previouslyUsedFrame, previousCost)+
                                   "\ncamToObj({0}, {1}) --- prevObj({2}, {3}) --- currObj({4}, {5})".format(thetaCam, phiCam, thetaObjPrev, phiObjPrev, thetaObjCurr, phiObjCurr)+
                                   "\nMove: Arrows/WASD --- Rise: R/F --- Roll: Q/E --- Pivot H: Z/X --- Pivot V: PageUp/Down --- FOV: V --- Frustum: U --- Point: P GoToCam: C --- "+
                                   "Show Frustum Billboard: Space ---\nSpeed: M Wheel --- Playback Obj (Trajectory) Views: (Shift) K --- Toggle Camera/Object Control: Enter --- Toggle Show Modified Scene: M --- " +
                                   "Place Object On Grid: (Shift) G --- Top Down View: T")
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
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            pressedNum = np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9))
            
            objectChanged = self.glWidget.scene.filmedScenes[0].showFilmedObject(pressedNum)
            if objectChanged :
                self.glWidget.scene.currentObjectViewFrame = -1
            
        if e.key() == QtCore.Qt.Key_C :
            newFOV = self.glWidget.scene.goToCamera()
            if newFOV != None :
                self.glWidget.cameraFOV = np.copy(newFOV)
                
        if e.key() == QtCore.Qt.Key_G :
            self.glWidget.scene.filmedScenes[0].placeObjectOnNextCostGridPointAtBestOrientation(not e.modifiers() & QtCore.Qt.Modifier.SHIFT)
        
        if e.key() == QtCore.Qt.Key_T :
            self.glWidget.scene.viewMat = QtGui.QMatrix4x4()

            cameraPos = QtGui.QVector3D(0.0, 15.0, 0.0)
            self.glWidget.scene.viewMat.lookAt(cameraPos, QtGui.QVector3D(0, 0, 0), QtGui.QVector3D(1, 0, 0))
            
            self.glWidget.scene.viewMat.rotate(-90, 1, 0, 0)
            self.glWidget.scene.viewMat.rotate(-90, 0, 0, 1)
            self.glWidget.scene.viewMat.translate(-6, 0, 0)
            self.glWidget.scene.viewMat.translate(0, 2, 0)
            self.glWidget.scene.viewMat = np.array(self.glWidget.scene.viewMat.data(), np.float32).reshape([4, 4]).T
            
                
        if e.key() == QtCore.Qt.Key_Return :
            self.doPlayControl = not self.doPlayControl
        if e.key() == QtCore.Qt.Key_M :
            self.glWidget.scene.filmedScenes[0].doRenderModifiedScene = not self.glWidget.scene.filmedScenes[0].doRenderModifiedScene
            
        if e.key() == QtCore.Qt.Key_Space :
            self.glWidget.scene.toggleShowFrustumBillboard()
        if e.key() == QtCore.Qt.Key_K :
            if e.modifiers() & QtCore.Qt.Modifier.SHIFT :
                self.glWidget.scene.doPlaybackObjectTrajectory = not self.glWidget.scene.doPlaybackObjectTrajectory
                self.glWidget.scene.doPlaybackObjectViews = False
            else :
                self.glWidget.scene.doPlaybackObjectViews = not self.glWidget.scene.doPlaybackObjectViews
                self.glWidget.scene.doPlaybackObjectTrajectory = False
                
            if self.glWidget.scene.doPlaybackObjectViews or self.glWidget.scene.doPlaybackObjectTrajectory :
                self.glWidget.scene.playbackLastTime = time.time()
        
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
        self.faceDetectionWidget.cleanup()
        self.glWidget.cleanup()
    
    def changeScene(self) :
        sceneLoc = QtGui.QFileDialog.getOpenFileName(self, "Load Scene", os.path.expanduser("~")+"/PhD/data/", "Filmed Scenes (filmed_scene*.npy);;OBJ files (*.obj);;PLY files (*.ply)")[0]
        if sceneLoc != "" :
            self.glWidget.setScene(sceneLoc)
        self.setFocus()
        
    def useHeadLight(self) :
        self.glWidget.setShaders(VS_HEAD_LIGHT, FS_HEAD_LIGHT)
    
    def useDirLight(self) :
        self.glWidget.setShaders(VS_DIR_LIGHT, FS_DIR_LIGHT)
        
    def toggleEdges(self) :
        self.doShowEdges = not self.doShowEdges
        self.glWidget.setShowEdges(self.doShowEdges)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        self.faceDetectionWidget = FaceDetectionWidget(True)
        
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
#         self.changeSceneButton.setEnabled(False)
        self.useHeadLightButton = QtGui.QPushButton("Use Head Light")
        self.useHeadLightButton.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.useHeadLightButton.setEnabled(False)
        self.useDirLightButton = QtGui.QPushButton("Use Directional Light")
        self.useDirLightButton.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.useDirLightButton.setEnabled(False)
        self.toggleEdgesButton = QtGui.QPushButton("Toggle Edges")
        self.toggleEdgesButton.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.toggleEdgesButton.setEnabled(False)
        
        
        ## SIGNALS ##
        
        self.changeSceneButton.clicked.connect(self.changeScene)
        self.useHeadLightButton.clicked.connect(self.useHeadLight)
        self.useDirLightButton.clicked.connect(self.useDirLight)
        self.toggleEdgesButton.clicked.connect(self.toggleEdges)
        self.faceDetectionWidget.bboxMoved.connect(self.glWidget.scene.moveCameraInRange)
        
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
        mainLayout.addWidget(self.faceDetectionWidget)
        
        self.setLayout(mainLayout)


# In[28]:

MAX_VELOCITY = 1.5 ## units/second
MAX_ANGULAR_VELOCITY = np.pi/4.0 ## rad/second
class GLModifiedScene() :
    def __init__(self, bgImage, filmedObject, cameraExtrinsics, cameraIntrinsics) :
        self.initDone = False
        
        self.velocity = 0.0
        self.acceleration = 2.0
        
        self.angularVelocity = 0.0
        self.angularAcceleration = np.pi*0.75
        
        self.bgImage = bgImage
        self.filmedObject = filmedObject
        self.viewport = np.array([0, 0, self.bgImage.shape[1], self.bgImage.shape[0]], np.float32)
        self.cameraExtrinsics = np.copy(cameraExtrinsics)
        self.cameraIntrinsics = np.copy(cameraIntrinsics)
        
        self.cameraViewMat, self.cameraProjectionMat = cvCameraToOpenGL(self.cameraExtrinsics, self.cameraIntrinsics, np.array(self.bgImage.shape[0:2]))
        
        
        self.screenHeightRatio = 0.7
        ## screen height in clip space is 2 so need to multiply this ratio by 2
        self.renderBillboard = GLBillboard(self.bgImage, self.screenHeightRatio*2.0, np.eye(4, dtype=np.float32), False, None, False)
        
        self.trajectory = GLTrajectory(filmedObject.trajectoryPoints, cameraIntrinsics, cameraExtrinsics, filmedObject.filmedObjectData[DICT_REPRESENTATIVE_COLOR], False)
        
        ## eventually I can use these to control the car
#         moveDirection = np.array([[-1.0, 0.0]], dtype=np.float32)
#         position = np.array([[932.0, 538.0]], dtype=np.float32)
        
#         self.moveDirectionIndicatorCameraSpace = GLTrajectory(np.concatenate([position, position+moveDirection*100.0]), cameraIntrinsics, cameraExtrinsics, doDrawProjectedPoints=False, doSmoothing=False)
#         self.moveDirectionIndicatorWorldSpace = GLTrajectory(np.concatenate([position, position+moveDirection*100.0]), cameraIntrinsics, cameraExtrinsics, doSmoothing=False)

        objPos = getWorldSpacePosAndNorm(self.filmedObject.modelMat, posOnly=True)
        objDirPosInWorldSpace = np.dot(self.filmedObject.modelMat, self.filmedObject.forwardDir).flatten()[:-1]
        self.updateObjectMovementIndicators(objPos, (objDirPosInWorldSpace-objPos)/np.linalg.norm(objDirPosInWorldSpace-objPos))
        
    def __del__(self) :
        del self.bgImage, self.renderBillboard, self.moveDirectionIndicatorCameraSpace, self.moveDirectionIndicatorWorldSpace
        
    def setShaders(self) :
        self.renderBillboard.setShaders()
        self.trajectory.setShaders()
        self.moveDirectionIndicatorCameraSpace.setShaders()
        self.moveDirectionIndicatorWorldSpace.setShaders()
        self.initDone = True
        
    def updateObjectMovementIndicators(self, positionWorld, moveDirectionWorld) :
        T = np.dot(self.cameraIntrinsics, self.cameraExtrinsics[:-1, [0, 1, 3]])
        positionCamera = np.dot(T, np.concatenate([positionWorld[:-1], [1]]).reshape([3, 1])).flatten()
        positionCamera = positionCamera[:-1]/positionCamera[-1]
        moveDirectionPosCamera = np.dot(T, np.concatenate([(positionWorld+moveDirectionWorld)[:-1], [1]]).reshape([3, 1])).flatten()
        moveDirectionPosCamera = moveDirectionPosCamera[:-1]/moveDirectionPosCamera[-1]
        moveDirectionCamera = moveDirectionPosCamera-positionCamera
        moveDirectionCamera /= np.linalg.norm(moveDirectionCamera)
        
#         print(positionCamera, moveDirectionPosCamera)
            
        self.moveDirectionIndicatorCameraSpace = GLTrajectory(np.concatenate([[positionCamera], [positionCamera+moveDirectionCamera*100.0]]), self.cameraIntrinsics, self.cameraExtrinsics,
                                                              doDrawProjectedPoints=False, doSmoothing=False)
        self.moveDirectionIndicatorWorldSpace = GLTrajectory(np.concatenate([[positionCamera], [positionCamera+moveDirectionCamera*100.0]]), self.cameraIntrinsics, self.cameraExtrinsics,
                                                             doSmoothing=False)
        if self.initDone :
            self.moveDirectionIndicatorCameraSpace.setShaders()
            self.moveDirectionIndicatorWorldSpace.setShaders()
        
        
    def controlFilmedObject(self, doAccelerate, doTurn, deltaTime) :
        ####################### change velocity based on acceleration #######################
        if doAccelerate == 1.0 :
            acceleration = self.acceleration
            ## if I want to go forwards while going backwards, accelerate faster
            if self.velocity < 0.0 :
                acceleration = self.acceleration*2

            self.velocity = np.min([MAX_VELOCITY, self.velocity + acceleration*deltaTime*doAccelerate])
        elif doAccelerate == -1.0 :
            acceleration = self.acceleration
            ## if I want to go backwards while going forwards, accelerate faster
            if self.velocity > 0.0 :
                acceleration = self.acceleration*2
            self.velocity = np.max([-MAX_VELOCITY, self.velocity + acceleration*deltaTime*doAccelerate])
        else :
            ## decrease velocity based on direction
            if self.velocity < 0.0 :
                self.velocity = np.min([0.0, self.velocity + self.acceleration*deltaTime])
            else :
                self.velocity = np.max([0.0, self.velocity - self.acceleration*deltaTime])

        ####################### change angular velocity based on angular acceleration #######################
        if doTurn == 1.0 :
            self.angularVelocity = np.min([MAX_ANGULAR_VELOCITY, self.angularVelocity + self.angularAcceleration*deltaTime*doTurn])
        elif doTurn == -1.0 :
            self.angularVelocity = np.max([-MAX_ANGULAR_VELOCITY, self.angularVelocity + self.angularAcceleration*deltaTime*doTurn])
        else :
            ## decrease angular velocity based on direction
            if self.angularVelocity < 0.0 :
                self.angularVelocity = np.min([0.0, self.angularVelocity + self.angularAcceleration*deltaTime])
            else :
                self.angularVelocity = np.max([0.0, self.angularVelocity - self.angularAcceleration*deltaTime])
        
        
        objPos = getWorldSpacePosAndNorm(self.filmedObject.modelMat, posOnly=True)
        objDirPosInWorldSpace = np.dot(self.filmedObject.modelMat, self.filmedObject.forwardDir).flatten()[:-1]
        
        angle = self.angularVelocity*deltaTime*np.abs(self.velocity)/MAX_VELOCITY
        desiredDirection = (objDirPosInWorldSpace-objPos)/np.linalg.norm(objDirPosInWorldSpace-objPos)
        if angle != 0.0 :
            if self.velocity > 0.0 :
                turnAxis = np.array([0.0, 0.0, 1.0])
            else :
                turnAxis = np.array([0.0, 0.0, -1.0])
            desiredDirection = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(angle, turnAxis)), np.concatenate([desiredDirection, [1]]).reshape([4, 1])).flatten()
            desiredDirection = desiredDirection[:-1]/desiredDirection[-1]
            desiredDirection /= np.linalg.norm(desiredDirection)
        
        if self.velocity != 0.0 :
            positionWorld = objPos+desiredDirection*self.velocity*deltaTime
            self.filmedObject.setObjectPosAndDir(positionWorld, desiredDirection)
            self.updateObjectMovementIndicators(positionWorld, desiredDirection)
        
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
            
            ## this takes a rendered GLFilmedObject after view and projection transformations and scales it and moves it to align with the renderBillboard so that if I render the GLFilmedObject using the original
            ## camera matrices then I can safely visualize it on top of the static background image
            alignToBillboardTMat = np.array([[(self.screenHeightRatio*self.bgImage.shape[1])/(self.bgImage.shape[0]*viewportAspectRatio), 0, 0, 1.0-self.screenHeightRatio*self.renderBillboard.aspectRatio/viewportAspectRatio],
                                             [0, self.screenHeightRatio, 0, 1.0-self.screenHeightRatio],
                                             [0, 0, self.screenHeightRatio, 0],
                                             [0, 0, 0, 1]], np.float32)
            self.filmedObject.draw(np.dot(alignToBillboardTMat, self.cameraProjectionMat), self.cameraViewMat, False, False, True, False)
            
            
            ## render the indicators for where the object should be (both camera and world space)
#             self.moveDirectionIndicatorCameraSpace.draw(np.dot(tMat, np.array([[self.renderBillboard.scale, 0, 0, 0],
#                                                                                [0, -self.renderBillboard.scale, 0, 0],
#                                                                                [0, 0, self.renderBillboard.scale, 0],
#                                                                                [0, 0, 0, 1]], np.float32)))
            
#             self.moveDirectionIndicatorWorldSpace.draw(np.dot(projectionMat, viewMat))
            
            
            if isDepthTestOn :
                gl.glEnable(gl.GL_DEPTH_TEST)


# In[29]:

class GLColoredSphere() :
    def __init__(self, modelMat=np.eye(4, dtype=np.float32), longitudeLinesColors=np.array([[255.0, 0, 0], [0, 255.0, 0], [0, 0, 255.0], [255.0, 0, 255.0]]), numLatitudeLines=11) :
        self.initDone = False
        
        self.modelMat = modelMat.astype(np.float32)
        self.longitudeLinesColors = longitudeLinesColors
        self.numLongitudeLines = len(self.longitudeLinesColors)+1
        self.numLatitudeLines = numLatitudeLines        
        self.setGeometryAndBuffers()
        
    def __del__(self) :
        del self.longitudeLinesColors, self.vertices, self.indices, self.colors
        
    def setShaders(self) :
        self.shaders_program = compileShaders(VS_COLOR_NO_SHADE, FS_COLOR_NO_SHADE)
        if np.any(self.shaders_program == None) :
            self.initDone = False
            return
        self.initDone = True
        
    def setGeometryAndBuffers(self) :
        u, v = np.mgrid[0:2*np.pi:complex(0, self.numLongitudeLines), 0:np.pi:complex(0, self.numLatitudeLines)]
        x=np.cos(u)*np.sin(v)
        y=np.sin(u)*np.sin(v)
        z=np.cos(v)
        self.vertices = np.array([x.T.flatten(), y.T.flatten(), z.T.flatten()]).T
        ## remove last vertex on each latitude line as it's a duplicate of the first one
        self.vertices = np.delete(self.vertices, np.arange(self.numLongitudeLines, self.numLongitudeLines*self.numLatitudeLines+1, self.numLongitudeLines)-1, axis=0)
        ## remove all vertices in the first and last tituted lines apart from one
        self.vertices = (self.vertices[self.numLongitudeLines-2:-self.numLongitudeLines+2]).astype(np.float32)
        ## build indices to triangulate the vertices of the sphere
        ## triangles for top lid of the sphere
        self.indices = np.array([np.array([0, i, j]) for i, j in zip(np.arange(1, self.numLongitudeLines),
                                                                np.concatenate([np.arange(2, self.numLongitudeLines), [1]]))]).flatten()
        ## triangles for each row apart from lids
        firstRowTriangleIndices = np.concatenate([np.array([np.array([0, self.numLongitudeLines-1, 1, 1, self.numLongitudeLines-1, self.numLongitudeLines])+1+i for i in np.arange(0, self.numLongitudeLines-2)]).flatten(),
                                                  np.array([self.numLongitudeLines-1, (self.numLongitudeLines-1)*2, 1, 1, (self.numLongitudeLines-1)*2, self.numLongitudeLines])])
        self.indices = np.concatenate([self.indices,
                                  np.array([firstRowTriangleIndices+j*(self.numLongitudeLines-1) for j in np.arange(0, self.numLatitudeLines-3)]).flatten()])
        ## triangles for bottom lid
        self.indices = np.concatenate([self.indices,
                                       np.array([np.array([len(self.vertices)-1, j, i]) for i, j in zip(np.arange(len(self.vertices)-self.numLongitudeLines, len(self.vertices)-1),
                                                                                                   np.concatenate([np.arange(len(self.vertices)-self.numLongitudeLines+1, len(self.vertices)-1),
                                                                                                                   [len(self.vertices)-self.numLongitudeLines]]))]).flatten()]).astype(np.int32)
        
        
        self.colors = np.concatenate([np.ones([1, 3]),
                                      self.longitudeLinesColors[np.arange(len(self.longitudeLinesColors)).reshape([len(self.longitudeLinesColors), 1]).repeat(self.numLatitudeLines-2, axis=1).T.flatten(), :]/255.0,
                                      np.ones([1, 3])]).astype(np.float32)
        
        self.indexBuffer = glvbo.VBO(self.indices, gl.GL_STATIC_DRAW, gl.GL_ELEMENT_ARRAY_BUFFER)
        self.verticesBuffer = glvbo.VBO(self.vertices, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        self.colorBuffer = glvbo.VBO(self.colors, gl.GL_STATIC_DRAW, gl.GL_ARRAY_BUFFER)
        
    def draw(self, projectionMat, viewMat) :
        if self.initDone :
            gl.glUseProgram(self.shaders_program)
            
            ## send mvp
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shaders_program, "m_pvm"), 1, gl.GL_FALSE, np.dot(projectionMat, np.dot(viewMat, self.modelMat)).T)
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
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)
#             gl.glDrawElements(gl.GL_POINTS, len(self.indices), gl.GL_UNSIGNED_INT, None)

            ## clean up
            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)

            gl.glUseProgram(0)
            

class GLCostOnGrid() :
    def __init__(self, filmedObject, numAngleDivisions=16, worldGroundPlanePoints=None, gridSpace=np.array([-5, 4], float), gridSpacing=float(1.0)) :
        self.initDone = False
        
        self.coloredSpheres = []
        self.cameraViewMat, _ = cvCameraToOpenGL(filmedObject.cameraExtrinsics, filmedObject.cameraIntrinsics, np.array([100.0, 100.0]))
        
        self.showCostOnGrid(filmedObject, numAngleDivisions, worldGroundPlanePoints, gridSpace, gridSpacing)
        
    def __del__(self) :
        del self.numAngleDivisions, self.worldGroundPlanePoints, self.gridSpacing, self.coloredSpheres, self.gridPoints, self.angles
        
    def showCostOnGrid(self, filmedObject, numAngleDivisions, worldGroundPlanePoints, gridSpace, gridSpacing) :
        tmp = time.time()
        if len(self.coloredSpheres) > 0 :
            self.initDone = False
            del self.coloredSpheres
            self.coloredSpheres = []
            
        self.numAngleDivisions = numAngleDivisions
        self.gridSpace = gridSpace
        self.gridSpacing = gridSpacing
        self.worldGroundPlanePoints = np.copy(worldGroundPlanePoints)
        if np.any(worldGroundPlanePoints == None) :
            self.gridPoints = np.mgrid[self.gridSpace[0]:self.gridSpace[1]:self.gridSpacing, self.gridSpace[0]:self.gridSpace[1]:self.gridSpacing]
            self.gridPoints = self.gridPoints.reshape([2, self.gridPoints.shape[1]*self.gridPoints.shape[2]]).T
        else :
            self.gridPoints = getGridPointsInPolygon2D(self.worldGroundPlanePoints[:, :-1], self.gridSpacing)
            
        self.angles = np.arange(0, np.pi*2, np.pi/self.numAngleDivisions*2)
        allPositions, allBestMatchesCost = self.computeCostOnGrid(filmedObject)
        
        for position, bestMatchesCost in zip(allPositions, allBestMatchesCost) :
            modelMat = np.array([[0.1, 0, 0, 0],
                                 [0, 0.1, 0, 0],
                                 [0, 0, 0.1, 0],
                                 [0, 0, 0, 1]], dtype=np.float32)
            modelMat[:-1, -1] = position
            self.coloredSpheres.append(GLColoredSphere(modelMat, cm.jet(bestMatchesCost, bytes=True)[:, :-1].astype(np.float32)))
            
        print("DONE in", time.time()-tmp)
    
    def computeCostOnGrid(self, filmedObject) :
        allPositions = []
        allBestMatchesCost = []
        for i, loc in enumerate(self.gridPoints) :
            
            objPos, bestMatchesCost, bestOrientationMatchCost, bestOrientationMatchFrameIdx, bestOrientationModelMat = self.getCostsAtGridPoint(i, filmedObject)
#             print(objPos, bestMatchesCost)
            allPositions.append(objPos)
            allBestMatchesCost.append(bestMatchesCost)

        allBestMatchesCost = np.array(allBestMatchesCost)
        allBestMatchesCost = np.log(allBestMatchesCost)/np.max(np.log(allBestMatchesCost))
        
        return allPositions, allBestMatchesCost
        
    def getCostsAtGridPoint(self, idx, filmedObject) :
        """ Returns position and costs of visualizing a filmedObject at a certain location on a grid and a number of different orientations """
        
        bestMatchesCost = []
        bestOrientationModelMat = np.eye(4, dtype=np.float32)
        bestOrientationMatchCost = 1e20
        bestOrientationMatchFrameIdx = 0
        for angle in self.angles :
            modelMat, orientationDirection, bestMatchFrameIdx, bestMatchCost = self.getCostAtGridPointAndAngle(idx, angle, filmedObject)
            bestMatchesCost.append(bestMatchCost)
            
            if bestMatchCost < bestOrientationMatchCost :
                bestOrientationMatchCost = bestMatchCost
                bestOrientationModelMat = np.copy(modelMat).astype(np.float32)
                bestOrientationMatchFrameIdx = bestMatchFrameIdx
            
        return modelMat[:-1, -1].flatten(), np.array(bestMatchesCost).flatten(), bestOrientationMatchCost, bestOrientationMatchFrameIdx, bestOrientationModelMat
            
    def getCostAtGridPointAndAngle(self, idx, angle, filmedObject) :
        """ Returns modelMat, orientation direction based on angle, index of best frame and best cost of visualizing a filmedObject at a certain location on a grid at given orientation """
        
        T = np.dot(filmedObject.cameraIntrinsics, filmedObject.cameraExtrinsics[:-1, [0, 1, 3]])
        
        objPos = np.array([self.gridPoints[idx, 0], self.gridPoints[idx, 1], 0.0])
        modelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(angle, np.array([0.0, 0.0, -1.0])))
        modelMat[:-1, -1] = objPos
        
        ## find object location and direction in camera space
        objectPosWorld = np.dot(modelMat, np.array([[0.0], [0.0], [0.0], [1.0]])).flatten()
        objectPosWorld = objectPosWorld[:-1]/objectPosWorld[-1]
        objectPosCamera = np.dot(T, np.concatenate([objectPosWorld[:-1], [1.0]])).flatten()
        objectPosCamera = objectPosCamera[:-1]/objectPosCamera[-1]

        objectDirPosWorld = np.dot(modelMat, filmedObject.forwardDir).flatten()
        objectDirPosWorld = objectDirPosWorld[:-1]/objectDirPosWorld[-1]
        objectDirPosCamera = np.dot(T, np.concatenate([objectDirPosWorld[:-1], [1.0]])).flatten()
        objectDirPosCamera = objectDirPosCamera[:-1]/objectDirPosCamera[-1]
        
#         distances = viewToObjectDirAngleDistance(filmedObject, filmedObject.cameraExtrinsics, modelMat)
        distances = viewToObjectDirAngleDistance(filmedObject, self.cameraViewMat, modelMat)
        return modelMat, [objectPosCamera, objectDirPosCamera], int(np.argmin(distances).flatten()), float(np.min(distances).flatten())
        
    def setShaders(self) :
        for i in xrange(len(self.coloredSpheres)) :
            self.coloredSpheres[i].setShaders()
        self.initDone = True
        
    def draw(self, projectionMat, viewMat) :
        if self.initDone :
            for i in xrange(len(self.coloredSpheres)) :
                self.coloredSpheres[i].draw(projectionMat, viewMat)


# In[30]:

class FaceDetectionWidget(QtGui.QWidget) :
    bboxMoved = QtCore.Signal(float, float)
    
    def __init__(self, doShowCapturedImage) :
        super(FaceDetectionWidget, self).__init__()
        
        self.doShowCapturedImage = doShowCapturedImage
        self.imageHeight = 80.0
        self.cropTo = np.array([80, 90])
        self.isTrackerRunning = False
        self.readyToInit = False
        self.numFramesFaceDetected = 0
        self.desiredTrackFPS = 20
        self.bbox = np.array([0, 0, 0, 0], float) ## (x, y, w, h)
        self.trackLastTime = time.time()
        
        self.createGUI()
        
        self.vc = cv2.VideoCapture(0)
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
    def cleanup(self) :
        self.vc.release()        
        
    def trackInFrame(self) :
        if time.time() - self.trackLastTime > 1.0/self.desiredTrackFPS :
            self.trackLastTime = time.time()
            rval, frame = self.vc.read()
            doRestartInit = False
            if rval :
                gray = cv2.flip(cv2.cvtColor(cv2.resize(frame, (int(self.imageHeight/frame.shape[0]*frame.shape[1]), int(self.imageHeight))), cv2.COLOR_BGR2GRAY), 1)
                cropBy = (np.array(gray.shape)-self.cropTo)/2
                gray = gray[cropBy[0]:cropBy[0]+self.cropTo[0], cropBy[1]:cropBy[1]+self.cropTo[1]]
                if self.doShowCapturedImage :
                    self.imageLabel.setImage(gray.reshape([gray.shape[0], gray.shape[1], 1]).repeat(3, axis=-1))

                if not self.isTrackerRunning :
                    if self.readyToInit :
                        ## init CMT tracker

                        self.tracker = CMT.CMT()
                        self.tracker.estimate_scale = True
                        self.tracker.estimate_rotation = False

                        self.tracker.initialise(gray, (self.bbox[0], self.bbox[1]), 
                                                      (self.bbox[0]+self.bbox[2], self.bbox[1]+self.bbox[3]), 
                                                      (self.bbox[0]+self.bbox[2], self.bbox[1]), 
                                                      (self.bbox[0], self.bbox[1]+self.bbox[3]))
                        self.imageLabel.setBBox(self.bbox, [0, 255, 0, 255])
                        self.isTrackerRunning = True
                    else :
                        ## track using face detector
                        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(self.cropTo[0]*0.4), int(self.cropTo[0]*0.4)), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
                        if len(faces) > 0 :
                            self.numFramesFaceDetected += 1
                            if self.doShowCapturedImage :
                                ## show found bbox
                                self.imageLabel.setBBox(np.array(faces[0], float), [255, 255, 0, 255])
                            else :
                                self.imageLabel.setBBox(None, None)

                            if self.numFramesFaceDetected == 1 :
                                ## set self.bbox
                                self.bbox = np.array(faces[0], float)
                            else :
                                bboxDist = np.sqrt(np.sum((np.array(faces[0], float)-self.bbox)**2))
                                ## if the bbox hasn't changed too much, the new bbox is the average of the previous and the new
                                if bboxDist < 10.0 :
                                    self.bbox = np.average(np.vstack([self.bbox.reshape([1, 4]), np.array(faces[0], float).reshape([1, 4])]), axis=0)

                                if self.numFramesFaceDetected == 30 :
                                    self.readyToInit = True
                        else :
                            ## if face hasn't been found, need to restart initialization
                            doRestartInit = True
                else :
                    self.tracker.process_frame(gray)
                    # Draw updated estimate
                    if self.tracker.has_result:
                        ## update bbox
                        self.bbox = np.array([self.tracker.tl[0], self.tracker.tl[1], self.tracker.br[0]-self.tracker.tl[0], self.tracker.br[1]-self.tracker.tl[1]], float)
    #                     self.bbox = np.array([-30, -30, 60, 60], float)
                        self.imageLabel.setBBox(self.bbox, [0, 255, 0, 255])

                        centerPoint = np.array([gray.shape[1]/2.0, gray.shape[0]/2.0])
                        boxCenter = self.bbox[0:2]+self.bbox[2:]/2.0
                        moveTo = (boxCenter-centerPoint)/centerPoint
                        self.bboxMoved.emit(moveTo[0], -moveTo[1]) ## need a - for the y coord because right now I'm using opencv image coords and I want to use opengl ones
    #                     print(moveTo)
            else :
                ## if frame is not available from webcam, need to restart initialization
                doRestartInit = True

            if doRestartInit :
                self.imageLabel.setBBox(None, None)
                self.numFramesFaceDetected = 0
                self.isTrackerRunning = False
                self.readyToInit = False

    def createGUI(self) :
        
        self.imageLabel = ImageLabel("Video Capture")
        self.imageLabel.setFixedSize(self.imageHeight, self.imageHeight)
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.imageLabel)
        
        self.setLayout(mainLayout)


# In[52]:

window = Window()
window.show()
app.exec_()


# In[ ]:

def getGridPairIndices(width, height) :
## deal with pixels that have East and South neighbours i.e. all of them apart from last column and last row
    pairIdxs = np.zeros(((width*height-(width+height-1))*2, 2), dtype=int)
## each column contains idxs [0, h-2]
    idxs = np.arange(0, height-1, dtype=int).reshape((height-1, 1)).repeat(width-1, axis=-1)
## each column contains idxs [0, h-2]+h*i where i is the column index 
## (i.e. now I have indices of all nodes in the grid apart from last col and row)
    idxs += (np.arange(0, width-1)*height).reshape((1, width-1)).repeat(height-1, axis=0)
    # figure(); imshow(idxs)
## now flatten idxs and repeat once so that I have the idx for each node that has E and S neighbours twice
    idxs = np.ndarray.flatten(idxs.T).repeat(2)
## idxs for each "left" node (that is connected to the edge) are the ones just computed
    pairIdxs[:, 0] = idxs
## idxs for each "right" node are to the E and S so need to sum "left" idx to height and to 1
# print np.ndarray.flatten(np.array([[patchSize[0]], [1]]).repeat(np.prod(patchSize)-(np.sum(patchSize)-1), axis=-1).T)
    pairIdxs[:, 1] = idxs + np.ndarray.flatten(np.array([[height], [1]]).repeat(width*height-(width+height-1), axis=-1).T)
    
## deal with pixels that have only East neighbours
## get "left" nodes
    leftNodes = np.arange(height-1, width*height-1, height)
## now connect "left" nodes to the nodes to their East (i.e. sum to height) and add them to the list of pair indices
    pairIdxs = np.concatenate((pairIdxs, np.array([leftNodes, leftNodes+height]).T), axis=0)
    
## deal with pixels that have only South neighbours
## get "top" nodes
    topNodes = np.arange(width*height-height, width*height-1)
## now connect "to" nodes to the nodes to their South (i.e. sum to 1) and add them to the list of pair indices
    pairIdxs = np.concatenate((pairIdxs, np.array([topNodes, topNodes+1]).T), axis=0)
    
    return pairIdxs

def backgroundCut(bgImage, image, k1=30.0/255.0, k2=60.0/255.0, K=5.0/255.0, sigmaZ=10.0/255.0) :
    """ Given an image and a static background bgImage, it computes fg/bg segmentation
    
    based on BGcut [Sun et al. ECCV2006] with modifications seen in Video Synposis [Pritch et al. PAMI2008]"""
    ## as seen in Sun's background cut (with the mods made in pritch synopsis paper)
#     figure("bgImage"); imshow(bgImage); figure("image"); imshow(image)
    
    if np.all(bgImage.shape != image.shape) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
    
    height, width, channels = bgImage.shape
    maxCost = 10000000.0#np.sys.float_info.max
    
    bgPixels = bgImage.reshape([height*width, channels], order='F')/255.0
    imagePixels = image.reshape([height*width, channels], order='F')/255.0
    
    s = time.time()
    ## build graph
    numLabels = 2
    numNodes = height*width
    gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels)
    
    
    ############################### COMPUTE UNARIES ###############################
    unaries = np.zeros((numNodes,numLabels))
    
    dr = np.sqrt(np.sum((imagePixels-bgPixels)**2.0, axis=-1))
    
    unaries[dr<=k1, 1] = (k1-dr)[dr<=k1]
    unaries[dr>k2, 0] = maxCost
    unaries[np.all(np.array([dr>k1, k2>dr]), axis=0), 0] = (dr-k1)[np.all(np.array([dr>k1, k2>dr]), axis=0)]
        
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, np.arange(0, numNodes, 1))
    
    
    ############################### COMPUTE PAIRWISE ###############################
    pairIndices = getGridPairIndices(width, height)
    
    pairwise = np.zeros(len(pairIndices))
    
    zrs = np.max([np.sqrt(np.sum((imagePixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 0], :])**2.0, axis=-1)),
                  np.sqrt(np.sum((imagePixels[pairIndices[:, 1], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))], axis=0)
    
    imPixelsDiff = np.sqrt(np.sum((imagePixels[pairIndices[:, 0], :]-imagePixels[pairIndices[:, 1], :])**2.0, axis=-1))
    bgPixelsDiff = np.sqrt(np.sum((bgPixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))
    drs = imPixelsDiff/(1+((bgPixelsDiff/K)**2.0)*np.exp(-(zrs**2)/sigmaZ))
    beta = 2.0/np.mean(imPixelsDiff)
    pairwise = np.exp(-beta*drs)
    
    ## visualize
    if False :
        contrastMap = np.zeros(len(bgPixels))
        for i in np.arange((width-1)*(height-1)*2) :
            contrastMap[pairIndices[i, 0]] += drs[i]
        figure(); imshow(np.reshape(np.sqrt(np.copy(contrastMap)), [height, width], 'F'))
    
    # add functions
    fids = gm.addFunctions(np.array([[0.0, 1.0],[1.0, 0.0]]).reshape((1, 2, 2)).repeat(len(pairwise), axis=0)*
                           pairwise.reshape((len(pairwise), 1, 1)).repeat(2, axis=1).repeat(2, axis=2))
    
    # add second order factors
    gm.addFactors(fids, pairIndices)
    
    print(gm)
    
    
    graphCut = opengm.inference.GraphCut(gm=gm)
    graphCut.infer()    
    labels = np.array(graphCut.arg(), dtype=int)
    reshapedLabels = np.reshape(np.copy(labels), [height, width], 'F')
    
    return reshapedLabels

def backgroundCut3D(bgImage, images, k1=30.0/255.0, k2=60.0/255.0, K=5.0/255.0, sigmaZ=10.0/255.0) :
    """ Given a stack of temporally sequential images and a static background bgImage, it computes temporally consistent fg/bg segmentation
    
    based on BGcut [Sun et al. ECCV2006] with modifications seen in Video Synposis [Pritch et al. PAMI2008]"""
    ## as seen in Sun's background cut (with the mods made in pritch synopsis paper)
#     figure("bgImage"); imshow(bgImage); figure("image"); imshow(image)
    
    if np.all(bgImage.shape != images.shape[:-1]) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
    
    height, width, channels, numImages = images.shape
    maxCost = 10000000.0#np.sys.float_info.max
    
    bgPixels = bgImage.reshape([height*width, channels], order='F')/255.0
    
    s = time.time()
    ## build graph
    numLabels = 2
    gm = opengm.gm(np.ones(height*width*numImages,dtype=opengm.label_type)*numLabels)
    
    for i in np.arange(numImages) :
        imagePixels1 = images[:, :, :, i].reshape([height*width, channels], order='F')/255.0


        ############################### COMPUTE UNARIES ###############################
        unaries = np.zeros((height*width,numLabels))

        dr = np.sqrt(np.sum((imagePixels1-bgPixels)**2.0, axis=-1))

        unaries[dr<=k1, 1] = (k1-dr)[dr<=k1]
        unaries[dr>k2, 0] = maxCost
        unaries[np.all(np.array([dr>k1, k2>dr]), axis=0), 0] = (dr-k1)[np.all(np.array([dr>k1, k2>dr]), axis=0)]

        # add functions
        fids = gm.addFunctions(unaries)
        # add first order factors
        gm.addFactors(fids, np.arange(i*height*width, (i+1)*height*width, 1))


        ############################### COMPUTE PAIRWISE ###############################
        for j in np.arange(2) :
            if j == 0 or (i > 0 and j ==1) :
                pairIndices = getGridPairIndices(width, height)

                imagePixels2 = imagePixels1
                if i > 0 and j == 1 :
                    ## in this case compute pairwise between temporally neighbouring pixels in current image and previous one
                    pairIndices = np.concatenate([[np.arange(width*height)], [np.arange(width*height)]]).T
                    imagePixels2 = images[:, :, :, i-1].reshape([height*width, channels], order='F')/255.0

                pairwise = np.zeros(len(pairIndices))

                zrs = np.max([np.sqrt(np.sum((imagePixels2[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 0], :])**2.0, axis=-1)),
                              np.sqrt(np.sum((imagePixels1[pairIndices[:, 1], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))], axis=0)

                imPixelsDiff = np.sqrt(np.sum((imagePixels2[pairIndices[:, 0], :]-imagePixels1[pairIndices[:, 1], :])**2.0, axis=-1))
                bgPixelsDiff = np.sqrt(np.sum((bgPixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))
                drs = imPixelsDiff/(1+((bgPixelsDiff/K)**2.0)*np.exp(-(zrs**2)/sigmaZ))
                beta = 2.0/np.mean(imPixelsDiff)
                pairwise = np.exp(-beta*drs)

                ## visualize
                if False :
                    contrastMap = np.zeros(len(bgPixels))
                    for i in np.arange((width-1)*(height-1)*2) :
                        contrastMap[pairIndices[i, 0]] += drs[i]
                    figure(); imshow(np.reshape(np.sqrt(np.copy(contrastMap)), [height, width], 'F'))

                # add functions
                fids = gm.addFunctions(np.array([[0.0, 1.0],[1.0, 0.0]]).reshape((1, 2, 2)).repeat(len(pairwise), axis=0)*
                                       pairwise.reshape((len(pairwise), 1, 1)).repeat(2, axis=1).repeat(2, axis=2))

                if j == 0 :
                    ## in this case compute pairwise between neighbouring pixels in the current image
                    # add second order factors
                    gm.addFactors(fids, pairIndices+(i*height*width))
                elif i > 0 and j == 1 :
                    ## in this case compute pairwise between temporally neighbouring pixels in current image and previous one
                    pairIndices[:, 0] += ((i-1)*height*width)
                    pairIndices[:, 1] += (i*height*width)
                    gm.addFactors(fids, pairIndices)
    
    print(gm)
    
    
    graphCut = opengm.inference.GraphCut(gm=gm)
    graphCut.infer()    
    labels = np.array(graphCut.arg(), dtype=int)
    reshapedLabels = np.reshape(np.copy(labels), [height, width, numImages], 'F')
    
    return reshapedLabels
    
# resizeMultiplier = 1.0

# fgMask = backgroundCut(cv2.resize(bgImage, (0, 0), fx=resizeMultiplier, fy=resizeMultiplier, interpolation=cv2.INTER_AREA),
#                        cv2.resize(im, (0, 0), fx=resizeMultiplier, fy=resizeMultiplier, interpolation=cv2.INTER_AREA))

# fgMask2 = backgroundCut3D(cv2.resize(bgImage, (0, 0), fx=resizeMultiplier, fy=resizeMultiplier, interpolation=cv2.INTER_AREA),
#                           cv2.resize(im, (0, 0), fx=resizeMultiplier, fy=resizeMultiplier, interpolation=cv2.INTER_AREA).reshape([im.shape[0], im.shape[1], im.shape[2], 1]))

# # fgMask = cv2.morphologyEx(fgMask.astype(float), cv2.MORPH_OPEN, np.ones((5,5),np.uint8), iterations=1)
# # fgMask = cv2.morphologyEx(fgMask.astype(float), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)*255
# figure(); imshow(cv2.resize(fgMask.astype(np.uint8).reshape([fgMask.shape[0], fgMask.shape[1], 1]), (bgImage.shape[1], bgImage.shape[0]), interpolation=cv2.INTER_CUBIC))

# for i in xrange(fgMask2.shape[-1]) :
#     figure(); imshow(cv2.resize(fgMask2[:, :, i].astype(np.uint8).reshape([fgMask2[:, :, i].shape[0], 
#                                                                            fgMask2[:, :, i].shape[1], 1]), (bgImage.shape[1], bgImage.shape[0]), interpolation=cv2.INTER_CUBIC))


# In[32]:

def smoothLabels(bgImage, image, segmentation, prevLabels, k1=30.0/255.0, k2=60.0/255.0, K=5.0/255.0, sigmaZ=10.0/255.0) :
    ## as seen in Sun's background cut (with the mods made in pritch synopsis paper)
#     figure("bgImage"); imshow(bgImage); figure("image"); imshow(image)
    
    if np.all(bgImage.shape != image.shape) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
    
    height, width, channels = bgImage.shape
    maxCost = 10000000.0#np.sys.float_info.max
    
    bgPixels = bgImage.reshape([height*width, channels], order='F')/255.0
    imagePixels = image.reshape([height*width, channels], order='F')/255.0
    
    labelIds = np.sort(list(set(prevLabels.flatten())))
#     print(len(labelIds))
    
#     print(len(list(set(prevLabels.flatten()))))
#     figure(); imshow(prevLabels)
    
    s = time.time()
    ## build graph
    numLabels = len(labelIds)
    numNodes = height*width
    gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels)
    ## num of labels should be the number of different blobs and I should combine that cost with either the original unaries or with the segmentation I had before or a combination of the two
    ## then the pairwise should probaly be the same as before as that reduces the cost of cutting around edges of the foreground objects but probably a measure simply based on gradients would 
    ## do too as the unaries should constrain the cuts to be around the foreground object...
    
    ############################### COMPUTE UNARIES ###############################
    unaries = np.zeros((numNodes,numLabels))
    
#     dr = np.sqrt(np.sum((imagePixels-bgPixels)**2.0, axis=-1))
    
#     unaries[dr<=k1, 1] = (k1-dr)[dr<=k1]
#     unaries[dr>k2, 0] = maxCost
#     unaries[np.all(np.array([dr>k1, k2>dr]), axis=0), 0] = (dr-k1)[np.all(np.array([dr>k1, k2>dr]), axis=0)]

    ## unaries for the background
    unaries[:, 0] = segmentation.reshape([height*width], order='F')*numLabels
#     figure(); imshow(unaries[:, 0].reshape([height, width], order='F'))
    for i in np.arange(1, numLabels) :
        unaries[:, i] = prevLabels.reshape([height*width], order='F') != labelIds[i]
#         figure(); imshow(unaries[:, i].reshape([height, width], order='F'))
        
    
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, np.arange(0, numNodes, 1))
    
    
    ############################### COMPUTE PAIRWISE ###############################
    pairIndices = getGridPairIndices(width, height)
    
    pairwise = np.zeros(len(pairIndices))
    
    zrs = np.max([np.sqrt(np.sum((imagePixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 0], :])**2.0, axis=-1)),
                  np.sqrt(np.sum((imagePixels[pairIndices[:, 1], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))], axis=0)
    
    imPixelsDiff = np.sqrt(np.sum((imagePixels[pairIndices[:, 0], :]-imagePixels[pairIndices[:, 1], :])**2.0, axis=-1))
    bgPixelsDiff = np.sqrt(np.sum((bgPixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))
    drs = imPixelsDiff/(1+((bgPixelsDiff/K)**2.0)*np.exp(-(zrs**2)/sigmaZ))
    beta = 2.0/np.mean(imPixelsDiff)
    pairwise = np.exp(-beta*drs)
    
    ## visualize
    if False :
        contrastMap = np.zeros(len(bgPixels))
        for i in np.arange((width-1)*(height-1)*2) :
            contrastMap[pairIndices[i, 0]] += pairwise[i]
        figure(); imshow(np.reshape(np.sqrt(np.copy(contrastMap)), [height, width], 'F'))
    
    # add functions
    maxPairwiseAtOnce = 100000
    
    for i in np.arange(0, len(pairwise), maxPairwiseAtOnce) :
#         print((1.0-np.eye(numLabels)).reshape((1, numLabels, numLabels)).repeat(len(pairwise[i:i+maxPairwiseAtOnce]), axis=0).shape)
        fids = gm.addFunctions((1.0-np.eye(numLabels)).reshape((1, numLabels, numLabels)).repeat(len(pairwise[i:i+maxPairwiseAtOnce]), axis=0)*
                               pairwise[i:i+maxPairwiseAtOnce].reshape((len(pairwise[i:i+maxPairwiseAtOnce]), 1, 1)).repeat(numLabels, axis=1).repeat(numLabels, axis=2))
    
        # add second order factors
        gm.addFactors(fids, pairIndices[i:i+maxPairwiseAtOnce, :])
    
#     print(gm)
    
    
    graphCut = opengm.inference.TrwsExternal(gm=gm)
    graphCut.infer()
    labels = np.array(graphCut.arg(), dtype=int)
    
    ## set labels back to original labels
    tmp = np.zeros_like(labels)
    for i in np.arange(1, numLabels) :
        tmp[labels == i] = labelIds[i]
    
    reshapedLabels = np.reshape(np.copy(tmp), [height, width], 'F')
    
    return reshapedLabels

# newLabels = smoothLabels(bgImage[375:540, 1035:1275, :], ims[375:540, 1035:1275, :, 10], fgMasks[375:540, 1035:1275, 10], masksLabels[375:540, 1035:1275, 0])
# newLabels = smoothLabels(bgImage[265:540, 965:1275, :], ims[265:540, 965:1275, :, 10], fgMasks[265:540, 965:1275, 10], masksLabels[265:540, 965:1275, 0])
# newLabels = smoothLabels(bgImage, ims[:, :, :, 10], fgMasks[:, :, 10], masksLabels[:, :, 0])


# In[33]:

close("all")
dataLoc = "/home/ilisescu/PhD/data/havana/"

resizeMultiplier = 1.0

bgImage = cv2.resize(np.array(Image.open(dataLoc+"median.png")), (0, 0), fx=resizeMultiplier, fy=resizeMultiplier, interpolation=cv2.INTER_AREA)
figure(); imshow(bgImage)

frameLocs = np.sort(glob.glob(dataLoc+"frame-*.png"))[2310:2590]
numNeighboringFrames = 2 ## add numNeighboringFrames/2 before and numNeighboringFrames/2 after

if True :
    ## fgMasks gets loaded later when removing blobs
    fgMasks = np.load(dataLoc+"segmentation-2311to2590.npy").astype(np.uint8)
    pass
else :
    fgMasks = np.zeros([bgImage.shape[0], bgImage.shape[1], len(frameLocs)-numNeighboringFrames], dtype=np.uint8)
    print(len(frameLocs), fgMasks.shape)
    for maskIdx, frameIdx in enumerate(np.arange(numNeighboringFrames/2, len(frameLocs)-numNeighboringFrames/2)) :
    # for maskIdx, frameIdx in enumerate(np.arange(numNeighboringFrames/2, 10-numNeighboringFrames/2)) :
        ims = np.zeros([bgImage.shape[0], bgImage.shape[1], bgImage.shape[2], numNeighboringFrames+1], dtype=np.uint8)
        print(maskIdx, frameIdx, ims.shape)

        ## load the images
        for idx, i in enumerate(np.arange(frameIdx-numNeighboringFrames/2, frameIdx+1+numNeighboringFrames/2)) :
            im = np.array(Image.open(frameLocs[i]))
            ims[:, :, :, idx] = cv2.resize(im, (0, 0), fx=resizeMultiplier, fy=resizeMultiplier, interpolation=cv2.INTER_AREA)

        ## compute mask
        fgMasks[:, :, maskIdx] = backgroundCut3D(bgImage, ims)[:, :, numNeighboringFrames/2]
        fgMasks[:, :, maskIdx] = cv2.morphologyEx(fgMasks[:, :, maskIdx].astype(float), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
        sys.stdout.flush()
    #         figure(); imshow(im)
    #         diff = np.sqrt(np.sum((im/255.0-bgImage/255.0)**2, axis=-1))
    #         figure(); imshow(diff)


# In[272]:

## remove blobs smaller than threshold
blobMinArea= 35
fgMasks = np.load(dataLoc+"segmentation-2311to2590.npy").astype(np.uint8)
for i in np.arange(fgMasks.shape[-1]) :
    labelling = measure.label(fgMasks[:, :, i])
    for region in measure.regionprops(labelling) :
        if len(np.argwhere(labelling == region["label"])) < blobMinArea :
            fgMasks[labelling == region["label"], i] = 0
    #         print(region["label"],region["area"])


# In[273]:

figure()
img = None
for i in xrange(fgMasks.shape[-1]):
    if img is None:
        img = mpl.pylab.imshow(fgMasks[:, :, i])
    else:
        img.set_data(fgMasks[:, :, i])
    mpl.pylab.pause(0.01)
    mpl.pylab.draw()


# In[35]:

## this would happen after I computed the temporally consistent masks and used close on them
# fgMasksSubset = fgMasks[:, :, 0:]
# imsSubset = ims[:, :, :, 0:]
startFrame = 0
# fgMasks = fgMasks.astype(int)
masksLabels = np.zeros_like(fgMasks).astype(np.uint16)
masksLabels[:, :, startFrame] = measure.label(fgMasks[:, :, startFrame].astype(bool))
for i in np.arange(startFrame+1, fgMasks.shape[-1]) :
# for i in np.arange(startFrame+1, startFrame+5) :
    currentImage = np.array(Image.open(frameLocs[i+numNeighboringFrames/2]))
    currentImage = cv2.resize(currentImage, (0, 0), fx=resizeMultiplier, fy=resizeMultiplier, interpolation=cv2.INTER_AREA)
    ## do the labelling for each frame i and i-1
    labels, maxLabel = measure.label(fgMasks[:, :, i-1:i+1].astype(bool), return_num=True)
    ## match the labels between i-1 and the original i-1 labels and store mapping
    ## if the newLabel happens more than once in the second column, it means two blobs have merged as two old labels have been mapped to a single new one
#     figure(); imshow(np.copy(masksLabels[:, :, i-1]))
#     figure(); imshow(np.copy(labels[:, :, 0]))
    labelsMap = np.array(list(set([(oldLabel, newLabel) for oldLabel, newLabel in zip(masksLabels[fgMasks[:, :, i-1] != 0, i-1], labels[fgMasks[:, :, i-1] != 0, 0])])))
    oldToNewLabelsMap = labelsMap[np.argsort(labelsMap[:, 0]).flatten(), :]
    newToOldLabelsMap = labelsMap[np.argsort(labelsMap[:, 1]).flatten(), :][:, ::-1]
    
#     print(maxLabel, len(list(set(labels[fgMasks[:, :, i-1] != 0, 0]))))
    
    mappingNewToOldLabels = np.zeros([maxLabel+1, 2], labelsMap.dtype)
    mappingNewToOldLabels[:, 0] = np.arange(maxLabel+1)
    
    
    ## find blobs that are in frame i but not in i-1 and assign them new ids starting from the max of masksLabels so that every new blob gets recorded
    newBlobIdsOnlyInCurrent = np.setdiff1d(list(set(labels[fgMasks[:, :, i] != 0, 1])), list(set(labels[fgMasks[:, :, i-1] != 0, 0])))
    mappingNewToOldLabels[newBlobIdsOnlyInCurrent, 1] = np.arange(np.max(masksLabels)+1, np.max(masksLabels)+len(newBlobIdsOnlyInCurrent)+1)
    
    ## find blobs that are in frame i-1 but not in i and set them to the background id (i.e. 0)
    newBlobIdsOnlyInPrevious = np.setdiff1d(list(set(labels[fgMasks[:, :, i-1] != 0, 0])), list(set(labels[fgMasks[:, :, i] != 0, 1])))
    mappingNewToOldLabels[newBlobIdsOnlyInPrevious, 1] = 0
    
    ## new ids of blobs that used to be separated and are now merged: assign them to max_label_so_far+1
    newMergedBlobFromMultipleOldOnes = np.sort(oldToNewLabelsMap[:, 1])
    newMergedBlobFromMultipleOldOnes = list(set(newMergedBlobFromMultipleOldOnes[np.argwhere(newMergedBlobFromMultipleOldOnes[:-1]-newMergedBlobFromMultipleOldOnes[1:] == 0).flatten()]))
    mappingNewToOldLabels[newMergedBlobFromMultipleOldOnes, 1] = np.max(mappingNewToOldLabels)+1
    
    listOfMergedOldBlobs = [oldToNewLabelsMap[np.argwhere(oldToNewLabelsMap[:, 1] == blobId).flatten(), 0] for blobId in newMergedBlobFromMultipleOldOnes]
    print("old blobs merged to new blob [X, Y]", [[tmp[0], tmp[1]] for tmp in zip(listOfMergedOldBlobs, newMergedBlobFromMultipleOldOnes)])
    
    
    ## new ids of blobs that used to be merged and are now separated
    newSeparateBlobsFromOneMergedOld = np.sort(newToOldLabelsMap[:, 1])
    newSeparateBlobsFromOneMergedOld = list(set(newSeparateBlobsFromOneMergedOld[np.argwhere(newSeparateBlobsFromOneMergedOld[:-1]-newSeparateBlobsFromOneMergedOld[1:] == 0).flatten()]))
    print("newly separated blobs", newSeparateBlobsFromOneMergedOld)
    
#     print(newBlobIdsOnlyInCurrent, newBlobIdsOnlyInPrevious, newMergedBlobFromMultipleOldOnes, newSeparateBlobsFromOneMergedOld)
    
    
#     print(mappingNewToOldLabels)
    
    ## assign the remaining blob ids to corresponding old labels
    unchangedNewBlobIdToOldId = np.array(list(set([(pair[0], pair[1]) for pair in newToOldLabelsMap if pair[0] not in newMergedBlobFromMultipleOldOnes])))
    mappingNewToOldLabels[unchangedNewBlobIdToOldId[:, 0], 1] = unchangedNewBlobIdToOldId[:, 1]
    
#     print(mappingNewToOldLabels)
    masksLabels[fgMasks[:, :, i] != 0, i] = mappingNewToOldLabels[labels[fgMasks[:, :, i] != 0, 1], 1]
    
#     figure(); imshow(masksLabels[:, :, i])
    
    ################################## USE SMOOTH LABELS TO ENSURE MERGED BLOBS GET IDS FROM PREVIOUS LABELING ##################################
    ## for every blob that came from merging old blobs do the 
    blobProperties = measure.regionprops(labels[:, :, 1])
    bboxBorder = 20
    for blob in blobProperties :
        if blob["label"] in newMergedBlobFromMultipleOldOnes :
            ## enlarging the bbox and checking forout of bounds coords
            bbox = np.min(np.vstack([[np.array(blob["bbox"])+np.array([-bboxBorder, -bboxBorder, bboxBorder, bboxBorder])],
                                     np.array([[labels.shape[0], labels.shape[1], labels.shape[0], labels.shape[1]]])]), axis=0)
            minRow, minCol, maxRow, maxCol = np.max(np.vstack([[bbox], np.zeros([1, 4], int)]), axis=0)
            
            masksLabels[minRow:maxRow, minCol:maxCol, i] = smoothLabels(bgImage[minRow:maxRow, minCol:maxCol, :], currentImage[minRow:maxRow, minCol:maxCol, :],
                                                                        labels[minRow:maxRow, minCol:maxCol, 1], masksLabels[minRow:maxRow, minCol:maxCol, i-1])
            
#             print(blob["label"], np.argwhere(masksLabels[:, :, i] == 1))
    
    ## then do something
#     figure(); imshow(masksLabels[:, :, i])
#     print(newBlobsFromMerge)

    
########## CHECK THAT WHAT IS PRINTED MAKE SENSE ##########
## to check if "old blobs merged to new blob" is correct, check if the blobs in X used to be separate in figure a and are now merged in blob Y figure b
## to check if "newly separated blobs" is correct, check if the printed blob is merged in figure a and is now separate in figure b
figure("a"); imshow(masksLabels[:, :, -2])
figure("b"); imshow(labels[:, :, 0])


# In[36]:

blobIdToShow = 42
imgsToShow = np.array(masksLabels == blobIdToShow, dtype=masksLabels.dtype)
figure()
img = None
for i in np.arange(np.min(np.argwhere(imgsToShow == 1)[:, -1]), np.max(np.argwhere(imgsToShow == 1)[:, -1])+1):
    if img is None:
        img = mpl.pylab.imshow(imgsToShow[:, :, i])
    else:
        img.set_data(imgsToShow[:, :, i])
    mpl.pylab.pause(0.01)
    mpl.pylab.draw()


# In[264]:

## here doing the same as findContours but using skimage instead of opencv
# from skimage import measure
# tmp = measure.find_contours(undistortedFgMask/255.0, 0.99)
# for i in xrange(len(tmp)) :
#     tmp[i] = tmp[i][:, ::-1].reshape([tmp[i].shape[0], 1, 2]).astype(np.int32)
    
# tmp2 = np.ones_like(undistortedIm).astype(np.uint8)*255

# for idx, cnt in enumerate(tmp) :
#     cv2.drawContours(tmp2, [cnt], 0, (idx, idx, idx), 1)
# figure(); imshow(tmp2)


# In[37]:

filmedSceneData = np.load(dataLoc+"filmed_scene-havana.npy").item()

## overrride so I only use one object but I should find which contour belongs to each object automatically evenutally like they find actors in video synopsis
contourIdxs = [18, 18, 15, 16, 16, 19, 17, 17, 15, 15, 17, 17, 17, 19, 20]
worldFootprints = {}
worldContours = {}

for frameIdx in np.arange(20, 250) : #np.arange(masksLabels.shape[-1]) :
    
    ################################ RETRIEVE THE FRAME AND ITS SEGMENTATION AND UNDISTORT ################################
    currentImage = np.array(Image.open(frameLocs[frameIdx+numNeighboringFrames/2]))
    currentMaskLabels = cv2.resize(masksLabels[:, :, frameIdx].astype(float).reshape([masksLabels.shape[0], masksLabels.shape[1], 1]),
                                   (np.round(bgImage.shape[1]/resizeMultiplier).astype(int), np.round(bgImage.shape[0]/resizeMultiplier).astype(int)), interpolation=cv2.INTER_CUBIC).astype(float)
#     undistortedIm, cameraIntrinsics, distortionCoeff, map1, map2 = undistortImage(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO], im, filmedSceneData[DICT_CAMERA_INTRINSICS])
    
    for blobId in [42] : #list(set(currentMaskLabels[currentMaskLabels != 0])) :
        if blobId not in worldFootprints :
            worldFootprints[blobId] = {}
        if blobId not in worldContours :
            worldContours[blobId] = {}
        
        ## REMOVE THIS IF STATEMENT ONCE I DO THIS FOR ALL BLOBIDS IN THE CURRENT MASK LABELS 
        if blobId in currentMaskLabels.flatten() :
            
            ################################ GET FOREGROUND MASK FOR CURRENT BLOB ID ################################
            fgMask = np.array(currentMaskLabels == blobId, dtype=np.uint8)*255

            undistortedFgMask, cameraIntrinsics, distortionCoeff, map1, map2 = undistortImage(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO], fgMask, filmedSceneData[DICT_CAMERA_INTRINSICS])
            undistortedFgMask[undistortedFgMask >= 128] = 255
            undistortedFgMask[undistortedFgMask < 128] = 0
            
#             figure(); imshow(undistortedFgMask)
            
#         gridPoints = np.indices(undistortedFgMask.shape[0:2][::-1]).reshape([2, np.prod(undistortedFgMask.shape[0:2])]).T
#         inverseT = np.linalg.inv(np.dot(cameraIntrinsics, filmedSceneData[DICT_CAMERA_EXTRINSICS][:-1, [0, 1, 3]]))
#         worldFgPoints = np.dot(inverseT, np.concatenate([gridPoints[undistortedFgMask[gridPoints[:, 1], gridPoints[:, 0]] > 0],
#                                                          np.ones([len(gridPoints[undistortedFgMask[gridPoints[:, 1], gridPoints[:, 0]] > 0]), 1], float)], axis=1).T)
#         worldFgPoints /= worldFgPoints[-1, :]
#         worldFgPoints[-1, :] = 0
#         worldFgPoints = worldFgPoints.T
#         # print(gridPoints.shape, worldFgPoints.shape)
#         figure(); scatter(worldFgPoints[:, 0], worldFgPoints[:, 1])
#         scatter(np.linalg.inv(filmedSceneData[DICT_CAMERA_EXTRINSICS])[0, -1], np.linalg.inv(filmedSceneData[DICT_CAMERA_EXTRINSICS])[1, -1], color="red")

#         ############################### FIND CONNECTED COMPONENTS ################################
#         contours, hierarchy = cv2.findContours(np.copy(undistortedFgMask).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         renderedContours = np.ones_like(undistortedIm).astype(np.uint8)*255

#         for idx, cnt in enumerate(contours) :
#             cv2.drawContours(renderedContours, [cnt], 0, (idx, idx, idx), 1)
#         figure(); imshow(undistortedIm)
#         figure(); imshow(renderedContours)

            ############################### FIND CONTOURS ################################
            contours, hierarchy = cv2.findContours(np.copy(undistortedFgMask).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#         ################################ FIND WHICH POINTS IN EACH CONTOUR ARE ON THE GROUND PLANE AND WHICH ARE NOT ################################

#         ## project contour of blob onto the ground plane and check intersection of line to camera with it
#         inverseT = np.linalg.inv(np.dot(cameraIntrinsics, filmedSceneData[DICT_CAMERA_EXTRINSICS][:-1, [0, 1, 3]]))
#         ## I should do this nex bit for each frame of each contour/object but I don't have that working at the minute
#         contourIdx = contourIdxs[frameIdx]
#         if False :
#             ## use the renderedContours but then I lose the ordering
#             gridPoints = np.indices(renderedContours.shape[0:2][::-1]).reshape([2, np.prod(renderedContours.shape[0:2])]).T
#             worldContourPoints = np.dot(inverseT, np.concatenate([gridPoints[renderedContours[gridPoints[:, 1], gridPoints[:, 0], 0] == contourIdx],
#                                                                   np.ones([len(gridPoints[renderedContours[gridPoints[:, 1], gridPoints[:, 0], 0] == contourIdx]), 1], float)], axis=1).T)
#         else :
#             ## use the contours found by findContours: if the approximation is simple, then I can simply draw lines between the points, otherwise, each point in contours[idx] is a pixel
#             ## might as well use simple approximation since findContours works on a binary mask so the contour is always pixellated and made up of lines AND it makes everything faster
#             worldContourPoints = np.dot(inverseT, np.concatenate([contours[contourIdx][:, 0, :],
#                                                                   np.ones([len(contours[contourIdx]), 1], float)], axis=1).T)


            ################################ PROJECT ALL CONTOURS FOUND FROM THE FOREGROUND MASK ONTO THE GROUND PLANE ################################
            ## project contour of blob onto the ground plane and check intersection of line to camera with it
            inverseT = np.linalg.inv(np.dot(cameraIntrinsics, filmedSceneData[DICT_CAMERA_EXTRINSICS][:-1, [0, 1, 3]]))
            ## each blob might have more than one contour
            worldContourPoints = []
            for contour in contours :
                worldContourPoints.append(np.dot(inverseT, np.concatenate([contour[:, 0, :], np.ones([len(contour), 1], float)], axis=1).T))
                
                worldContourPoints[-1] /= worldContourPoints[-1][-1, :]
                worldContourPoints[-1][-1, :] = 0
                worldContourPoints[-1] = worldContourPoints[-1].T
#             print(len(worldContourPoints), worldContourPoints[0].shape)
#             figure(); plot(worldContourPoints[0][:, 0], worldContourPoints[0][:, 1])
            
            worldCameraPos = np.linalg.inv(filmedSceneData[DICT_CAMERA_EXTRINSICS])[:-1, -1]

            ################################ FIND WHICH POINTS IN EACH CONTOUR ARE ON THE GROUND PLANE AND WHICH ARE NOT ################################
            worldAllContourPoints = np.vstack(worldContourPoints)
            cameraAllContourPoints = np.vstack(contours)[:, 0, :].astype(float)
            if len(worldAllContourPoints) != len(cameraAllContourPoints) :
                raise Exception("Something's wrong")
            isPointFootprint = np.ones(len(worldAllContourPoints), dtype=bool)
            
            if True :
                for idx, point in enumerate(cameraAllContourPoints) :
                    bottomPoint = np.array([point[0], undistortedFgMask.shape[0]])
                    pointToBottom = bottomPoint - point
                    pointToBottom /= np.linalg.norm(pointToBottom)
                    doBreak = False
                    for contourIdx in np.arange(len(contours)) :
                        for segment in np.concatenate([[np.arange(0, len(contours[contourIdx]))], [np.mod(np.arange(0, len(contours[contourIdx]))+1, len(contours[contourIdx]))]]).T :
                            if idx not in segment :
                                try :
                                    intersectionPoint = line2lineIntersection(np.concatenate([point, bottomPoint]), np.concatenate([contours[contourIdx][segment[0], 0, :].astype(float), 
                                                                                                                                    contours[contourIdx][segment[1], 0, :].astype(float)]))
                                except Exception as e:
#                                     print(e, bottomPoint, point, contours[contourIdx][segment[0], 0, :].astype(float), contours[contourIdx][segment[1], 0, :].astype(float))
                                    pass

                                if isABetweenBandC(intersectionPoint, contours[contourIdx][segment[0], 0, :].astype(float), contours[contourIdx][segment[1], 0, :].astype(float)) :
                                    pointToIntersection = intersectionPoint - point
                                    pointToIntersection /= np.linalg.norm(pointToIntersection)
                                    if not np.linalg.norm(pointToBottom + pointToIntersection) < 1e-10 :
                                        isPointFootprint[idx] = False
#                                         print(idx, bottomPoint, point, contours[contourIdx][segment[0], 0, :].astype(float), contours[contourIdx][segment[1], 0, :].astype(float), segment, intersectionPoint)
                                        ## finding one segment the ray intersects is enough to know this is not a footprint point so need to break out 
                                        ## of both this loop and the outer one looping through the multiple found contours
                                        doBreak = True
                                        break
                        if doBreak :
                            break
            else :
                for idx, point in enumerate(worldAllContourPoints[:, :-1]) :
                    pointToCamera = worldCameraPos[:-1] - point
                    pointToCamera /= np.linalg.norm(pointToCamera)
                    doBreak = False
                    for contourIdx in np.arange(len(worldContourPoints)) :
                        for segment in np.concatenate([[np.arange(0, len(worldContourPoints[contourIdx]))], [np.mod(np.arange(0, len(worldContourPoints[contourIdx]))+1, len(worldContourPoints[contourIdx]))]]).T :
                            if idx not in segment :
                                try :
                                    intersectionPoint = line2lineIntersection(np.concatenate([point, worldCameraPos[:-1]]), np.concatenate([worldContourPoints[contourIdx][segment[0], :-1], 
                                                                                                                                            worldContourPoints[contourIdx][segment[1], :-1]]))
                                except Exception as e:
                                    print(e)
                                    pass

                                if isABetweenBandC(intersectionPoint, worldContourPoints[contourIdx][segment[0], :-1], worldContourPoints[contourIdx][segment[1], :-1]) :
                                    pointToIntersection = intersectionPoint - point
                                    pointToIntersection /= np.linalg.norm(pointToIntersection)
                                    if not np.linalg.norm(pointToCamera + pointToIntersection) < 1e-10 :
                                        isPointFootprint[idx] = False
                                        ## finding one segment the ray intersects is enough to know this is not a footprint point so need to break out 
                                        ## of both this loop and the outer one looping through the multiple found contours
                                        doBreak = True
                                        break
                        if doBreak :
                            break
                        
#             figure(); scatter(worldAllContourPoints[:, 0], worldAllContourPoints[:, 1], color="red")
#             scatter(worldAllContourPoints[isPointFootprint, 0], worldAllContourPoints[isPointFootprint, 1], color="blue")
            
#         figure(); scatter(worldContourPoints[np.negative(isPointFootprint), 0], worldContourPoints[np.negative(isPointFootprint), 1], color="blue")
#         scatter(worldContourPoints[isPointFootprint, 0], worldContourPoints[isPointFootprint, 1], color="green")
#         scatter(worldCameraPos[0], worldCameraPos[1], color="red")

            ################################ SAVE THE PROJECTED FOOTPRINT AND BLOB CONTOURS FOR CURRENT BLOB ID AT THE CURRENT FRAME ################################
            worldFootprints[blobId][frameIdx] = worldAllContourPoints[isPointFootprint, :]
            worldContours[blobId][frameIdx] = worldAllContourPoints
#         figure(); imshow(renderedContours)
#         xlim([0, renderedContours.shape[1]])
#         ylim([renderedContours.shape[0], 0])
#         plot(contours[contourIdx][isPointFootprint, 0, 0], contours[contourIdx][isPointFootprint, 0, 1], color="green")


# In[38]:

for blobId in np.sort(worldFootprints.keys()) :
    frameIds = np.sort(worldFootprints[blobId].keys())
    numFootprints = len(frameIds)
    print(numFootprints)
    ## find transformation T of each frame's footprint points to the next frame
    Ts = []
    ## find transformation T of each frame's footprint points to the last frame
    TsToLast = []
    for pointsPair in zip(np.arange(0, numFootprints-1), np.arange(1, numFootprints)) :
        ## transformation from current frame to next
        Ts.append(icp(worldFootprints[blobId][frameIds[pointsPair[0]]][:, :-1].T, worldFootprints[blobId][frameIds[pointsPair[1]]][:, :-1].T))
        ## for the current frame just use the found T
        TsToLast.append(np.array(Ts[-1][1]))
        ## now update the previous frames' TsToLast
        startTime = time.time()
        for i in np.arange(0, len(TsToLast)-1) :
            ## get T to the current frame's footprint
            TsToLast[i] = np.dot(TsToLast[-1], TsToLast[i])

            ## transform the footprint to the current frame's footprint
            tmpFootprint = np.dot(TsToLast[i], np.vstack([worldFootprints[blobId][frameIds[i]][:, :-1].T, np.ones([1, len(worldFootprints[blobId][frameIds[i]])])]))
            tmpFootprint = tmpFootprint[:-1, :]/tmpFootprint[-1, :]

            ## refine transformation using icp
            TsToLast[i] = np.dot(np.array(icp(tmpFootprint, worldFootprints[blobId][frameIds[pointsPair[1]]][:, :-1].T)[1]), TsToLast[i])
            
        print(pointsPair, time.time()-startTime)
        sys.stdout.flush()
#             print(i)

Ts = [np.array(t[1]) for t in Ts]


# In[ ]:

# np.save(dataLoc+"Ts-blob42-2311to2590.npy", [np.array(t[1]) for t in Ts])
# np.save(dataLoc+"TsToLast-blob42-2311to2590.npy", TsToLast)
# np.save(dataLoc+"worldFootprints-blob42-2311to2590.npy", worldFootprints)
# np.save(dataLoc+"worldContours-blob42-2311to2590.npy", worldContours)
# Ts = np.load(dataLoc+"Ts-blob42-2311to2590.npy")
# TsToLast = np.load(dataLoc+"TsToLast-blob42-2311to2590.npy")
# worldFootprints = np.load(dataLoc+"worldFootprints-blob42-2311to2590.npy").item()
# worldContours = np.load(dataLoc+"worldContours-blob42-2311to2590.npy").item()


# In[40]:

numFootprintsToShow = len(Ts)
### SHOW THE ALIGNED FOOTPRINTS ###
figure()
cols = ["red", "green", "blue", "cyan", "magenta"]
blobId = 42
frameIds = np.sort(worldFootprints[blobId].keys())
cameraPos = np.linalg.inv(filmedSceneData[DICT_CAMERA_EXTRINSICS])[:-1, -1]
for i in np.arange(numFootprintsToShow)[::25] :
    plot(worldContours[blobId][frameIds[i]][:, 0], worldContours[blobId][frameIds[i]][:, 1], color=cols[np.mod(i, len(cols))])
    scatter(worldFootprints[blobId][frameIds[i]][:, 0], worldFootprints[blobId][frameIds[i]][:, 1], color=cols[np.mod(i, len(cols))])
    for j in np.arange(len(worldFootprints[blobId][frameIds[i]])) :
        plot([worldFootprints[blobId][frameIds[i]][j, 0], cameraPos[0]], [worldFootprints[blobId][frameIds[i]][j, 1], cameraPos[1]])
scatter(cameraPos[0], cameraPos[1])
# xlim([-1, 1])
# ylim([-3.7, -1.7])

### TRANSFORMS ACCUMULATED OVER TIME ###
figure()
plot(worldContours[blobId][frameIds[numFootprintsToShow]][:, 0], worldContours[blobId][frameIds[numFootprintsToShow]][:, 1], color=cols[np.mod(numFootprintsToShow, len(cols))])
scatter(worldFootprints[blobId][frameIds[numFootprintsToShow]][:, 0], worldFootprints[blobId][frameIds[numFootprintsToShow]][:, 1], color=cols[np.mod(numFootprintsToShow, len(cols))])
currentTransform = np.eye(3)
for i in np.arange(numFootprintsToShow)[::-1] :
#     currentTransform = np.dot(currentTransform, np.array(Ts[i][1]))
    currentTransform = np.dot(currentTransform, np.array(Ts[i]))
    transformedFootprint = np.dot(currentTransform, np.vstack([worldFootprints[blobId][frameIds[i]][:, :-1].T, np.ones([1, len(worldFootprints[blobId][frameIds[i]])])]))
    transformedFootprint = transformedFootprint[:-1, :]/transformedFootprint[-1, :]
    
    transformedContours = np.dot(currentTransform, np.vstack([worldContours[blobId][frameIds[i]][:, :-1].T, np.ones([1, len(worldContours[blobId][frameIds[i]])])]))
    transformedContours = transformedContours[:-1, :]/transformedContours[-1, :]
    plot(transformedContours[0, :], transformedContours[1, :], color=cols[np.mod(i, len(cols))])
    scatter(transformedFootprint[0, :], transformedFootprint[1, :], color=cols[np.mod(i, len(cols))])
    
    print(i, transformedFootprint.shape)
    
# xlim([-1, 1])
# ylim([-3.7, -1.7])
xlim([-1, 1])
ylim([-5, -7])

### ACCUMULATED TRANSFORMS ARE REFINED USING ICP AGAIN ###
figure()
plot(worldContours[blobId][frameIds[numFootprintsToShow]][:, 0], worldContours[blobId][frameIds[numFootprintsToShow]][:, 1], color=cols[np.mod(numFootprintsToShow, len(cols))])
scatter(worldFootprints[blobId][frameIds[numFootprintsToShow]][:, 0], worldFootprints[blobId][frameIds[numFootprintsToShow]][:, 1], color=cols[np.mod(numFootprintsToShow, len(cols))])
currentTransform = np.eye(3)
for i in np.arange(numFootprintsToShow)[::-1] :
    currentTransform = TsToLast[i]
    transformedFootprint = np.dot(currentTransform, np.vstack([worldFootprints[blobId][frameIds[i]][:, :-1].T, np.ones([1, len(worldFootprints[blobId][frameIds[i]])])]))
    transformedFootprint = transformedFootprint[:-1, :]/transformedFootprint[-1, :]
    
    transformedContours = np.dot(currentTransform, np.vstack([worldContours[blobId][frameIds[i]][:, :-1].T, np.ones([1, len(worldContours[blobId][frameIds[i]])])]))
    transformedContours = transformedContours[:-1, :]/transformedContours[-1, :]
    plot(transformedContours[0, :], transformedContours[1, :], color=cols[np.mod(i, len(cols))])
    scatter(transformedFootprint[0, :], transformedFootprint[1, :], color=cols[np.mod(i, len(cols))])
    
    print(i, transformedFootprint.shape)
    
# xlim([-1, 1])
# ylim([-3.7, -1.7])
xlim([-1, 1])
ylim([-5, -7])

### (JUST FOOTPRINT POINTS) ACCUMULATED TRANSFORMS ARE REFINED USING ICP AGAIN ###
figure()
plot(worldContours[blobId][frameIds[numFootprintsToShow]][:, 0], worldContours[blobId][frameIds[numFootprintsToShow]][:, 1], color=cols[np.mod(numFootprintsToShow, len(cols))])
scatter(worldFootprints[blobId][frameIds[numFootprintsToShow]][:, 0], worldFootprints[blobId][frameIds[numFootprintsToShow]][:, 1], color=cols[np.mod(numFootprintsToShow, len(cols))])
currentTransform = np.eye(3)
for i in np.arange(numFootprintsToShow)[::-1] :
    currentTransform = TsToLast[i]
    transformedFootprint = np.dot(currentTransform, np.vstack([worldFootprints[blobId][frameIds[i]][:, :-1].T, np.ones([1, len(worldFootprints[blobId][frameIds[i]])])]))
    transformedFootprint = transformedFootprint[:-1, :]/transformedFootprint[-1, :]
    plot(transformedFootprint[0, :], transformedFootprint[1, :], color=cols[np.mod(i, len(cols))])
    
    print(i, transformedFootprint.shape)
    
# xlim([-1, 1])
# ylim([-3.7, -1.7])
xlim([-1, 1])
ylim([-5, -7])


# In[41]:

alignedFootprintsAllPoints = np.empty([0, 2])
for i in np.arange(len(TsToLast)) :
    transformedFootprint = np.dot(TsToLast[i], np.vstack([worldFootprints[blobId][frameIds[i]][:, :-1].T, np.ones([1, len(worldFootprints[blobId][frameIds[i]])])]))
    transformedFootprint = transformedFootprint[:-1, :]/transformedFootprint[-1, :]
    alignedFootprintsAllPoints = np.concatenate([alignedFootprintsAllPoints, transformedFootprint.T], axis=0)

alignedFootprintsAllPoints = np.concatenate([alignedFootprintsAllPoints, worldFootprints[blobId][frameIds[-1]][:, :-1]], axis=0)

## fit shape using lines
figure();
scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], color="green")
currentInliers = []
for i in xrange(5) :
    currentPointsMask = np.ones(len(alignedFootprintsAllPoints), bool)
    currentPointsMask[currentInliers] = False
    currentOutliers = np.arange(len(alignedFootprintsAllPoints), dtype=int)
    currentOutliers = currentOutliers[currentPointsMask]
    print(len(currentInliers), len(currentOutliers))
    fittedModel = linear_model.RANSACRegressor(linear_model.LinearRegression())
    fittedModel.fit(alignedFootprintsAllPoints[currentPointsMask, 0][:, np.newaxis], alignedFootprintsAllPoints[currentPointsMask, 1])
    
    lineX = np.arange(-1, 2)
    lineY = fittedModel.predict(lineX[:, np.newaxis])
    
    currentInliers = currentOutliers[fittedModel.inlier_mask_]
    plot(lineX, lineY)
# scatter(alignedFootprintsAllPoints[fittedModel.inlier_mask_, 0], alignedFootprintsAllPoints[fittedModel.inlier_mask_, 1], color="green")
# scatter(alignedFootprintsAllPoints[np.logical_not(fittedModel.inlier_mask_), 0], alignedFootprintsAllPoints[np.logical_not(fittedModel.inlier_mask_), 1], color="red")
xlim([-1, 1])
ylim([-5, -7])

# inlier_mask = model_ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
# line_X = np.arange(-5, 5)
# line_y = model.predict(line_X[:, np.newaxis])
# line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])


# In[42]:

## compute distances from each point to each of the others (consider regularly sampling the footprintPointsBBox to reduce amount of calculations)
footprintPointsBBox = np.vstack([[np.min(alignedFootprintsAllPoints, axis=0)], [np.min(alignedFootprintsAllPoints[:, 0]), np.max(alignedFootprintsAllPoints[:, 1])],
                                 [np.max(alignedFootprintsAllPoints, axis=0)], [np.max(alignedFootprintsAllPoints[:, 0]), np.min(alignedFootprintsAllPoints[:, 1])]])
footprintPointsBBoxWidth = np.max(footprintPointsBBox[:, 0]) - np.min(footprintPointsBBox[:, 0])
footprintPointsBBoxHeight = np.max(footprintPointsBBox[:, 1]) - np.min(footprintPointsBBox[:, 1])

footprintPointsDistMat = ssd(alignedFootprintsAllPoints)
footprintPointsDistMat[footprintPointsDistMat>0] = np.sqrt(footprintPointsDistMat[footprintPointsDistMat>0])
## sum of distances to every other point for each point
footprintPointsAllDistsSum = np.sum(footprintPointsDistMat, axis=0)
## number of points within a certain threshold
areaFraction = 100
thresh = np.sqrt(footprintPointsBBoxWidth*footprintPointsBBoxHeight/(areaFraction*np.pi)) ## thresh is radius of circle having a fraction of the area of the footprintPointsBBox
footprintPointsNumCloseNeighs = np.sum((footprintPointsDistMat < thresh).astype(float), axis=0)/np.sum((footprintPointsDistMat < thresh*2).astype(float), axis=0)

# figure(); imshow(footprintPointsDistMat)


# In[43]:

## found here http://stackoverflow.com/questions/6652671/efficient-method-of-calculating-density-of-irregularly-spaced-points
def grid_density_gaussian_filter(x0, y0, x1, y1, w, h, data):
    kx = (w - 1) / (x1 - x0)
    ky = (h - 1) / (y1 - y0)
    r = 20
    border = r
    imgw = (w + 2 * border)
    imgh = (h + 2 * border)
    img = np.zeros((imgh,imgw))
    for x, y in data:
        ix = int((x - x0) * kx) + border
        iy = int((y - y0) * ky) + border
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy][ix] += 1
    return spimg.gaussian_filter(img, (r,r))  ## gaussian convolution

x0, y0, x1, y1 = np.min(alignedFootprintsAllPoints[:, 0]), np.min(alignedFootprintsAllPoints[:, 1]), np.max(alignedFootprintsAllPoints[:, 0]), np.max(alignedFootprintsAllPoints[:, 1])
densityImg = grid_density_gaussian_filter(x0, y0, x1, y1, 512, np.round(512*(y1-y0)/(x1-x0)).astype(int), alignedFootprintsAllPoints)
kx = (densityImg.shape[1] - 1) / (x1 - x0)
ky = (densityImg.shape[0] - 1) / (y1 - y0)
figure(); imshow(densityImg)
# scatter((alignedFootprintsAllPoints[:, 0] - x0) * kx, (alignedFootprintsAllPoints[:, 1] - y0) * ky, marker="o", facecolors='none', s=80, edgecolors=[0, 0, 0, 0.2])


# In[44]:

nonMaxSuppressedImg = np.zeros_like(densityImg)

for x in np.arange(1, densityImg.shape[1]-1) :
    for y in np.arange(1, densityImg.shape[0]-1) :
        pixelIdx = np.array([x, y], int)

        dirs = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]], int)
        perpendicularDirsIdxs = np.array([2, 3, 0, 1], int)
        gradients = (densityImg[pixelIdx[1]+dirs[:, 1], pixelIdx[0]+dirs[:, 0]]-densityImg[pixelIdx[1]-dirs[:, 1], pixelIdx[0]-dirs[:, 0]])/np.linalg.norm((pixelIdx+dirs)-(pixelIdx-dirs), axis=1)
        maxGradientDirIdx = np.argmax(np.abs(gradients))
        # figure(); imshow(densityImg); scatter(x, y)
        # plot([pixelIdx[0]-dirs[maxGradientDirIdx, 0], pixelIdx[0]+dirs[maxGradientDirIdx, 0]], [pixelIdx[1]-dirs[maxGradientDirIdx, 1], pixelIdx[1]+dirs[maxGradientDirIdx, 1]])
        negativePerpendicularPixelIdx = pixelIdx-dirs[perpendicularDirsIdxs[maxGradientDirIdx], :]
        positivePerpendicularPixelIdx = pixelIdx+dirs[perpendicularDirsIdxs[maxGradientDirIdx], :]

        if (densityImg[pixelIdx[1], pixelIdx[0]] - densityImg[negativePerpendicularPixelIdx[1], negativePerpendicularPixelIdx[0]] > 1e-10 and
            densityImg[pixelIdx[1], pixelIdx[0]] - densityImg[positivePerpendicularPixelIdx[1], positivePerpendicularPixelIdx[0]] > 1e-10) :
            nonMaxSuppressedImg[y, x] = 1
            
figure(); imshow(nonMaxSuppressedImg)


# In[45]:

densitiesAtFootprintPoints = densityImg[np.round((alignedFootprintsAllPoints[:, 1] - y0) * ky).astype(int), np.round((alignedFootprintsAllPoints[:, 0] - x0) * kx).astype(int)]
allDistsSumsOverDensities = footprintPointsAllDistsSum*np.sum(densitiesAtFootprintPoints)/densitiesAtFootprintPoints

def discretizeDataValuesToImage(x0, y0, x1, y1, w, h, data, dataValues):
    kx = (w - 1) / (x1 - x0)
    ky = (h - 1) / (y1 - y0)
    r = 20
    border = r
    imgw = (w + 2 * border)
    imgh = (h + 2 * border)
    img = np.zeros((imgh,imgw))
    for pointIdx, (x, y) in enumerate(data):
        ix = int((x - x0) * kx) + border
        iy = int((y - y0) * ky) + border
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy][ix] += dataValues[pointIdx]
    return spimg.gaussian_filter(img, (r,r))

accumulatedDistsSumsOverDensitiesImg = discretizeDataValuesToImage(x0, y0, x1, y1, densityImg.shape[1], densityImg.shape[0], alignedFootprintsAllPoints, allDistsSumsOverDensities)
accumulatedDistsSumsOverDensities = accumulatedDistsSumsOverDensitiesImg[np.round((alignedFootprintsAllPoints[:, 1] - y0) * ky).astype(int), np.round((alignedFootprintsAllPoints[:, 0] - x0) * kx).astype(int)]
figure(); imshow(accumulatedDistsSumsOverDensitiesImg)
xlim([0, accumulatedDistsSumsOverDensitiesImg.shape[1]]); ylim([0, accumulatedDistsSumsOverDensitiesImg.shape[0]])
plot((footprintPointsBBox[[0, 1, 2, 3, 0], 0] - x0) * kx, (footprintPointsBBox[[0, 1, 2, 3, 0], 1] - y0) * ky, c="red")


# In[46]:

def nnInterp(point, neighs, neighVals, p=6.0) :
    ## can't seem to find where I got this from
    return np.sum(neighVals/(np.sum((neighs-point)**2, axis=1)**(p/2.0)))/np.sum(1.0/(np.sum((neighs-point)**2, axis=1)**(p/2.0)))

## do the same non max suppression thing but straight onto the footprint points
def getInterpolatedValuesAtPoints(points, allPoints, allPointsVals, numNeighs=4) :
    distancesToAllPoints = ssd2(points, allPoints)
    closestNeighs = np.argsort(distancesToAllPoints, axis=1)[:, :numNeighs]
    valsAtPoints = np.zeros(len(points))
    for i in np.arange(len(valsAtPoints)) :
        valsAtPoints[i] = nnInterp(points[i, :], allPoints[closestNeighs[i, :], :], allPointsVals[closestNeighs[i, :]])
    return valsAtPoints

isLocalMax = np.zeros(len(alignedFootprintsAllPoints), dtype=bool)
numNeighs = 4

dirs = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]], float)
dirs /= np.linalg.norm(dirs, axis=1)[:, np.newaxis]
perpendicularDirsIdxs = np.array([2, 3, 0, 1], int)
h = 0.05 ## maybe set this based on the footprintBBox

# pointsVals = allDistsSumsOverDensities/np.max(allDistsSumsOverDensities)
pointsVals = densitiesAtFootprintPoints/np.max(densitiesAtFootprintPoints)

for pointIdx in np.arange(len(alignedFootprintsAllPoints)) :
    point = alignedFootprintsAllPoints[pointIdx, :]
    pointsOnPositiveDirs = point+dirs*h*0.5
    pointsOnNegativeDirs = point-dirs*h*0.5
    valsAtPointsOnPositiveDirs = getInterpolatedValuesAtPoints(pointsOnPositiveDirs, alignedFootprintsAllPoints, pointsVals, numNeighs)
    valsAtPointsOnNegativeDirs = getInterpolatedValuesAtPoints(pointsOnNegativeDirs, alignedFootprintsAllPoints, pointsVals, numNeighs)
    gradients = (valsAtPointsOnPositiveDirs-valsAtPointsOnNegativeDirs)/h
    maxGradientDirIdx = np.argmax(np.abs(gradients))

    if (pointsVals[pointIdx]-valsAtPointsOnPositiveDirs[perpendicularDirsIdxs[maxGradientDirIdx]] > 1e-10 and
        pointsVals[pointIdx]-valsAtPointsOnNegativeDirs[perpendicularDirsIdxs[maxGradientDirIdx]] > 1e-10) :
        isLocalMax[pointIdx] = True

# figure(); scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(np.log(allDistsSumsOverDensities)/np.max(np.log(allDistsSumsOverDensities)), alpha=1),
#                   marker="o", facecolors='none', s=80, edgecolors=[0, 0, 0, 0.2])
# scatter(alignedFootprintsAllPoints[pointIdx, 0], alignedFootprintsAllPoints[pointIdx, 1])
# plot([pointsOnNegativeDirs[maxGradientDirIdx, 0], pointsOnPositiveDirs[maxGradientDirIdx, 0]], [pointsOnNegativeDirs[maxGradientDirIdx, 1], pointsOnPositiveDirs[maxGradientDirIdx, 1]])


# In[47]:

## from here http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
## but it doesn't seem to work for some reason
def rotateVectorOntoVector3D(a, b) :
    """ DOES NOT WORK: returns rotation matrix that rotates vector a onto vector b """
    v = np.cross(a, b)
    v1, v2, v3 = v/np.linalg.norm(v)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0.0, -v3, v2],
                   [v3, 0.0, -v1],
                   [-v2, v1, 0.0]])
    if s != 0.0 :
        return np.eye(3) + vx + (vx**2)/(1+c)
    else :
        return np.eye(3)
    
def calipersOMBB(points) :
    """ computes minimum area oriented bounding box given a set of points using the calipers algorithm """
    
    convexHull = cv2.convexHull(points.astype(np.float32))[:, 0, :]
#     figure(); scatter(points[:, 0], points[:, 1]); plot(convexHull[np.mod(arange(len(convexHull)+1), len(convexHull)), 0], convexHull[np.mod(arange(len(convexHull)+1), len(convexHull)), 1])
#     scatter(convexHull[0, 0],convexHull[0, 1], color="red")
#     xlim([-1, 1])
#     ylim([-6.5, -4.5])
    
    minArea = 10000000000.0
    ombb = np.zeros([4, 2])
    for i in np.arange(len(convexHull))[0:] :
        j = np.mod(i+1, len(convexHull))
        
        segmentDir = convexHull[j, :] - convexHull[i, :]
        segmentDir /= np.linalg.norm(segmentDir)
        
        crossProduct = np.cross(np.array([1.0, 0.0, 0.0]), np.concatenate([segmentDir, [0]]))
        dotProduct = np.dot(segmentDir, np.array([1.0, 0.0]))
        #I know the last component of both vectors is 0 so the rotation matrix will be all zeros there
        T = quaternionTo4x4Rotation(angleAxisToQuaternion(np.arccos(dotProduct), crossProduct/np.linalg.norm(crossProduct)))[:-2, :-2]
        
        transformedPoints = np.dot(T, points.T-convexHull[i, :][:, np.newaxis]).T + convexHull[i, :]
        
        [x0, y0], [x1, y1] = np.min(transformedPoints, axis=0), np.max(transformedPoints, axis=0)
        
        transformedBBox = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        bbox = np.dot(np.linalg.inv(T), transformedBBox.T-convexHull[i, :][:, np.newaxis]).T + convexHull[i, :]
        
        bboxArea = (x1-x0)*(y1-y0)
        if bboxArea < minArea :
            minArea = np.copy(bboxArea)
            ombb = np.copy(bbox)
            
#         transformedHull = np.dot(T, convexHull.T-convexHull[i, :][:, np.newaxis]).T + convexHull[i, :]
#         scatter(transformedPoints[:, 0], transformedPoints[:, 1], color="cyan")
#         plot(transformedHull[np.mod(arange(len(convexHull)+1), len(convexHull)), 0], transformedHull[np.mod(arange(len(convexHull)+1), len(convexHull)), 1], color="magenta")
#         scatter(transformedHull[np.mod(arange(len(convexHull)+1), len(convexHull)), 0], transformedHull[np.mod(arange(len(convexHull)+1), len(convexHull)), 1], color="magenta")
#         plot(transformedBBox[np.mod(arange(len(transformedBBox)+1), len(transformedBBox)), 0], transformedBBox[np.mod(arange(len(transformedBBox)+1), len(transformedBBox)), 1], color="magenta")
#         scatter(convexHull[i, 0], convexHull[i, 1], color="green"); plot([convexHull[i, 0], convexHull[i, 0]+segmentDir[0]], [convexHull[i, 1], convexHull[i, 1]+segmentDir[1]], color="green")
#         plot([convexHull[i, 0], convexHull[i, 0]+1], [convexHull[i, 1], convexHull[i, 1]], color="green")
#         plot(bbox[np.mod(arange(len(bbox)+1), len(bbox)), 0], bbox[np.mod(arange(len(bbox)+1), len(bbox)), 1], color="red")
#     plot(ombb[np.mod(arange(len(ombb)+1), len(ombb)), 0], ombb[np.mod(arange(len(ombb)+1), len(ombb)), 1], color="green")
    return ombb
    
calipersOMBB(alignedFootprintsAllPoints)


# In[48]:

def sampleOBB(obb, numSubdivs) :
    return np.vstack([v1+np.arange(0.0, (numSubdivs+1.0)/numSubdivs, 1.0/numSubdivs)[:, np.newaxis]*(v2-v1)[np.newaxis, :] for v1, v2 in zip(obb, obb[np.mod(np.arange(1, len(obb)+1), len(obb)), :])])

def getOBBVerticesPerpendicularDirs(obb) :
    return np.vstack([((v2-v1)/np.linalg.norm(v2-v1))[newaxis, :] for v1, v2 in zip(obb, obb[np.arange(len(obb))-1, :])])

def updateOBBScale(obb, dirs, amounts) :
    """ scales the obb by moving the i'th segment (comprised of the i'th and i+1'th vertices in obb) by the i'th dir times the i'th amount """
    return np.vstack([obb[idx, :]+dirs[idx, :]*amounts[idx]+dirs[idx-1, :]*amounts[idx-1] for idx in np.arange(len(dirs))])

def functionToMinimize(p, src, dst, dirs, weights, allPoints) :
    ps = p.repeat(len(src)/len(p))[:, np.newaxis]
    updatedSrc = src+dirs*ps
    
    return np.sum(np.sum((updatedSrc-dst)**2, axis=1)*weights)

doUseNNInterp = False
currentOBB = calipersOMBB(alignedFootprintsAllPoints)
p = np.zeros(len(currentOBB))
numSubdivs = 10
numNeighs = 4

if doUseNNInterp : 
#     pointsVals = allDistsSumsOverDensities/np.max(allDistsSumsOverDensities)
    pointsVals = densitiesAtFootprintPoints/np.max(densitiesAtFootprintPoints)
#     pointsVals = accumulatedDistsSumsOverDensities/np.max(accumulatedDistsSumsOverDensities)
else : 
    pointsVals = allDistsSumsOverDensities/np.max(allDistsSumsOverDensities)
    
    ## ACCUMULATE THE POINTS VALUES OVER A GRID (basically getting what I have in accumulatedDistsSumsOverDensities if I use pointsVals = allDistsSumsOverDensities) AND USE TO FIND BEST NEIGHBOURS
    ## get grid bounds
    gridX0, gridY0, gridX1, gridY1 = np.min(currentOBB[:, 0]), np.min(currentOBB[:, 1]), np.max(currentOBB[:, 0]), np.max(currentOBB[:, 1])
    gridWidth, gridHeight = (gridX1-gridX0, gridY1-gridY0)
    ## extend by a factor
    extendFactor = 0.1
    gridExtend = np.array([gridWidth, gridHeight])*extendFactor
    gridX0, gridY0 = np.array([gridX0, gridY0])-gridExtend
    gridX1, gridY1 = np.array([gridX1, gridY1])+gridExtend
    discreteGridWidth = 512
    ## accumulate the pointsVals over discrete grid 
    accumulatedPointsValsDiscreteGrid = discretizeDataValuesToImage(gridX0, gridY0, gridX1, gridY1, discreteGridWidth, np.ceil(discreteGridWidth*gridHeight/gridWidth).astype(int), 
                                                                    alignedFootprintsAllPoints, pointsVals)
    gridScaleX = (accumulatedPointsValsDiscreteGrid.shape[1] - 1) / (gridX1 - gridX0)
    gridScaleY = (accumulatedPointsValsDiscreteGrid.shape[0] - 1) / (gridY1 - gridY0)
    pointsValsFromGrid = accumulatedPointsValsDiscreteGrid[np.round((alignedFootprintsAllPoints[:, 1] - gridY0) * gridScaleY).astype(int), np.round((alignedFootprintsAllPoints[:, 0] - gridX0) * gridScaleX).astype(int)]
    pointsValsFromGrid = pointsValsFromGrid/np.max(pointsValsFromGrid)


figure() 
doShowIterEvolution = True
if doShowIterEvolution :
    ion()

startTime = time.time()
numIters = 10
for iterNum in np.arange(numIters) :
    if doShowIterEvolution or iterNum == numIters-1 :
        cla()
        if doUseNNInterp :
            scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(np.log(1+pointsVals)/np.max(np.log(1+pointsVals)), alpha=1),
                    marker="o", s=60, edgecolors=[0, 0, 0, 0.2])
        else :
            scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(np.log(1+pointsValsFromGrid)/np.max(np.log(1+pointsValsFromGrid)), alpha=1),
                    marker="o", s=60, edgecolors=[0, 0, 0, 0.2])
            
        xlim([-1, 1])
        ylim([-6.5, -4.5])
    
    
    ## sample the current OBB
    sampledCurrentOBB = sampleOBB(currentOBB, numSubdivs)
    vertexMoveDir = getOBBVerticesPerpendicularDirs(currentOBB)
    pointsMoveDir = np.vstack([moveDir[np.newaxis, :].repeat(numSubdivs+1, axis=0) for moveDir in vertexMoveDir])
    ## length of subdivision for the segment the point belongs to
    pointsSubdivisionLength = np.concatenate([np.linalg.norm(v2-v1).repeat(numSubdivs+1)/numSubdivs for v1, v2 in zip(currentOBB, currentOBB[np.mod(np.arange(1, len(currentOBB)+1), len(currentOBB)), :])])
    
    ## find best neighbours in +- moveDir
    ## do this by discretizing the +- dir and find best value (not the fastests of things but it should do for now) --> one speed up would be use the point density image (seeing how I'm discretizing anyways) and do NNinterp
    discreteRatio = 0.05
    bestNeighbours = np.zeros_like(sampledCurrentOBB)
    bestNeighsValues = np.zeros(len(sampledCurrentOBB))
    
    ## only look inside obb if it's the first iteration because I know I start with an OMBB
    neighboursRange = np.arange(0.0, 1.0+discreteRatio, discreteRatio) if iterNum == 0 else np.arange(-1.0, 1.0+discreteRatio, discreteRatio)
    for pointIdx in np.arange(len(sampledCurrentOBB)) :
        closeNeighbours = (neighboursRange*pointsSubdivisionLength[pointIdx]*pointsMoveDir[pointIdx, :][:, np.newaxis]).T+sampledCurrentOBB[pointIdx, :][np.newaxis, :]
        if doUseNNInterp :
            ## use nnInterp
            valuesAtNeighbours = getInterpolatedValuesAtPoints(closeNeighbours, alignedFootprintsAllPoints, pointsVals, numNeighs)
        else :
            ## use discrete grid
            ## THIS COULD PROBABLY BE VECTORIZED MORE BUT IT'S ALREADY 10x faster than nnInterp and it doesn't suffer from the problem discussed in the 14/02/17 journal entry
            closeNeighboursGridSpace = (closeNeighbours-np.array([[gridX0, gridY0]]))*np.array([[gridScaleX, gridScaleY]])
            valuesAtNeighbours = []
            for closeNeighbour in closeNeighboursGridSpace :
                nearestNeighbours = np.concatenate([np.floor(closeNeighbour), np.ceil(closeNeighbour)])[[0, 1, 0, 3, 2, 3, 2, 1]].reshape([4, 2]).astype(int)
                distsToNNs = np.linalg.norm(nearestNeighbours-closeNeighbour, axis=1)
                weights = distsToNNs/np.sum(distsToNNs)
                valuesAtNeighbours.append(np.sum(accumulatedPointsValsDiscreteGrid[nearestNeighbours[:, 1], nearestNeighbours[:, 0]]*weights))
            valuesAtNeighbours = np.array(valuesAtNeighbours)
            
        bestNeighbourIdx = np.argmax(valuesAtNeighbours).flatten().astype(int)
        bestNeighbours[pointIdx, :] = closeNeighbours[bestNeighbourIdx, :]
        bestNeighsValues[pointIdx] = valuesAtNeighbours[bestNeighbourIdx]
        
    bestNeighsValues /= np.sum(bestNeighsValues)
    
    optResult = minimize(functionToMinimize, np.zeros_like(p), args=(sampledCurrentOBB, bestNeighbours, pointsMoveDir, bestNeighsValues, alignedFootprintsAllPoints), method='BFGS') #, method='Newton-CG')#, jac=jac, hess=hess)
    p = optResult.x
    currentOBB = updateOBBScale(currentOBB, vertexMoveDir, p)
    
    if doShowIterEvolution or iterNum == numIters-1 :
        plot(currentOBB[np.mod(arange(len(currentOBB)+1), len(currentOBB)), 0], currentOBB[np.mod(arange(len(currentOBB)+1), len(currentOBB)), 1], color="green")
        scatter(sampledCurrentOBB[:, 0], sampledCurrentOBB[:, 1], color="green")
        scatter(bestNeighbours[:, 0], bestNeighbours[:, 1], c=cm.jet(bestNeighsValues/np.max(bestNeighsValues), alpha=1))
        show()
        pause(0.05)
        
print(time.time()-startTime)


# In[512]:

def distortPoints(undistortedPoints, distortionCoeff, undistortedIntrinsics, distortedIntrinsics) :
    """ distorts points in an undistorted image space to the original image space as if they are 
    seen again through the distorting lens
    
    - as seen here: http://stackoverflow.com/a/35016615"""
    ## not sure what this does but it doesn't work without it
    tmp = cv2.undistortPoints(undistortedPoints.reshape([1, len(undistortedPoints), 2]),
                              undistortedIntrinsics, np.zeros(5))
    distortedPoints = cv2.projectPoints(np.concatenate([tmp, np.ones([1, tmp.shape[1], 1])], axis=-1), (0, 0, 0),
                                        (0, 0, 0), distortedIntrinsics, distortionCoeff)[0][:, 0, :]
    return distortedPoints

def patchImageIdxsFromCenterAndSize(patchCenter, patchHalfSize, imageShape) :
    """patchCenter is (x, y) and patchHalfSize is (width, height)
    returns [minRow, minCol, maxRow, maxCol]"""
    ## enlarge patch and make sure it's within bounds
    patchMinCorner = np.floor(np.max([np.zeros(2), patchCenter-patchHalfSize], axis=0)).astype(int)
    patchMaxCorner = np.ceil(np.min([np.array(imageShape[::-1]),
                                     patchCenter+patchHalfSize+1], axis=0)).astype(int)
    return np.concatenate([patchMinCorner[::-1], patchMaxCorner[::-1]])


########################## INIT AND GET FILMED SCENE AND OBJECT DATA AND STUFF ##########################

boundingVolumeVertexDrawIndices = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 0, 1, 5, 6, 2, 3, 7]
# filmedScene = GLFilmedScene("/home/ilisescu/PhD/data/havana/filmed_scene-havana.npy")
# filmedObject = filmedScene.filmedObjects[0]
filmedObject.footprintScale = 0.25
filmedObject.footprintAspectRatio=2.35
filmedObject.setGeometryAndBuffers()
filmedObjectHeight = 0.18 ## this would have to be eventually found at the same time as the footprint box
viewMat, projectionMat = cvCameraToOpenGL(filmedScene.cameraExtrinsics, filmedScene.cameraIntrinsics,
                                          filmedScene.medianImage.shape[:-1])


########################## GET TRAJECTORY OF CURRENT OBJECT ##########################

# f = open(filmedObject.filmedObjectData[DICT_TRACK_LOCATION], 'r')
f = open("/home/ilisescu/PhD/data/havana/{0}-track_longer.txt".format("blue_car1"), 'r')
lines = f.readlines()
vals = [np.array(i.split(" ")).astype(float) for i in lines]
vals = [(int(i[-1]), i[0:2]) for i in vals]
tmp = dict(vals)
trajectoryPointsFrameIds = np.sort(tmp.keys())
trajectoryPoints = np.array([tmp[key] for key in trajectoryPointsFrameIds])

## here I'm using code from GLFilmedObject
trajectoryPoints = trajectoryPoints + filmedScene.cameraIntrinsics[:2, -1] - filmedScene.filmedSceneData[DICT_CAMERA_INTRINSICS][:2, -1]
trajectory = GLTrajectory(trajectoryPoints, filmedScene.cameraIntrinsics, filmedScene.cameraExtrinsics, 
                          filmedObject.filmedObjectData[DICT_REPRESENTATIVE_COLOR], doSmoothing = False)


filmedObjectTransform = filmedObject.modelMat
distanceTransforms = np.zeros([720, 1280, len(trajectoryPointsFrameIds)], dtype=np.float32)
patchesImageIdxs = np.zeros([4, len(trajectoryPointsFrameIds)], dtype=np.int32)
for currentFrameIdx in np.arange(len(trajectoryPointsFrameIds)) :
    
    ########################## SET OBJECT ONTO TRAJECTORY AND RENDER ITS BOUNDING VOLUME ##########################

    ## I'm basically using utilities from GLFilmedObject but using a different trajectory 
    ## (because the trajectory saved for filmed objects are clamped to make sure the object is always fully visible)
    positionWorld = trajectory.worldTrajectoryPoints[currentFrameIdx, :]
    directionWorld = trajectory.worldTrajectoryDirections[currentFrameIdx, :]

    objPos, objFDir = getWorldSpacePosAndNorm(filmedObjectTransform, filmedObject.forwardDir)
    adjustAngle = np.arccos(np.clip(np.dot(objFDir, directionWorld), -1, 1))
    if np.abs(adjustAngle) > 1e-06 :
    #             print(adjustAngle, np.cross(directionWorld, objFDir))
        adjustAxis = np.cross(directionWorld, objFDir)
        adjustAxis /= np.linalg.norm(adjustAxis)
        filmedObjectTransform = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis)), filmedObjectTransform)
    filmedObjectTransform[:-1, -1] = positionWorld

    worldFootprintVertices = np.dot(filmedObjectTransform, np.hstack([filmedObject.footprintVertices[[0, 1, 4, 5], :],
                                                                      np.ones([4, 1])]).T)
    worldFootprintVertices = worldFootprintVertices[:-1, :]/worldFootprintVertices[-1, :]
    worldFootprintVertices = worldFootprintVertices.T
    worldBoundingVolumeVertices = np.vstack([worldFootprintVertices,
                                             worldFootprintVertices+np.array([[0, 0, 1.0]])*filmedObjectHeight])

    cameraBoundingVolumeVertices = worldToScreenSpace(viewMat, projectionMat, worldBoundingVolumeVertices,
                                                      filmedScene.medianImage.shape[1], filmedScene.medianImage.shape[0])

    # T = np.dot(filmedScene.cameraIntrinsics, filmedScene.cameraExtrinsics[:-1, [0, 1, 3]])
    # cameraFootprintVertices = np.dot(T, np.vstack([worldFootprintVertices[:-1, :],
    #                                                np.ones([1, len(worldFootprintVertices.T)])]))
    # cameraFootprintVertices = cameraFootprintVertices[:-1, :]/cameraFootprintVertices[-1, :]



    # currentFrameImg, _, _, _, _ = undistortImage(filmedScene.filmedSceneData[DICT_DISTORTION_PARAMETER],
    #                                              filmedScene.filmedSceneData[DICT_DISTORTION_RATIO],
    #                                              np.array(Image.open(dataLoc+"frame-{0:05}.png".format(trajectoryPointsFrameIds[currentFrameIdx]+1))),
    #                                              filmedScene.filmedSceneData[DICT_CAMERA_INTRINSICS])
    # figure(); imshow(currentFrameImg); xlim([0, currentFrameImg.shape[1]]); ylim([currentFrameImg.shape[0], 0])
    # plot(trajectoryPoints[:, 0], trajectoryPoints[:, 1],
    #      c=tuple(filmedObject.filmedObjectData[DICT_REPRESENTATIVE_COLOR]/255.0))
    # plot(cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, 0],
    #      cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, 1], c="green")
    # scatter(trajectoryPoints[[50, 30], 0], trajectoryPoints[[50, 30], 1])


    ########################## DISTORT THE cameraBoundingVolumeVertices SO THAT I CAN WORK IN THE ORIGINAL IMAGE SPACE ##########################

    currentFrameImg = np.array(Image.open(dataLoc+"frame-{0:05}.png".format(trajectoryPointsFrameIds[currentFrameIdx]+1)))
    # cameraBoundingVolumeVertices = cv2.undistortPoints(cameraBoundingVolumeVertices.reshape((1, len(cameraBoundingVolumeVertices), 2)),
    #                                                    filmedScene.cameraIntrinsics, -filmedScene.distortionCoeff,
    #                                                    P=filmedScene.filmedSceneData[DICT_CAMERA_INTRINSICS])[0, :, :]
    # cameraTrajectoryPoints = cv2.undistortPoints(trajectoryPoints.reshape((1, len(trajectoryPoints), 2)),
    #                                              filmedScene.cameraIntrinsics, -filmedScene.distortionCoeff,
    #                                              P=filmedScene.filmedSceneData[DICT_CAMERA_INTRINSICS])[0, :, :]
    cameraBoundingVolumeVertices = distortPoints(cameraBoundingVolumeVertices, filmedScene.distortionCoeff,
                                                 filmedScene.cameraIntrinsics, 
                                                 filmedScene.filmedSceneData[DICT_CAMERA_INTRINSICS])
    cameraTrajectoryPoints = distortPoints(trajectoryPoints, filmedScene.distortionCoeff, filmedScene.cameraIntrinsics, 
                                           filmedScene.filmedSceneData[DICT_CAMERA_INTRINSICS])

    tmpImg = np.copy(currentFrameImg)
    cv2.polylines(tmpImg, [cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, :].astype(np.int32).reshape((-1,1,2))], False, (0, 0, 255))
    Image.fromarray(tmpImg.astype(np.uint8)).save('/home/ilisescu/PhD/data/havana/segmentation_shapePrior_input/frame-{0:05}.png'.format(trajectoryPointsFrameIds[currentFrameIdx]+1))
#     figure(); imshow(currentFrameImg); xlim([0, currentFrameImg.shape[1]]); ylim([currentFrameImg.shape[0], 0])
#     plot(cameraTrajectoryPoints[:, 0], cameraTrajectoryPoints[:, 1],
#          c=tuple(filmedObject.filmedObjectData[DICT_REPRESENTATIVE_COLOR]/255.0))
#     plot(cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, 0],
#          cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndicboundingVolumeVertexDrawIndiceses, 1], c="green")
#     scatter(cameraTrajectoryPoints[[50, 30], 0], cameraTrajectoryPoints[[50, 30], 1])


    ########################## FIND CONVEX HULL OF PROJECTED BOUNDING VOLUME AND COMPUTE DIST TRANSFORM ##########################

    hullIdxs = ConvexHull(cameraBoundingVolumeVertices).vertices
    cameraConvexHull = cameraBoundingVolumeVertices[hullIdxs, :]
#     plot(cameraConvexHull[:, 0], cameraConvexHull[:, 1], c="red")

    volumeConvexHullImg = np.ones(currentFrameImg.shape[0:-1])
    cv2.polylines(volumeConvexHullImg, [cameraConvexHull.astype(np.int32).reshape((-1,1,2))], True, (0))
    distanceTransformImg = spimg.morphology.distance_transform_edt(volumeConvexHullImg)

    ## the patch will be of size 1+patchHalfSize*2 (unless image borders are hit of course) and centered on patchCenter
    patchSize = np.ceil(np.max(cameraConvexHull, axis=0))-np.floor(np.min(cameraConvexHull, axis=0))
    patchCenter = np.round(np.floor(np.min(cameraConvexHull, axis=0))+patchSize/2.0)
    enlargePercentage = 1.0
    patchHalfSize = np.ceil((patchSize + patchSize*enlargePercentage)/2.0)

    patchImageIdxs = patchImageIdxsFromCenterAndSize(patchCenter, patchHalfSize, currentFrameImg.shape[0:2])
#     figure(); imshow(currentFrameImg[patchImageIdxs[0]:patchImageIdxs[2], patchImageIdxs[1]:patchImageIdxs[3]])
#     plot(cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, 0]-patchImageIdxs[1],
#          cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, 1]-patchImageIdxs[0], c="green")
#     figure(); imshow(distanceTransformImg[patchImageIdxs[0]:patchImageIdxs[2], patchImageIdxs[1]:patchImageIdxs[3]])
#     plot(cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, 0]-patchImageIdxs[1],
#          cameraBoundingVolumeVertices[boundingVolumeVertexDrawIndices, 1]-patchImageIdxs[0], c="green")
    
    patchesImageIdxs[:, currentFrameIdx] = patchImageIdxs
    distanceTransforms[:, :, currentFrameIdx] = distanceTransformImg
    print(currentFrameIdx)
    sys.stdout.flush()


# In[429]:

def backgroundCut3DShapePrior(bgImage, images, shapePriors, shapePriorWeight=0.5, k1=30.0/255.0, k2=60.0/255.0, K=5.0/255.0, sigmaZ=10.0/255.0) :
    """ Given a stack of temporally sequential images, a static background bgImage and a stack of shapePriors under the form of unsigned distance functions,
    it computes temporally consistent fg/bg segmentation enforcing cuts to be close to shape boundaries by adding extra pairwise term as seen in Equation (1) in
    Interactive GC segmentation [Freedman and Zhang. CVPR2005]
    
    based on BGcut [Sun et al. ECCV2006] with modifications seen in Video Synposis [Pritch et al. PAMI2008]"""
    ## as seen in Sun's background cut (with the mods made in pritch synopsis paper)
#     figure("bgImage"); imshow(bgImage); figure("image"); imshow(image)
    
    if np.all(bgImage.shape != images.shape[:-1]) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
    
    height, width, channels, numImages = images.shape
    maxCost = 10000000.0#np.sys.float_info.max
    
    bgPixels = bgImage.reshape([height*width, channels], order='F')/255.0
    
    s = time.time()
    ## build graph
    numLabels = 2
    gm = opengm.gm(np.ones(height*width*numImages,dtype=opengm.label_type)*numLabels)
    
    for i in np.arange(numImages) :
        imagePixels1 = images[:, :, :, i].reshape([height*width, channels], order='F')/255.0
        shapePrior1 = shapePriors[:, :, i].reshape([height*width], order='F')


        ############################### COMPUTE UNARIES ###############################
        unaries = np.zeros((height*width,numLabels))

        dr = np.sqrt(np.sum((imagePixels1-bgPixels)**2.0, axis=-1))

        unaries[dr<=k1, 1] = (k1-dr)[dr<=k1]
        unaries[dr>k2, 0] = maxCost
        unaries[np.all(np.array([dr>k1, k2>dr]), axis=0), 0] = (dr-k1)[np.all(np.array([dr>k1, k2>dr]), axis=0)]
        unaries *= (1.0-shapePriorWeight)

        # add functions
        fids = gm.addFunctions(unaries)
        # add first order factors
        gm.addFactors(fids, np.arange(i*height*width, (i+1)*height*width, 1))


        ############################### COMPUTE PAIRWISE ###############################
        for j in np.arange(2) :
            if j == 0 or (i > 0 and j ==1) :
                pairIndices = getGridPairIndices(width, height)

                imagePixels2 = imagePixels1
                shapePrior2 = shapePrior1
                if i > 0 and j == 1 :
                    ## in this case compute pairwise between temporally neighbouring pixels in current image and previous one
                    pairIndices = np.concatenate([[np.arange(width*height)], [np.arange(width*height)]]).T
                    imagePixels2 = images[:, :, :, i-1].reshape([height*width, channels], order='F')/255.0
                    shapePrior2 = shapePriors[:, :, i-1].reshape([height*width], order='F')

                pairwise = np.zeros(len(pairIndices))

                ### not sure why I use pairIndices[:, 0] on the second image and pairIndices[:, 1] on the first image and not the other way around but I don't think it makes any difference
                zrs = np.max([np.sqrt(np.sum((imagePixels2[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 0], :])**2.0, axis=-1)),
                              np.sqrt(np.sum((imagePixels1[pairIndices[:, 1], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))], axis=0)

                imPixelsDiff = np.sqrt(np.sum((imagePixels2[pairIndices[:, 0], :]-imagePixels1[pairIndices[:, 1], :])**2.0, axis=-1))
                bgPixelsDiff = np.sqrt(np.sum((bgPixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))
                drs = imPixelsDiff/(1+((bgPixelsDiff/K)**2.0)*np.exp(-(zrs**2)/sigmaZ))
                beta = 2.0/np.mean(imPixelsDiff)
                pairwise = (1.0-shapePriorWeight)*np.exp(-beta*drs)

                ## visualize
                if False :
                    contrastMap = np.zeros(len(bgPixels))
                    for pairIndicesIdx in np.arange((width-1)*(height-1)*2) :
                        contrastMap[pairIndices[pairIndicesIdx, 0]] += drs[pairIndicesIdx]
                    figure(); imshow(np.reshape(np.sqrt(np.copy(contrastMap)), [height, width], 'F'))

                ## not sure it makes sense to compute the shape prior thing for the temporal neighbours but let's see
                if j == 0 : #or True :
#                     print("DOING THA SHIT", i, j)
                    ## here I get the distance transform at p and at q and divide by two rather than getting the distance transform at (p+q)/2 which is how they write it down in the paper
                    ## not sure it's the same thing but i think it is
                    shapePairwise = (shapePrior2[pairIndices[:, 0]]+shapePrior1[pairIndices[:, 1]])/2.0
#                     print(len(pairIndices), np.max(shapePriorWeight*shapePairwise), np.max(pairwise))
                    pairwise += (shapePriorWeight*shapePairwise)
                    
#                     contrastMap = np.zeros(len(bgPixels))
#                     for pairIndicesIdx in np.arange((width-1)*(height-1)*2) :
#                         contrastMap[pairIndices[pairIndicesIdx, 0]] += shapePairwise[pairIndicesIdx]
#                     figure(); imshow(np.reshape(np.sqrt(np.copy(contrastMap)), [height, width], 'F'))
                
                # add functions
                fids = gm.addFunctions(np.array([[0.0, 1.0],[1.0, 0.0]]).reshape((1, 2, 2)).repeat(len(pairwise), axis=0)*
                                       pairwise.reshape((len(pairwise), 1, 1)).repeat(2, axis=1).repeat(2, axis=2))

                if j == 0 :
                    ## in this case compute pairwise between neighbouring pixels in the current image
                    # add second order factors
                    gm.addFactors(fids, pairIndices+(i*height*width))
                elif i > 0 and j == 1 :
                    ## in this case compute pairwise between temporally neighbouring pixels in current image and previous one
                    # add second order factors
                    pairIndices[:, 0] += ((i-1)*height*width)
                    pairIndices[:, 1] += (i*height*width)
                    gm.addFactors(fids, pairIndices)
    
#     print(gm)
    
    
    graphCut = opengm.inference.GraphCut(gm=gm)
    graphCut.infer()    
    labels = np.array(graphCut.arg(), dtype=int)
    reshapedLabels = np.reshape(np.copy(labels), [height, width, numImages], 'F')
    
    return reshapedLabels


# In[513]:

# figure(); imshow(distanceTransforms[patchesImageIdxs[0, 0]:patchesImageIdxs[2, 0],
#                                     patchesImageIdxs[1, 0]:patchesImageIdxs[3, 0], 0])

numNeighboringFrames = 5
currentFgMasks = np.zeros([bgImage.shape[0], bgImage.shape[1], len(trajectoryPoints)], dtype=np.uint8)
for currentFrameIdx in np.arange(len(trajectoryPoints)) :
    startTime = time.time()
    imgsIdxs = np.arange(np.max([0, currentFrameIdx-numNeighboringFrames/2]),
                         np.min([len(trajectoryPoints), currentFrameIdx+1+numNeighboringFrames/2]))
    currentFrameImgsIdx = int(np.argwhere(imgsIdxs == currentFrameIdx).flatten())
    
    commonPatchImageIdxs = np.concatenate([np.min(patchesImageIdxs[0:2, imgsIdxs], axis=1),
                                           np.max(patchesImageIdxs[2:, imgsIdxs], axis=1)])
    
    ims = np.zeros([commonPatchImageIdxs[2]-commonPatchImageIdxs[0],
                    commonPatchImageIdxs[3]-commonPatchImageIdxs[1], bgImage.shape[2], len(imgsIdxs)], dtype=np.uint8)
    ## load the images
    for idx, i in enumerate(imgsIdxs) :
        ims[:, :, :, idx] = np.array(Image.open(dataLoc+"frame-{0:05}.png".format(trajectoryPointsFrameIds[i]+1)))[commonPatchImageIdxs[0]:commonPatchImageIdxs[2],
                                                                                                                   commonPatchImageIdxs[1]:commonPatchImageIdxs[3], :]
    distanceTransformsPatches = distanceTransforms[commonPatchImageIdxs[0]:commonPatchImageIdxs[2],
                                                   commonPatchImageIdxs[1]:commonPatchImageIdxs[3], imgsIdxs]
    distanceTransformsPatches /= np.max(distanceTransformsPatches.reshape([np.prod(distanceTransformsPatches.shape[0:2]), len(imgsIdxs)]), axis=0)[np.newaxis, np.newaxis, :]
    currentFgMasks[commonPatchImageIdxs[0]:commonPatchImageIdxs[2],
                   commonPatchImageIdxs[1]:commonPatchImageIdxs[3],
                   currentFrameIdx] = backgroundCut3DShapePrior(bgImage[commonPatchImageIdxs[0]:commonPatchImageIdxs[2],
                                                                        commonPatchImageIdxs[1]:commonPatchImageIdxs[3], :], ims, distanceTransformsPatches, shapePriorWeight=0.7)[:, :, currentFrameImgsIdx]
    
    print(currentFrameIdx, currentFrameImgsIdx, imgsIdxs, time.time()-startTime)
#     ## compute mask
#     fgMasks[:, :, maskIdx] = backgroundCut3D(bgImage, ims)[:, :, numNeighboringFrames/2]
#     fgMasks[:, :, maskIdx] = cv2.morphologyEx(fgMasks[:, :, maskIdx].astype(float), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    sys.stdout.flush()


# In[514]:

# ims = np.zeros([bgImage.shape[0], bgImage.shape[1], bgImage.shape[2], len(imgsIdxs)], dtype=np.uint8)
# for idx, i in enumerate(imgsIdxs) :
#     ims[:, :, :, idx] = np.array(Image.open(dataLoc+"frame-{0:05}.png".format(trajectoryPointsFrameIds[i]+1)))
# tmp = backgroundCut3D(bgImage, ims)[:, :, currentFrameImgsIdx]
# figure(); imshow(tmp)
# figure()
# img = None
# for i in xrange(currentFgMasks.shape[-1]):
#     if img is None:
#         img = mpl.pylab.imshow(currentFgMasks[:, :, i])
#     else:
#         img.set_data(currentFgMasks[:, :, i])
#     mpl.pylab.pause(0.01)
#     mpl.pylab.draw()

saveLoc = '/home/ilisescu/PhD/data/havana/segmentation_tighterbox_5neighs_shapePrior_0.7/'
if not os.path.isdir(saveLoc):
    os.makedirs(saveLoc)
figure()
colors = cm.jet(np.arange(4)/3.0)
# create a patch (proxy artist) for every color 
patches = [ matplotlib.patches.Patch(color=tuple(colors[i]), label=["old FG", "BG", "old+new FG", "new FG"][i]) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
img = None
for i in xrange(currentFgMasks.shape[-1]):
    if i+4 < fgMasks.shape[-1] :
        imToShow = currentFgMasks[:, :, i]*2.0-fgMasks[:, :, i+4]
        if img is None:
            img = mpl.pylab.imshow(imToShow, interpolation='none')
        else:
            img.set_data(imToShow)
        mpl.pylab.pause(0.01)
        mpl.pylab.draw()
        savefig(saveLoc + 'figure-{0:05}.png'.format(trajectoryPointsFrameIds[i]+1), bbox_inches='tight', dpi=200)


# In[501]:

imToShow = currentFgMasks[:, :, 12]*2.0-fgMasks[:, :, 12+4]
figure(figsize=(8, 5), dpi=200); 
imshow(imToShow, interpolation='none')


# In[422]:

figure(); imshow(ims[:, :, :, 1])


# In[502]:

figure(); imshow(currentFgMasks[:, :, 12])
figure(); imshow(fgMasks[:, :, 12+4])


# In[421]:

figure(); imshow(currentFgMasks[:, :, currentFrameIdx])
# figure(); imshow(fgMasks[:, :, currentFrameIdx+4])
# figure(); imshow(distanceTransforms[commonPatchImageIdxs[0]:commonPatchImageIdxs[2], commonPatchImageIdxs[1]:commonPatchImageIdxs[3], imgsIdxs[2]])


# In[58]:

figure()
# xs, ys = np.meshgrid(np.arange(x0-0.5, x1+0.5+0.01, 0.01), np.arange(y0-0.5, y1+0.5+0.01, 0.01))
# gridPoints = np.hstack([xs.flatten()[:, np.newaxis], ys.flatten()[:, np.newaxis]])
# valuesAtGrid = getInterpolatedValuesAtPoints(gridPoints, alignedFootprintsAllPoints, pointsVals, numNeighs)
scatter(gridPoints[:, 0], gridPoints[:, 1], c=cm.jet(valuesAtGrid/np.max(valuesAtGrid), alpha=1), edgecolors='none')
print(gridPoints.shape)


# In[570]:

distancesToAllPoints = np.sqrt(ssd2(sampleOBB(currentOBB, numSubdivs), alignedFootprintsAllPoints))
medianValue = np.median(alignedFootprintsAllPoints, axis=0)
thresh = np.median(np.sqrt(np.sum((alignedFootprintsAllPoints-medianValue)**2, axis=1))) ## median absolute deviation
thresh = np.median(pointsSubdivisionLength)
pointsWithinThresh = np.any(np.vstack([d <= thresh for d in distancesToAllPoints]), axis=0)
scatter(alignedFootprintsAllPoints[pointsWithinThresh, 0], alignedFootprintsAllPoints[pointsWithinThresh, 1], color="yellow")
# scatter(sampleOBB(currentOBB, numSubdivs)[:, 0], sampleOBB(currentOBB, numSubdivs)[:, 1], color="magenta")
# print(distancesToAllPoints.shape)
# print(np.median(distancesToAllPoints))


# In[361]:

figure(); scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(np.log(1+pointsVals)/np.max(np.log(1+pointsVals)), alpha=1),
                  marker="o", facecolors='none', s=60, edgecolors=[0, 0, 0, 0.2])
scatter(alignedFootprintsAllPoints[isLocalMax, 0], alignedFootprintsAllPoints[isLocalMax, 1], c="cyan")


# In[591]:

figure()
scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(densitiesAtFootprintPoints/np.max(densitiesAtFootprintPoints), alpha=1),
        marker="o", facecolors='none', s=60, edgecolors=[0, 0, 0, 0.2])
figure()
scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(accumulatedDistsSumsOverDensities/np.max(accumulatedDistsSumsOverDensities), alpha=1),
        marker="o", facecolors='none', s=60, edgecolors=[0, 0, 0, 0.2])


# In[127]:

## trying a dynamic programming based fitting of a footprint box

## the unary is based on allDistsSumsOverDensities, interpolated based on nearest neighbours
## the pairwise makes sure that the box stays a box (i.e. cross product between ((i-1)-i)x((i+1)-i) is positive and is a right angle, assuming I order the vertices of the ox clockwise)
## I discretize the directions in which each vertex can move and compute the unary by moving the vertex in each direction by a certain stepSize (can fix it for now but can also set a decreasing one, or smth else(2nd deriv?))

x0, y0, x1, y1 = np.min(alignedFootprintsAllPoints[:, 0]), np.min(alignedFootprintsAllPoints[:, 1]), np.max(alignedFootprintsAllPoints[:, 0]), np.max(alignedFootprintsAllPoints[:, 1])
stepSize = np.sqrt((x1-x0)**2+(y1-y0)**2)*0.1
numDirections = 16
numNeighs = 4

displacementVecs = []
for angle in np.arange(0, 2*np.pi, np.pi*2/numDirections) :
    displacementVecs.append(np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(angle, np.array([0.0, 0.0, 1.0])))[:2, :2], np.array([1.0, 0.0])))
displacementVecs = np.array(displacementVecs)
displacementVecs = displacementVecs/np.linalg.norm(displacementVecs, axis=1)[:, np.newaxis]

totalUnaryCost = allDistsSumsOverDensities/np.sum(allDistsSumsOverDensities)
totalUnaryCost = np.exp(-totalUnaryCost/0.001)

currentFootprintBox = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
for iterNum in np.arange(1) :
    maxCost = 10000000.0#np.sys.float_info.max
    
    s = time.time()
    ## build graph
    numLabels = numDirections
    numNodes = len(currentFootprintBox)
    gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels)
    
    
    ############################### COMPUTE UNARIES ###############################
    unaries = np.zeros((numNodes,numLabels))
    
    for nodeIdx in np.arange(numNodes) :
        displacedPoints = currentFootprintBox[nodeIdx, :]+displacementVecs*stepSize
        distancesToAllPoints = ssd2(displacedPoints, alignedFootprintsAllPoints)
        closestNeighs = np.argsort(distancesToAllPoints, axis=1)[:, :numNeighs]
        
        for labelIdx in np.arange(numLabels) :
            unaries[nodeIdx, labelIdx] = nnInterp(displacedPoints[labelIdx, :], alignedFootprintsAllPoints[closestNeighs[labelIdx, :], :], totalUnaryCost[closestNeighs[labelIdx, :]])
        
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, np.arange(0, numNodes, 1))
    
    
    ############################### COMPUTE PAIRWISE ###############################
    pairIndices = np.vstack([[np.arange(numNodes)], [np.mod(np.arange(1, numNodes+1), numNodes)]]).T
    print(pairIndices)
    ###################################### I DON'T KNOW IF I CAN DO WHAT I WAS PLANNING HERE -- SEE JOURNAL 10/02/17 ######################################
#     pairIndices = getGridPairIndices(width, height)
    
#     pairwise = np.zeros(len(pairIndices))
    
#     zrs = np.max([np.sqrt(np.sum((imagePixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 0], :])**2.0, axis=-1)),
#                   np.sqrt(np.sum((imagePixels[pairIndices[:, 1], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))], axis=0)
    
#     imPixelsDiff = np.sqrt(np.sum((imagePixels[pairIndices[:, 0], :]-imagePixels[pairIndices[:, 1], :])**2.0, axis=-1))
#     bgPixelsDiff = np.sqrt(np.sum((bgPixels[pairIndices[:, 0], :]-bgPixels[pairIndices[:, 1], :])**2.0, axis=-1))
#     drs = imPixelsDiff/(1+((bgPixelsDiff/K)**2.0)*np.exp(-(zrs**2)/sigmaZ))
#     beta = 2.0/np.mean(imPixelsDiff)
#     pairwise = np.exp(-beta*drs)
    
#     ## visualize
#     if False :
#         contrastMap = np.zeros(len(bgPixels))
#         for i in np.arange((width-1)*(height-1)*2) :
#             contrastMap[pairIndices[i, 0]] += drs[i]
#         figure(); imshow(np.reshape(np.sqrt(np.copy(contrastMap)), [height, width], 'F'))
    
#     # add functions
#     fids = gm.addFunctions(np.array([[0.0, 1.0],[1.0, 0.0]]).reshape((1, 2, 2)).repeat(len(pairwise), axis=0)*
#                            pairwise.reshape((len(pairwise), 1, 1)).repeat(2, axis=1).repeat(2, axis=2))
    
#     # add second order factors
#     gm.addFactors(fids, pairIndices)
    
    print(gm)
    
    
#     dynProg = opengm.inference.DynamicProgramming(gm=gm)
#     dynProg.infer()    
#     labels = np.array(dynProg.arg(), dtype=int)
#     reshapedLabels = np.reshape(np.copy(labels), [height, width], 'F')


# In[85]:

np.arange(0, 2*np.pi, np.pi*2/numDirections)


# In[62]:

figure();
# scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(footprintPointsAllDistsSum/np.max(footprintPointsAllDistsSum), alpha=1))
# scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(footprintPointsNumCloseNeighs/np.max(footprintPointsNumCloseNeighs), alpha=1))
scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], c=cm.jet(np.log(allDistsSumsOverDensities)/np.max(np.log(allDistsSumsOverDensities)), alpha=1))
# gca().add_artist(pyplot.Circle(tuple(alignedFootprintsAllPoints[np.argmax(footprintPointsNumCloseNeighs), :]), thresh, color='r'))
plot(footprintPointsBBox[[0, 1, 2, 3, 0], 0], footprintPointsBBox[[0, 1, 2, 3, 0], 1])


# In[102]:

scatter(np.mean(alignedFootprintsAllPoints, axis=0)[0], np.mean(alignedFootprintsAllPoints, axis=0)[1])


# In[150]:

## get manual footprint bbox just to test and see how this all looks
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], color="green")
# xlim([-1, 1])
# ylim([-5, -7])

# # ax.imshow(medianImage)

# clickedPoints = np.empty([0, 2])

# def onclick(event):
# #     print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
# #           (event.button, event.x, event.y, event.xdata, event.ydata))
#     global clickedPoints
#     if event.xdata != None and event.ydata != None :
#         clickedPoints = np.concatenate([clickedPoints, np.array([[event.xdata, event.ydata]])])
#         cla()
#         gca().scatter(alignedFootprintsAllPoints[:, 0], alignedFootprintsAllPoints[:, 1], color="green")
#         xlim([-1, 1])
#         ylim([-5, -7])
#         gca().plot(clickedPoints[:, 0], clickedPoints[:, 1])
#         show()
#         print(clickedPoints); sys.stdout.flush()

# cid = fig.canvas.mpl_connect('button_press_event', onclick)

## box of footprint for last frame
manualFootprintBox = np.array([[-0.18951613, -5.96354167, 0.0],
                               [ 0.11693548, -5.94270833, 0.0],
                               [ 0.06451613, -5.34375   , 0.0],
                               [-0.25806452, -5.36643836, 0.0]])
longEdgeLength = np.linalg.norm(manualFootprintBBox[1, :]-manualFootprintBBox[2, :])
boxCenter = line2lineIntersection(np.concatenate(manualFootprintBBox[[0, 2], :-1]), np.concatenate(manualFootprintBBox[[1, 3], :-1]))
trajectory = []
for i in np.arange(len(TsToLast)) :
    transformedBoxCenter = np.dot(np.linalg.inv(TsToLast[i]), np.concatenate([boxCenter, [1]])[:, np.newaxis])
    transformedBoxCenter = transformedBoxCenter[:-1, :]/transformedBoxCenter[-1, :]
    trajectory.append(transformedBoxCenter.flatten())
trajectory.append(boxCenter)
trajectory = np.array(trajectory)
directions = trajectory[1:, :]-trajectory[:-1, :]
directions = (directions.T/np.linalg.norm(directions, axis=1)).T
directions = np.vstack([directions, directions[-1:, :]])
directionsAngles = (np.arcsin(np.cross(directions, np.array([[1.0, 0.0]]), axis=1))+np.pi/2)/np.pi
directionsAngles = np.concatenate([directionsAngles, directionsAngles[-1:]])
# figure(); scatter(trajectory[:, 0], trajectory[:, 1], c=cm.jet(directionsAngles, alpha=1))

locSaveVis = "/media/ilisescu/Data1/PhD/data/havana/3dLoppingVis/"
if not os.path.exists(locSaveVis) :
    os.makedirs(locSaveVis)

maxLabel = np.max(masksLabels)
## remember to use the camera intrinsics after undistortion as the world footprints are defined in that frame of reference
worldToCameraT = np.dot(cameraIntrinsics, filmedSceneData[DICT_CAMERA_EXTRINSICS][:-1, [0, 1, 3]])
for frameIdx in np.arange(numNeighboringFrames/2, len(frameLocs)-numNeighboringFrames/2)[240:] :
    frameIm = np.array(Image.open(frameLocs[frameIdx]))    
    
    footprintBoxIm = np.zeros([int(cameraIntrinsics[1, -1]*2), int(cameraIntrinsics[0, -1]*2), 4], frameIm.dtype)
    for blobId in np.sort(worldFootprints.keys()) :
        if frameIdx-numNeighboringFrames/2 in worldFootprints[blobId] :
            tIdx = np.int(np.argwhere(np.sort(worldFootprints[blobId].keys())==frameIdx-numNeighboringFrames/2).flatten())
            if tIdx < len(TsToLast) :
                worldFootprintBox = np.dot(np.linalg.inv(TsToLast[tIdx]), np.concatenate([manualFootprintBox[:, :-1], np.ones([len(manualFootprintBox), 1], float)], axis=1).T)
                worldFootprintBox = (worldFootprintBox[:-1, :]/worldFootprintBox[-1, :]).T
                worldFootprintBox = np.concatenate([worldFootprintBox, np.ones([len(worldFootprintBox), 1], float)], axis=1)
            else :
                worldFootprintBox = np.copy(manualFootprintBox)
                
            cameraFootprintBox = np.dot(worldToCameraT, np.concatenate([worldFootprintBox[:, :-1], np.ones([len(worldFootprintBox), 1], float)], axis=1).T)
            cameraFootprintBox = cameraFootprintBox[:-1, :]/cameraFootprintBox[-1, :]
            cv2.polylines(footprintBoxIm, [cameraFootprintBox[:, [0, 1, 2, 3, 0]].T.astype(np.int32)], False, np.array([255, 0, 0, 255]), thickness=3)
            
            cameraFootprintBoxDirection = np.dot(worldToCameraT, np.concatenate([np.vstack([[trajectory[tIdx, :]], [trajectory[tIdx, :]+directions[tIdx, :]*longEdgeLength/2]]), np.ones([2, 1], float)], axis=1).T)
            cameraFootprintBoxDirection = cameraFootprintBoxDirection[:-1, :]/cameraFootprintBoxDirection[-1, :]
            cv2.polylines(footprintBoxIm, [cameraFootprintBoxDirection.T.astype(np.int32)], False, np.array(cm.jet(directionsAngles[tIdx], alpha=1, bytes=True), dtype=int), thickness=3)
    
    undistortedFrameIm = undistortImage(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO], frameIm, filmedSceneData[DICT_CAMERA_INTRINSICS])[0]
#     figure(); imshow(undistortedFrameIm)
#     figure(); imshow(footprintsIm)
#     figure(); imshow(footprintBoxIm)
#     print(frameIdx)
    Image.fromarray(footprintBoxIm).save(locSaveVis+"footprintBox-frame-{0:05}.png".format(frameIdx+1))


# In[145]:

len(np.arange(numNeighboringFrames/2, len(frameLocs)-numNeighboringFrames/2))


# In[173]:

locSaveVis = "/media/ilisescu/Data1/PhD/data/havana/3dLoppingVis/"
if not os.path.exists(locSaveVis) :
    os.makedirs(locSaveVis)

maxLabel = np.max(masksLabels)
## remember to use the camera intrinsics after undistortion as the world footprints are defined in that frame of reference
worldToCameraT = np.dot(cameraIntrinsics, filmedSceneData[DICT_CAMERA_EXTRINSICS][:-1, [0, 1, 3]])
for frameIdx in np.arange(numNeighboringFrames/2, len(frameLocs)-numNeighboringFrames/2) :
    frameIm = np.array(Image.open(frameLocs[frameIdx]))
#     figure(); imshow(frameIm)
    
    fgMaskIm = mpl.cm.jet(fgMasks[:, :, frameIdx-numNeighboringFrames/2].astype(float), bytes=True)
#     figure(); imshow(fgMaskIm)
    
    maskLabelIm = mpl.cm.Set1(masksLabels[:, :, frameIdx-numNeighboringFrames/2].astype(float)/maxLabel, bytes=True)
    maskLabelIm[masksLabels[:, :, frameIdx-numNeighboringFrames/2] == 0, :] = np.array([0, 0, 0, 255])
#     figure(); imshow(maskLabelIm)
    
    bboxesIm = np.zeros_like(maskLabelIm)
    presentBlobIds = []
    for region in measure.regionprops(masksLabels[:, :, frameIdx-numNeighboringFrames/2]) :
        bbox = np.array(region["bbox"])
        presentBlobIds.append(region["label"])
        cv2.rectangle(bboxesIm, tuple(bbox[0:2][::-1]), tuple(bbox[2:][::-1]-1), np.array([255, 0, 0, 255]), thickness=2)
#     figure(); imshow(bboxesIm)
    
    
    maskLabelsContoursIm = np.zeros_like(maskLabelIm)
    colorsPerBlob = mpl.cm.Set1(np.array(presentBlobIds, float)/maxLabel, bytes=True)
    for colorIdx, blobId in enumerate(presentBlobIds) :
        contours, hierarchy = cv2.findContours((masksLabels[:, :, frameIdx-numNeighboringFrames/2] == blobId).astype(np.uint8)*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for idx, cnt in enumerate(contours) :
            cv2.drawContours(maskLabelsContoursIm, [cnt], 0, tuple(colorsPerBlob[colorIdx, :].astype(int)), 1)
    
#     figure(); imshow(maskLabelsContoursIm)
    
    footprintsIm = np.zeros([int(cameraIntrinsics[1, -1]*2), int(cameraIntrinsics[0, -1]*2), 4], maskLabelIm.dtype)
    for blobId in np.sort(worldFootprints.keys()) :
        if frameIdx-numNeighboringFrames/2 in worldFootprints[blobId] :
#             print(worldFootprints[blobId][frameIdx-numNeighboringFrames/2].shape)
            cameraFootprint = np.dot(worldToCameraT, np.concatenate([worldFootprints[blobId][frameIdx-numNeighboringFrames/2][:, :-1], 
                                                                     np.ones([len(worldFootprints[blobId][frameIdx-numNeighboringFrames/2]), 1], float)], axis=1).T)
            cameraFootprint = cameraFootprint[:-1, :]/cameraFootprint[-1, :]
#             print(cameraFootprint.T.shape, cameraFootprint.dtype)
            cv2.polylines(footprintsIm, [cameraFootprint.T.astype(np.int32)], False, np.array([255, 0, 0, 255]))
    
    undistortedMaskLabelIm = undistortImage(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO], maskLabelIm, filmedSceneData[DICT_CAMERA_INTRINSICS])[0]
    undistortedFrameIm = undistortImage(filmedSceneData[DICT_DISTORTION_PARAMETER], filmedSceneData[DICT_DISTORTION_RATIO], frameIm, filmedSceneData[DICT_CAMERA_INTRINSICS])[0]
#     figure(); imshow(undistortedMaskLabelIm)
#     figure(); imshow(undistortedFrameIm)
#     figure(); imshow(footprintsIm)
    print(frameIdx)
    
    Image.fromarray(frameIm).save(locSaveVis+"frame-{0:05}.png".format(frameIdx+1))
    Image.fromarray(fgMaskIm).save(locSaveVis+"fgMask-frame-{0:05}.png".format(frameIdx+1))
    Image.fromarray(maskLabelIm).save(locSaveVis+"maskLabel-frame-{0:05}.png".format(frameIdx+1))
    Image.fromarray(bboxesIm).save(locSaveVis+"bboxes-frame-{0:05}.png".format(frameIdx+1))
    Image.fromarray(maskLabelsContoursIm).save(locSaveVis+"maskLabelsContours-frame-{0:05}.png".format(frameIdx+1))
    Image.fromarray(undistortedMaskLabelIm).save(locSaveVis+"undistortedMaskLabel-frame-{0:05}.png".format(frameIdx+1))
    Image.fromarray(undistortedFrameIm).save(locSaveVis+"undistortedFrame-frame-{0:05}.png".format(frameIdx+1))
    Image.fromarray(footprintsIm).save(locSaveVis+"footprints-frame-{0:05}.png".format(frameIdx+1))
    
    
# print(len(frameLocs))
# print(fgMasks.shape)
# print(numNeighboringFrames)


# In[34]:

# figure(); imshow(fgMasks[:, :, 0:3])
labels = measure.label(fgMasks[:, :, :5].astype(bool))

figure()
img = None
for i in xrange(labels.shape[-1]):
    if img is None:
        img = mpl.pylab.imshow(labels[:, :, i])
    else:
        img.set_data(labels[:, :, i])
    mpl.pylab.pause(1.0)
    mpl.pylab.draw()


# In[297]:

np.argwhere(masksLabels[:, :, -2] == 51)
figure(); imshow(masksLabels[:, :, -1])


# In[298]:


figure()
img = None
for i in xrange(masksLabels.shape[-1]):
    if img is None:
        img = mpl.pylab.imshow(masksLabels[:, :, i])
    else:
        img.set_data(masksLabels[:, :, i])
    mpl.pylab.pause(1.0)
    mpl.pylab.draw()


# In[41]:

figure(); imshow(newLabels)
figure(); imshow(masksLabels[265:540, 965:1275, 0])
# figure(); imshow(masksLabels[375:540, 1035:1275, 0])
figure(); imshow(labels[265:540, 965:1275, 1])
# figure(); imshow(labels[375:540, 1035:1275, 1])


# In[97]:

# filmedSceneLoc = "/home/ilisescu/PhD/data/havana/filmed_scene-havana.npy"
# filmedSceneData = np.load(filmedSceneLoc).item()
# cameraIntrinsics = filmedSceneData[DICT_CAMERA_INTRINSICS]
# cameraExtrinsics = filmedSceneData[DICT_CAMERA_EXTRINSICS]

# cameraGroundPoints = filmedSceneData[DICT_GROUND_MESH_POINTS]
# worldGroundPoints = np.dot(np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])), np.concatenate([cameraGroundPoints, np.ones([len(cameraGroundPoints), 1])], axis=1).T)
# worldGroundPoints = np.vstack([worldGroundPoints[:-1, :]/worldGroundPoints[-1, :], np.zeros([1, len(cameraGroundPoints)])]).T
# ## now triangulate the points 
# triangles = np.array(triangulate2DPolygon(worldGroundPoints[:, :-1], False))
# gridPoints, validPoints, triangles = getGridPointsInPolygon2D(worldGroundPoints[:, :-1], 1.0)


# In[98]:

# figure()
# scatter(gridPoints[:, 0], gridPoints[:, 1])
# scatter(gridPoints[validPoints, 0], gridPoints[validPoints, 1], c="r")
# for triangle in triangles :
#     plot(triangle[[0, 1, 2, 0], 0], triangle[[0, 1, 2, 0], 1], c="m")
# # plot(worldGroundPoints[:, 0], worldGroundPoints[:, 1])


# In[22]:

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# video_capture = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
#     )

#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()


# In[320]:

# figure(); imshow(medianImage); xlim([0, medianImage.shape[1]]); ylim([medianImage.shape[0], 0])
# plot(trajectoryPoints[:, 0], trajectoryPoints[:, 1], color='r')

# T = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])
# inverseT = np.linalg.inv(T)

# numAngleDivisions = 16
# ## set up object location in 3D world
# forwardDir = np.array([[0.1], [0.0], [0.0], [1.0]]) # in object space

# ## set up grid 
# spacing = float(1.0)
# gridSpace = np.array([-5, 5], float)
# gridPoints = np.mgrid[gridSpace[0]:gridSpace[1]:spacing, gridSpace[0]:gridSpace[1]:spacing]
# gridPoints = gridPoints.reshape([2, gridPoints.shape[1]*gridPoints.shape[2]]).T
# allDirections = []
# allBestMatchScores = []
# for i, loc in enumerate(gridPoints) :
#     print("loc", i, "of", len(gridPoints)); sys.stdout.flush()
#     objPos = np.array([loc[0], loc[1], 0.0])
#     directions = []
#     bestMatchScores = []
#     for angle in np.arange(0, np.pi*2, np.pi/numAngleDivisions*2)[0:] :
#         ## set up orientation of object in 3D world
#     #     print("angle:", angle*180/np.pi)
#         modelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(angle, np.array([0.0, 0.0, -1.0])))
#         modelMat[:-1, -1] = objPos

#         ## find object location and direction in camera space
#         objectPosWorld = np.dot(modelMat, np.array([[0.0], [0.0], [0.0], [1.0]])).flatten()
#         objectPosWorld = objectPosWorld[:-1]/objectPosWorld[-1]
#         objectPosCamera = np.dot(T, np.concatenate([objectPosWorld[:-1], [1.0]])).flatten()
#         objectPosCamera = objectPosCamera[:-1]/objectPosCamera[-1]

#         objectDirPosWorld = np.dot(modelMat, forwardDir).flatten()
#         objectDirPosWorld = objectDirPosWorld[:-1]/objectDirPosWorld[-1]
#         objectDirPosCamera = np.dot(T, np.concatenate([objectDirPosWorld[:-1], [1.0]])).flatten()
#         objectDirPosCamera = objectDirPosCamera[:-1]/objectDirPosCamera[-1]

#         directions.append([objectPosCamera, objectDirPosCamera])

#         ## find best matching frame based on object orientation and existing views from captured scene
#         camPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
#         cameraToObjDir = objPos-camPos
#         cameraToObjDir /= np.linalg.norm(cameraToObjDir)
#         ## in object space from world space
#         cameraPosObjSpace = np.dot(np.linalg.inv(modelMat), np.concatenate([objPos-cameraToObjDir, [1]]).reshape([4, 1])).flatten()
#         cameraPosObjSpace = cameraPosObjSpace[:-1]/cameraPosObjSpace[-1]
#         cameraToObjDir = np.zeros(3)-cameraPosObjSpace
#         cameraToObjDir /= np.linalg.norm(cameraToObjDir)

#         directionAngleDistances = np.abs(np.arccos(np.clip(np.dot(cameraToObjDir.reshape([1, 3]), tmpDirections.T), -1.0, 1.0))*180.0/np.pi)
#         bestMatchScore = np.min(directionAngleDistances).flatten()
#         bestMatchScores.append(bestMatchScore)
#     #     print(bestMatchScore)


#     ## plot all best matching scores for each direction
#     bestMatchScores = np.array(bestMatchScores).flatten()
    
#     allDirections.append(directions)
#     allBestMatchScores.append(bestMatchScores)
    
# allBestMatchScores = np.array(allBestMatchScores)
# allBestMatchScores = allBestMatchScores/np.max(allBestMatchScores)

# for directions, bestMatchScores in zip(allDirections, allBestMatchScores) :
# #     print(np.dot(inverseT, np.concatenate([directions[0][0], [1.0]])).flatten()/np.dot(inverseT, np.concatenate([directions[0][0], [1.0]])).flatten()[-1])
# #     print()
#     for direction, bestMatchScore in zip(directions, bestMatchScores) : 
# #         scatter(direction[0][0], direction[0][1], marker="x", color="yellow")
# #         print(cm.jet(bestMatchScore, bytes=True))
#         plot([direction[0][0], direction[1][0]], [direction[0][1], direction[1][1]], c=cm.jet(bestMatchScore))
# #     print("-----------------")
# # close("all")


# In[349]:

# fig = figure()
# ax = fig.add_subplot(111, aspect='equal', projection='3d')
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)
# numLongitudeLines = 5#numAngleDivisions+1
# numLatitudeLines = 5#11
# u, v = np.mgrid[0:2*np.pi:complex(0, numLongitudeLines), 0:np.pi:complex(0, numLatitudeLines)]
# x=np.cos(u)*np.sin(v)
# y=np.sin(u)*np.sin(v)
# z=np.cos(v)
# vertices = np.array([x.T.flatten(), y.T.flatten(), z.T.flatten()]).T
# ## remove last vertex on each latitude line as it's a duplicate of the first one
# vertices = np.delete(vertices, arange(numLongitudeLines, numLongitudeLines*numLatitudeLines+1, numLongitudeLines)-1, axis=0)
# ## remove all vertices in the first and last tituted lines apart from one
# vertices = vertices[numLongitudeLines-2:-numLongitudeLines+2]
# ## build indices to triangulate the vertices of the sphere
# ## triangles for top lid of the sphere
# indices = np.array([np.array([0, i, j]) for i, j in zip(np.arange(1, numLongitudeLines),
#                                                         np.concatenate([np.arange(2, numLongitudeLines), [1]]))]).flatten()
# ## triangles for each row apart from lids
# firstRowTriangleIndices = np.concatenate([np.array([np.array([0, numLongitudeLines-1, 1, 1, numLongitudeLines-1, numLongitudeLines])+1+i for i in np.arange(0, numLongitudeLines-2)]).flatten(),
#                                           np.array([numLongitudeLines-1, (numLongitudeLines-1)*2, 1, 1, (numLongitudeLines-1)*2, numLongitudeLines])])
# indices = np.concatenate([indices,
#                           np.array([firstRowTriangleIndices+j*(numLongitudeLines-1) for j in arange(0, numLatitudeLines-3)]).flatten()])
# ## triangles for bottom lid
# indices = np.concatenate([indices,
#                           np.array([np.array([len(vertices)-1, j, i]) for i, j in zip(np.arange(len(vertices)-numLongitudeLines, len(vertices)-1),
#                                                                                       np.concatenate([np.arange(len(vertices)-numLongitudeLines+1, len(vertices)-1), [len(vertices)-numLongitudeLines]]))]).flatten()])
# ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="r", linewidth=.2)


# In[15]:

# # %pylab
# filmedSceneLoc = "/home/ilisescu/PhD/data/havana/"
# cameraExtrinsics = np.array([[0.820045839796, 0.57100067645, -0.0385103638868, 1.67922756789],
#                                           [0.22275752409, -0.380450047102, -0.897572753108, -0.831720502302],
#                                           [-0.527165918942, 0.727472328789, -0.439181175316, 6.76268742928],
#                                           [0.0, 0.0, 0.0, 1.0]], np.float32)

# cameraIntrinsics = np.array([[702.736053, 0.0, 640.0],
#                                   [0.0, 702.736053, 360.0],
#                                   [0.0, 0.0, 1.0]])
# originalIntrinsics = np.copy(cameraIntrinsics)

# medianImage = np.array(Image.open(filmedSceneLoc+"median.png"), np.uint8)

# distortionParameter = -0.19
# distortionRatio = -0.19
# medianImage, cameraIntrinsics, distortionCoeff = undistortImage(distortionParameter, distortionRatio, medianImage, cameraIntrinsics)
# figure(); imshow(medianImage)

# usedFrame = np.array(Image.open(filmedSceneLoc+"frame-{0:05d}.png".format(2346+1)), np.uint8)
# figure(); imshow(usedFrame)
# usedFrame = undistortImage(distortionParameter, distortionRatio, usedFrame, originalIntrinsics)[0]
# figure(); imshow(usedFrame)


# # CODE TO ADJUST FOR CAMERA DISTORTIONS USING A BILLBOARD AND HOMOGRAPHIES

# In[531]:

# if False :
#     filmedSceneLoc = "/home/ilisescu/PhD/data/havana/"
# #     spriteIdx = 1; frameSubset = [5, -10]; segmentationThreshold = 0.8; filmedObjectIdx = 0  ## blue_car1
# #     spriteIdx = 7; frameSubset = [5, -10]; segmentationThreshold = 0.8; filmedObjectIdx = 1 ## red_car1
#     spriteIdx = 9; frameSubset = [40, -120]; segmentationThreshold = 0.8; filmedObjectIdx = 2 ## white_bus1
# else :
#     filmedSceneLoc = "/media/ilisescu/Data1/PhD/data/theme_park_sunny/"
#     spriteIdx = 10; frameSubset = [0, -1]; segmentationThreshold = 1.5; filmedObjectIdx = 0 ## person 2
# filmedSceneData = np.load(glob.glob(filmedSceneLoc+"filmed_scene-*.npy")[0]).item()
# cameraExtrinsics = filmedSceneData[DICT_CAMERA_EXTRINSICS].astype(np.float32)

# cameraIntrinsics = filmedSceneData[DICT_CAMERA_INTRINSICS]
# originalIntrinsics = np.copy(cameraIntrinsics)

# medianImage = np.array(Image.open(filmedSceneLoc+"median.png"), np.uint8)
# distortionParameter = filmedSceneData[DICT_DISTORTION_PARAMETER]
# distortionRatio = filmedSceneData[DICT_DISTORTION_RATIO]
# medianImage, cameraIntrinsics, distortionCoeff, _, _ = undistortImage(distortionParameter, distortionRatio, medianImage, cameraIntrinsics)


# if False :
#     ## OLD WAY OF DOING THINGS
#     spriteIdx = 1 ## blue_car1
#     # spriteIdx = 7 ## red_car1
#     # spriteIdx = 9 ## white_bus1
#     objectData = np.load(np.sort(glob.glob(filmedSceneLoc+"semantic_sequence-*.npy"))[spriteIdx]).item()
#     trajectoryPoints = np.array([objectData[DICT_BBOX_CENTERS][key] for key in np.sort(objectData[DICT_BBOX_CENTERS].keys())])[frameSubset[0]:frameSubset[1]-1, :]
#     trajectoryPoints = cv2.undistortPoints(trajectoryPoints.reshape((1, len(trajectoryPoints), 2)), originalIntrinsics, distortionCoeff, P=cameraIntrinsics)[0, :, :]
#     figure(); imshow(medianImage)
#     scatter(trajectoryPoints[:, 0], trajectoryPoints[:, 1])
#     # print(np.sort(objectData[DICT_BBOX_CENTERS].keys())[frameSubset[0]:frameSubset[1]-1])
#     if objectData[DICT_SEQUENCE_NAME] == "blue_car1" or objectData[DICT_SEQUENCE_NAME] == "red_car1" or objectData[DICT_SEQUENCE_NAME] == "white_bus1":
#         print("READING NUKE TRACK FROM", "/home/ilisescu/PhD/data/havana/{0}-track.txt".format(objectData[DICT_SEQUENCE_NAME]))
#         f = open("/home/ilisescu/PhD/data/havana/{0}-track.txt".format(objectData[DICT_SEQUENCE_NAME]), 'r')
#         lines = f.readlines()
#         vals = [np.array(i.split(" ")).astype(float) for i in lines]
#         vals = [(int(i[-1]), i[0:2]) for i in vals]
#         tmp = dict(vals)
#         trajectoryPoints = np.array([tmp[key] for key in np.sort(objectData[DICT_BBOX_CENTERS].keys())[frameSubset[0]:frameSubset[1]-1]])
#     #     trajectoryPoints = trajectoryPoints + (np.array(medianImage.shape[0:2])[::-1].reshape([1, 2]) - np.array(np.array(Image.open(filmedSceneLoc+"median.png"), np.uint8).shape[0:-1])[::-1].reshape([1, 2]))/2
#         ## the nuke tracks are defined in the original camera's coordinate system but after the undistortion
#         trajectoryPoints = trajectoryPoints + cameraIntrinsics[:2, -1] - originalIntrinsics[:2, -1]
#     # print(np.sort(objectData[DICT_BBOX_CENTERS].keys())[frameSubset[0]:frameSubset[1]-1])
# else :
#     print("LOADING:", np.sort(glob.glob(filmedSceneLoc+"semantic_sequence-*.npy"))[spriteIdx])
#     objectData = np.load(np.sort(glob.glob(filmedSceneLoc+"semantic_sequence-*.npy"))[spriteIdx]).item()
#     print("LOADING:", np.sort(glob.glob(filmedSceneLoc+"filmed_object-*.npy"))[filmedObjectIdx])
#     filmedObjectData = np.load(np.sort(glob.glob(filmedSceneLoc+"filmed_object-*.npy"))[filmedObjectIdx]).item()
#     f = open(filmedObjectData[DICT_TRACK_LOCATION], 'r')
#     lines = f.readlines()
#     vals = [np.array(i.split(" ")).astype(float) for i in lines]
#     vals = [(int(i[-1]), i[0:2]) for i in vals]
#     tmp = dict(vals)
#     sortedFrameKeys = np.sort(tmp.keys())[frameSubset[0]:frameSubset[1]-1]
#     trajectoryPoints = np.array([tmp[key] for key in np.sort(tmp.keys())[frameSubset[0]:frameSubset[1]-1]])
# #     trajectoryPoints = filmedObjectData[DICT_TRAJECTORY_POINTS][frameSubset[0]:frameSubset[1]-1]
#     trajectoryPoints = trajectoryPoints + cameraIntrinsics[:2, -1] - originalIntrinsics[:2, -1]
#     print(len(trajectoryPoints))

# figure(); imshow(medianImage)
# scatter(trajectoryPoints[:, 0], trajectoryPoints[:, 1], color='r')
# scatter(trajectoryPoints[210, 0], trajectoryPoints[210, 1], color='b', marker="x")
    
# trajectory = GLTrajectory(trajectoryPoints, cameraIntrinsics, cameraExtrinsics, objectData[DICT_REPRESENTATIVE_COLOR], doSmoothing=False)

# preloadedPatches = np.load(objectData[DICT_PATCHES_LOCATION]).item()


# In[464]:

# def searchWorldSpacePoint(worldTargetPoint, worldStartPoint, cameraStartPoint, worldDir, verbose=False) :
#     cameraClosestPointToIntersection = np.zeros([1, 2])
#     worldClosestPointToIntersection = np.zeros([1, 3])
#     ## init ##
#     pointDist = 1600.0
#     dirRatio = 1.0
#     increment = np.copy(dirRatio)
# #     print("STARTING POINT", worldTargetPoint, "...")
#     worldCurrentPoint = worldStartPoint+worldDir*dirRatio
#     foundInside = False
#     iterNum = 0 
#     while pointDist > 0.1 and iterNum < 100 :
#         iterNum += 1
#         cameraCurrentPoint = worldToScreenSpace(viewMat, projectionMat, worldCurrentPoint, width, height)
#         cameraClosestPointToIntersection[0, :] = cameraCurrentPoint
#         worldClosestPointToIntersection[0, :] = worldCurrentPoint

#         dotProduct = np.dot(cameraCurrentPoint-cameraStartPoint, worldTargetPoint-cameraStartPoint)
#         squaredDist = np.linalg.norm(cameraCurrentPoint-cameraStartPoint)**2

#         if verbose :
#             print(dotProduct, squaredDist, np.linalg.norm(cameraCurrentPoint-worldTargetPoint))
#         ## flip dirRatio direction if the worldTargetPoint is outside of the line segment (cameraCurrentPoint-cameraStartPoint) in the direction of cameraStartPoint
#         if dotProduct < 0 :
#             if verbose :
#                 print("FLIPPING")
#             dirRatio *= -1
#             increment *= -1
#             worldCurrentPoint = worldStartPoint+worldDir*dirRatio
#             continue

#         ## if worldTargetPoint is within the line segment then set the increment to half the current length and set currentPoint to the middle of the half segment closest to cameraStartPoint
#         if dotProduct < squaredDist :
#             increment *= 0.5
#             foundInside = True
#             dirRatio -= increment
#         ## if the worldTargetPoint is outside the line segment
#         else :
#             ## set the increment to half the current length only if the worldTargetPoint has been within the line segment (otherwise don't split but keep increasing the length of the line segment I'm looking within)
#             if foundInside :
#                 increment *= 0.5
#             ## if foundInside == True this sets currentPoint to the middle of the half segment furthest from cameraStartPoint, otherwise it doubles the length of the line segment
#             dirRatio += increment

#         if verbose :
#             print("DIR RATIO", dirRatio, increment, foundInside)

#         worldCurrentPoint = worldStartPoint+worldDir*dirRatio

#         pointDist = np.linalg.norm(cameraCurrentPoint-worldTargetPoint)
#     if iterNum >= 100 :
#         print("...REACHED MAXIMUM ITER COUNT")
# #     else :
# #         print("...DONE")

#     return worldClosestPointToIntersection, cameraClosestPointToIntersection

# def findBillboardSize(worldPos, worldOrientDir, worldUpDir, projectionMat, viewMat, worldToCameraHomography, patchData, verbose=False, doReturnExtraInfo=False) :    
#     worldOrientDirPos = worldPos + worldOrientDir

#     ## find projections of world coords into camera space
#     cameraPos = worldToScreenSpace(viewMat, projectionMat, worldPos, width, height)
#     cameraOrientDirPos = worldToScreenSpace(viewMat, projectionMat, worldOrientDirPos, width, height)
    
#     worldUpDirPos = worldPos+worldUpDir
#     cameraUpDirPos = worldToScreenSpace(viewMat, projectionMat, worldUpDirPos, width, height)


#     ########### FIND BILLBOARD WIDTH BASED ON HOW IT PROJECTS INTO SCREEN SPACE AND HOW IT THEN RELATES WITH THE SEGMENTED PATCH ###########
#     cameraDirLeftIntersection = line2lineIntersection(np.array([cameraPos, cameraOrientDirPos]).flatten(),
#                                                       np.array([patchData['top_left_pos'][::-1], np.array([patchData['top_left_pos'][0]+patchData['patch_size'][0], patchData['top_left_pos'][1]])[::-1]]).flatten())

#     cameraDirRightIntersection = line2lineIntersection(np.array([cameraPos, cameraOrientDirPos]).flatten(),
#                                                        np.array([(patchData['top_left_pos']+patchData['patch_size'])[::-1],
#                                                                  np.array([patchData['top_left_pos'][0], patchData['top_left_pos'][1]+patchData['patch_size'][1]])[::-1]]).flatten())

#     worldDirLeftIntersection = np.dot(np.linalg.inv(worldToCameraHomography), np.concatenate([cameraDirLeftIntersection, [1]]).reshape([3, 1])).flatten()
#     worldDirLeftIntersection /= worldDirLeftIntersection[-1]
#     worldDirLeftIntersection[-1] = 0

#     worldDirRightIntersection = np.dot(np.linalg.inv(worldToCameraHomography), np.concatenate([cameraDirRightIntersection, [1]]).reshape([3, 1])).flatten()
#     worldDirRightIntersection /= worldDirRightIntersection[-1]
#     worldDirRightIntersection[-1] = 0

#     billboardWidth = np.max([np.linalg.norm(worldPos-worldDirLeftIntersection), np.linalg.norm(worldPos-worldDirRightIntersection)])*2

#     ########### FIND BILLBOARD HEIGHT IN A SIMILAR MANNER TO ITS WIDTH ###########
#     cameraUpDirTopIntersection = line2lineIntersection(np.array([cameraPos, cameraUpDirPos]).flatten(),
#                                                        np.array([patchData['top_left_pos'][::-1], np.array([patchData['top_left_pos'][0], patchData['top_left_pos'][1]+patchData['patch_size'][1]])[::-1]]).flatten())

#     cameraUpDirBottomIntersection = line2lineIntersection(np.array([cameraPos, cameraUpDirPos]).flatten(),
#                                                           np.array([(patchData['top_left_pos']+patchData['patch_size'])[::-1],
#                                                                     np.array([patchData['top_left_pos'][0]+patchData['patch_size'][0], patchData['top_left_pos'][1]])[::-1]]).flatten())
#     ## to find the height I can't do the same thing as I did for the width because I can't project screen space points back into the world (previously I could because I knew the points were on the ground plane)
#     ## instead, do a binary search type thing along the worldUpDir to find the world space points that project closest to the screen space points found above (i.e cameraUpDirTopIntersection and cameraUpDirBottomIntersection)

#     worldClosestPointsToIntersection = np.empty([0, 3])
#     cameraClosestPointsToIntersection = np.empty([0, 2])
#     for i, point in enumerate([cameraUpDirTopIntersection, cameraUpDirBottomIntersection]) :
#         worldClosestPointToIntersection, cameraClosestPointToIntersection = searchWorldSpacePoint(point, worldPos, cameraPos, worldUpDir, verbose)
#         worldClosestPointsToIntersection = np.concatenate([worldClosestPointsToIntersection, worldClosestPointToIntersection], axis=0)
#         cameraClosestPointsToIntersection = np.concatenate([cameraClosestPointsToIntersection, cameraClosestPointToIntersection], axis=0)


#     billboardHeight = np.max([np.linalg.norm(worldPos-worldClosestPointsToIntersection[0, :]), np.linalg.norm(worldPos-worldClosestPointsToIntersection[1, :])])*2
    
#     if doReturnExtraInfo :
#         return (billboardWidth, billboardHeight, cameraPos, cameraOrientDirPos, cameraUpDirPos, cameraDirLeftIntersection,
#                 cameraDirRightIntersection, cameraUpDirTopIntersection, cameraUpDirBottomIntersection, cameraClosestPointsToIntersection)
#     else :
#         return billboardWidth, billboardHeight
    

# def getUndistortedPatchDataWithThresholdedAlpha(patchImageLoc, patchAlphaLoc, medianImage, distortionParameter, distortionRatio, cameraIntrinsics, threshold=0.8) :
#     patchImage = np.array(patchImageLoc).astype(np.uint8)
#     patchImage = undistortImage(distortionParameter, distortionRatio, patchImage, cameraIntrinsics)[0]
#     patchAlpha = np.array(patchAlphaLoc).astype(np.uint8)
#     patchAlpha = undistortImage(distortionParameter, distortionRatio, patchAlpha, cameraIntrinsics)[0][:, :, -1]
    
#     ## threshold the alpha based on bg diff 
#     if True :
#         visiblePixels = np.argwhere(patchAlpha != 0)
#         diffs = np.sqrt(np.sum((patchImage[visiblePixels[:, 0], visiblePixels[:, 1], :] - medianImage[visiblePixels[:, 0], visiblePixels[:, 1], :])**2, axis=1))
#         tmp = np.zeros([medianImage.shape[0], medianImage.shape[1]], np.uint8)
#         tmp[visiblePixels[:, 0], visiblePixels[:, 1]] = diffs
#         tmp = cv2.medianBlur(tmp, 7)
#         tmp = tmp/float(np.max(tmp))
#         med = np.median(tmp[visiblePixels[:, 0], visiblePixels[:, 1]])
#         tmp[tmp<med*threshold] = 0
#         tmp[tmp>0] = np.max(tmp)
#         patchAlpha = (tmp/np.max(tmp)*255).astype(np.uint8)
        

#     visiblePixels = np.argwhere(patchAlpha != 0)
#     topLeft = np.min(visiblePixels, axis=0)
#     patchSize = np.max(visiblePixels, axis=0) - topLeft + 1

#     colors = np.concatenate([patchImage[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)

#     patchData = {'top_left_pos':topLeft, 'sprite_colors':colors[:, [2, 1, 0, 3]],
#                  'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}
    
#     return patchImage, patchAlpha, patchData


# In[532]:

# ## find max billboard size
# height, width = medianImage.shape[:2]
# viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, medianImage.shape)
# verbose = False

# maxBillboardWidth = 0
# maxBillboardHeight = 0
# usedOrientationsAndSizes = {}
# for frameIdx in arange(len(trajectoryPoints))[0:] :
#     patchKey = sortedFrameKeys[frameIdx]
#     patchFrameName = "frame-{0:05}.png".format(patchKey+1)
#     patchImage, patchAlpha, patchData = getUndistortedPatchDataWithThresholdedAlpha(Image.open(filmedSceneLoc+patchFrameName), Image.open(objectData[DICT_MASK_LOCATION]+patchFrameName),
#                                                                                     medianImage, distortionParameter, distortionRatio, originalIntrinsics, segmentationThreshold)
    
#     ## find billboard width and height: first align it using tangent and then normal of the trajectory, and keep track of which orientation dir give the billboard with smallest area along with width and height
#     worldPos = np.copy(trajectory.worldTrajectoryPoints[frameIdx, :])
#     worldMovingDir = np.copy(trajectory.worldTrajectoryDirections[frameIdx, :])
#     worldMoveNormalDir = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2, np.array([0.0, 0.0, 1.0]))), np.concatenate([worldMovingDir, [1]]))
#     worldMoveNormalDir = worldMoveNormalDir[:-1]/np.linalg.norm(worldMoveNormalDir[:-1])
#     currentBillboardArea = 10000000000.0
#     for currentWorldOrientDir in [worldMoveNormalDir, worldMovingDir] :
#         ## check if the projection of the orientation direction into image space has a positive x coordinate; if it doesn't, I need to flip the orientation direction, which will produce the same exact billboard, but
#         ## it will ensure it projects into image space with the front face visible
#         currentDirMultiplier = 1.0
#         if (worldToScreenSpace(viewMat, projectionMat, worldPos + currentWorldOrientDir, width, height)-worldToScreenSpace(viewMat, projectionMat, worldPos, width, height))[0] < 0 :
#             currentWorldOrientDir *= -1.0
#             currentDirMultiplier = -1.0
#         worldUpDir = np.array([0.0, 0.0, 1.0])
#         currentBillboardWidth, currentBillboardHeight = findBillboardSize(worldPos, currentWorldOrientDir, worldUpDir, projectionMat, viewMat, np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]), patchData, verbose)
    
#         if currentBillboardWidth*currentBillboardHeight < currentBillboardArea :
# #             print("NEW AREA", currentBillboardWidth*currentBillboardHeight, currentBillboardWidth, currentBillboardHeight, currentWorldOrientDir)
#             currentBillboardArea = currentBillboardWidth*currentBillboardHeight
#             billboardWidth = currentBillboardWidth
#             billboardHeight = currentBillboardHeight
#             worldOrientDir = currentWorldOrientDir
#         ## I'm really only interested in saving the dirMultiplier for the tangent direction, worldMovingDir
#         dirMultiplier = currentDirMultiplier
            
#     usedOrientationsAndSizes[frameIdx] = [worldOrientDir, billboardWidth, billboardHeight, dirMultiplier]
#     if billboardWidth > maxBillboardWidth :
#         maxBillboardWidth = billboardWidth
#         print("NEW MAX WIDTH", frameIdx, maxBillboardWidth)
#     if billboardHeight > maxBillboardHeight :
#         maxBillboardHeight = billboardHeight
#         print("NEW MAX HEIGHT", frameIdx, billboardHeight)
#     print(frameIdx, patchKey, billboardWidth, billboardHeight, maxBillboardWidth, maxBillboardHeight)


# In[513]:

# ## once max billboard size is found, perform the undistortion thingy and such using the thingy
# updatedPatches = {}  
# for frameIdx in np.sort(usedOrientationsAndSizes.keys())[[213]] :
#     patchKey = sortedFrameKeys[frameIdx]
#     patchFrameName = "frame-{0:05}.png".format(patchKey+1)
#     patchImage, patchAlpha, patchData = getUndistortedPatchDataWithThresholdedAlpha(Image.open(filmedSceneLoc+patchFrameName), Image.open(objectData[DICT_MASK_LOCATION]+patchFrameName),
#                                                                                     medianImage, distortionParameter, distortionRatio, originalIntrinsics, segmentationThreshold)
    
    
#     ### HERE NEED TO ACTUALLY CHANGE worldOrientDir SO THAT I PICK THE SMALLEST AREA BILLBOARD ###
#     worldPos = np.copy(trajectory.worldTrajectoryPoints[frameIdx, :])
#     worldOrientDir, billboardWidth, billboardHeight, dirMultiplier = usedOrientationsAndSizes[frameIdx]
    
# #     ## find billboard width and height --> keep track of worldOrientDir and width and height in the loop above instead of recomputing width and height    
    
# #     worldOrientDir = np.copy(trajectory.worldTrajectoryDirections[frameIdx, :])
# #     worldOrientDir = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2, np.array([0.0, 0.0, 1.0]))), np.concatenate([worldOrientDir, [0]]))
# #     worldOrientDir = worldOrientDir[:-1]/np.linalg.norm(worldOrientDir[:-1])
# #     ## check if the projection of the orientation direction into image space has a positive x coordinate; if it doesn't, I need to flip the orientation direction, which will produce the same exact billboard, but
# #     ## it will ensure it projects into image space with the front face visible
# #     if (worldToScreenSpace(viewMat, projectionMat, worldPos + worldOrientDir, width, height)-worldToScreenSpace(viewMat, projectionMat, worldPos, width, height))[0] < 0 :
# #         worldOrientDir *= -1.0
# #     worldUpDir = np.array([0.0, 0.0, 1.0])
# #     billboardWidth, billboardHeight = findBillboardSize(worldPos, worldOrientDir, worldUpDir, projectionMat, viewMat, np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]), patchData, verbose)
    
#     ###############################################################################################
    
    
#     ########### ROTATE BILLBOARD TO ALIGN WITH MOVING DIRECTION AND PLACE AT POINT ON TRAJECTORY ###########
#     billboardModelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(-np.pi/2.0, np.array([1.0, 0.0, 0.0]))) ## rotate billboard ccw along x axis to put it up
#     adjustAngle = np.arccos(np.clip(np.dot(np.array([1.0, 0.0, 0.0]), worldOrientDir), -1, 1))
#     adjustAxis = np.cross(worldOrientDir, np.array([1.0, 0.0, 0.0]))
#     adjustAxis /= np.linalg.norm(adjustAxis)
#     billboardModelMat = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis)), billboardModelMat)
#     billboardModelMat[:-1, -1] = worldPos

#     billboard = GLBillboard(np.zeros([500, np.ceil(500*billboardWidth/billboardHeight), 4], np.uint8), billboardHeight, modelMat=billboardModelMat)

#     worldBillboardVertices = np.dot(billboard.modelMat, np.concatenate([billboard.vertices, np.ones([len(billboard.vertices), 1])], axis=1).T).T[:, :-1]

#     ## this could be done with one matrix computation but cba right now
#     screenBillboardVertices = np.zeros([len(worldBillboardVertices), 2])
#     for i, vertex in enumerate(worldBillboardVertices) :
#         screenBillboardVertices[i, :] = worldToScreenSpace(viewMat, projectionMat, vertex, width, height)
#     #     print(vertex, screenBillboardVertices[i, :])

#     footprintBillboardModelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))
#     footprintBillboardModelMat[:-1, -1] = worldPos
#     ## this is for the havana cars
#     footprintBillboard = GLBillboard(np.zeros([1000*0.081, 1000*0.18, 4], np.uint8), 0.265, modelMat=footprintBillboardModelMat)
#     worldFootprintBillboardVertices = np.dot(footprintBillboard.modelMat, np.concatenate([footprintBillboard.vertices, np.ones([len(footprintBillboard.vertices), 1])], axis=1).T).T[:, :-1]
#     ## this could be done with one matrix computation but cba right now
#     screenFootprintBillboardVertices = np.zeros([len(worldFootprintBillboardVertices), 2])
#     for i, vertex in enumerate(worldFootprintBillboardVertices) :
#         screenFootprintBillboardVertices[i, :] = worldToScreenSpace(viewMat, projectionMat, vertex, width, height)
        
#     texHeight = 0
#     texWidth = 0
#     if maxBillboardWidth > maxBillboardHeight :
#         texWidth = 512
#         texHeight = texWidth*maxBillboardHeight/maxBillboardWidth
#     else :
#         texHeight = 512
#         texWidth = texHeight*maxBillboardWidth/maxBillboardHeight#*billboardWidth/billboardHeight
        
#     ######################### SCALE COMPENSATION BASED ON BILLBOARD WIDTH IN WORLD SPACE #########################
#     if True :
#         widthScale = billboardWidth/maxBillboardWidth
#         heightScale = billboardHeight/maxBillboardHeight
#         scaledTexHeight = texHeight*heightScale
#         scaledTexWidth = scaledTexHeight*billboardWidth/billboardHeight#texWidth*widthScale
#         print(frameIdx, scaledTexWidth, scaledTexHeight)
#     ######################### SCALE COMPENSATION BASED ON WIDTH OF PATCH BILLBOARD PROJECTS TO IN CAMERA SPACE ######################### (PROBABLY OBSOLETE NOW)
#     else :
#         billboardWidth = np.max(screenFootprintBillboardVertices[:, 0]) - np.min(screenFootprintBillboardVertices[:, 0])
#         billboardHeight = np.max(screenFootprintBillboardVertices[:, 1]) - np.min(screenFootprintBillboardVertices[:, 1])
#         patchScale = billboardWidth/maxBillboardWidth
#         print(frameIdx, billboardWidth, maxBillboardWidth, patchScale)
#         scaledTexHeight = texHeight*patchScale
#         scaledTexWidth = texWidth*patchScale
    
#     ## when defining the rectangle in texture space, need to make sure that it uses the same conventions as screenBillboardVertices, which in this case, it means y goes down and x goes left
#     billboardHomography = cv2.findHomography(screenBillboardVertices[[0, 2, 4, 1], :], np.array([[texWidth-(texWidth-scaledTexWidth)/2.0, texHeight-(texHeight-scaledTexHeight)/2.0],
#                                                                                                  [(texWidth-scaledTexWidth)/2.0, texHeight-(texHeight-scaledTexHeight)/2.0],
#                                                                                                  [(texWidth-scaledTexWidth)/2.0, (texHeight-scaledTexHeight)/2.0],
#                                                                                                  [texWidth-(texWidth-scaledTexWidth)/2.0, (texHeight-scaledTexHeight)/2.0]], dtype=float))[0]
#     tmp = cv2.warpPerspective(np.concatenate([patchImage, patchAlpha.reshape([patchAlpha.shape[0], patchAlpha.shape[1], 1])], axis=-1), billboardHomography, (int(np.ceil(texWidth)), int(np.ceil(texHeight))))
# #     figure(); imshow(tmp)
# #     print(tmp.shape, billboardWidth, billboardHeight, texWidth)
# #     print(screenBillboardVertices)

# #     print(frameIdx)
    
#     visiblePixels = np.argwhere(tmp[:, :, -1] != 0)
    
#     colors = np.concatenate([tmp[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)

#     updatedPatches[patchKey] = {'top_left_pos':np.zeros(2, int), 'sprite_colors':colors[:, [2, 1, 0, 3]],
#                                 'visible_indices': visiblePixels, 'patch_size': np.array([int(np.ceil(texHeight)), int(np.ceil(texWidth))], int)}


# In[519]:

# print(texWidth, texHeight, billboardWidth, billboardHeight, maxBillboardWidth, maxBillboardHeight)
# tmpBillboard = GLBillboard(np.zeros([texHeight, texWidth, 4], np.uint8), 1.0, modelMat=billboardModelMat)
# tmpWorldBillboardVertices = np.dot(tmpBillboard.modelMat, np.concatenate([tmpBillboard.vertices, np.ones([len(tmpBillboard.vertices), 1])], axis=1).T).T[:, :-1]
# ## this could be done with one matrix computation but cba right now
# tmpScreenBillboardVertices = np.zeros([len(tmpWorldBillboardVertices), 2])
# for i, vertex in enumerate(tmpWorldBillboardVertices) :
#     tmpScreenBillboardVertices[i, :] = worldToScreenSpace(viewMat, projectionMat, vertex, width, height)
    
# tmpBillboardHomography = cv2.findHomography(np.array([[texWidth, texHeight], [0, texHeight], [0, 0], [texWidth, 0]], dtype=float), tmpScreenBillboardVertices[[0, 2, 4, 1], :])[0]

# tmp = cv2.warpPerspective(patchColors, tmpBillboardHomography, (int(compositedImage.shape[1]), int(compositedImage.shape[0])))
# figure(); imshow(tmp)
# figure(); imshow(compositedImage)
# xlim([0, medianImage.shape[1]])
# ylim([medianImage.shape[0], 0])
# plot(tmpScreenBillboardVertices[[0, 1, 4, 2, 0], 0], tmpScreenBillboardVertices[[0, 1, 4, 2, 0], 1], color='magenta', linewidth=2)


# In[510]:

# print(np.linalg.norm(tmpWorldBillboardVertices[0, :]-tmpWorldBillboardVertices[2, :]))
# print(maxBillboardWidth/np.linalg.norm(tmpWorldBillboardVertices[0, :]-tmpWorldBillboardVertices[2, :]))


# In[520]:

# print(np.linalg.norm(tmpWorldBillboardVertices[4, :]-tmpWorldBillboardVertices[2, :]))
# print(maxBillboardHeight/np.linalg.norm(tmpWorldBillboardVertices[4, :]-tmpWorldBillboardVertices[2, :]))


# In[515]:

# patchData = updatedPatches[np.sort(updatedPatches.keys())[-1]]
# patchColors = np.zeros(np.concatenate([patchData['patch_size'], [4]]), dtype=np.uint8)
# patchColors[patchData['visible_indices'][:, 0], patchData['visible_indices'][:, 1], :] = patchData['sprite_colors'][:, [2, 1, 0, 3]]
# figure(); imshow(patchColors)


# In[514]:

# ### VISUALIZE BILLBOARD AND STUFF ###
# (billboardWidth, billboardHeight, cameraPos, cameraOrientDirPos, cameraUpDirPos, cameraDirLeftIntersection,
#  cameraDirRightIntersection, cameraUpDirTopIntersection, cameraUpDirBottomIntersection, cameraClosestPointsToIntersection) = findBillboardSize(worldPos, worldOrientDir, worldUpDir, projectionMat,
#                                                                                                                                                viewMat, np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]),
#                                                                                                                                                patchData, verbose, True)
# compositedImage = np.copy(medianImage)
# patchColors = np.zeros(np.concatenate([patchData['patch_size'], [4]]), dtype=np.uint8)
# patchColors[patchData['visible_indices'][:, 0], patchData['visible_indices'][:, 1], :] = patchData['sprite_colors'][:, [2, 1, 0, 3]]

# compositedImage[patchData['top_left_pos'][0]:patchData['top_left_pos'][0]+patchData['patch_size'][0],
#                 patchData['top_left_pos'][1]:patchData['top_left_pos'][1]+patchData['patch_size'][1], :] = (compositedImage[patchData['top_left_pos'][0]:patchData['top_left_pos'][0]+patchData['patch_size'][0],
#                                                                                                                             patchData['top_left_pos'][1]:patchData['top_left_pos'][1]+patchData['patch_size'][1], :]*
#                                                                                                            (1.0-patchColors[:, :, -1]/255.0).reshape([patchColors.shape[0], patchColors.shape[1], 1]) +
#                                                                                                            patchColors[:, :, :-1]*(patchColors[:, :, -1]/255.0).reshape([patchColors.shape[0], patchColors.shape[1], 1]))


# figure(); imshow(compositedImage)
# xlim([0, medianImage.shape[1]])
# ylim([medianImage.shape[0], 0])
# # xlim([(medianImage.shape[1]-1280)/2, 1280+(medianImage.shape[1]-1280)/2])
# # ylim([720+(medianImage.shape[0]-720)/2, (medianImage.shape[0]-720)/2])

# scatter(trajectory.cameraTrajectoryPoints[:, 0], trajectory.cameraTrajectoryPoints[:, 1], color=tuple(trajectory.drawColor/255.0), marker='o', facecolors='none', s=90)
# scatter(trajectoryPoints[:, 0], trajectoryPoints[:, 1], color='red', marker='x', s=90)
# plot([cameraPos[0], cameraOrientDirPos[0]], [cameraPos[1], cameraOrientDirPos[1]], color='yellow', linewidth=2)
# plot([cameraPos[0], cameraUpDirPos[0]], [cameraPos[1], cameraUpDirPos[1]], color='yellow', linewidth=2)
# plot([patchData['top_left_pos'][1], patchData['top_left_pos'][1], patchData['top_left_pos'][1]+patchData['patch_size'][1],
#       patchData['top_left_pos'][1]+patchData['patch_size'][1], patchData['top_left_pos'][1]], [patchData['top_left_pos'][0], patchData['top_left_pos'][0]+patchData['patch_size'][0],
#                                                                                                patchData['top_left_pos'][0]+patchData['patch_size'][0], patchData['top_left_pos'][0],
#                                                                                                patchData['top_left_pos'][0]], color='red', linewidth=2)

# scatter([cameraDirLeftIntersection[0], cameraDirRightIntersection[0], cameraUpDirTopIntersection[0], cameraUpDirBottomIntersection[0]],
#         [cameraDirLeftIntersection[1], cameraDirRightIntersection[1], cameraUpDirTopIntersection[1], cameraUpDirBottomIntersection[1]], color='blue', s=90)
# scatter([cameraClosestPointsToIntersection[:, 0]], [cameraClosestPointsToIntersection[:, 1]], color="yellow", marker="x", s=90)
# plot(screenBillboardVertices[[0, 1, 4, 2, 0], 0], screenBillboardVertices[[0, 1, 4, 2, 0], 1], color='magenta', linewidth=2)
# plot(screenFootprintBillboardVertices[[0, 1, 4, 2, 0], 0], screenFootprintBillboardVertices[[0, 1, 4, 2, 0], 1], color='cyan', linewidth=2)

# # xlim([patchData['top_left_pos'][1]-patchData['patch_size'][1]*0.5, patchData['top_left_pos'][1]+patchData['patch_size'][1]*1.5])
# # ylim([patchData['top_left_pos'][0]+patchData['patch_size'][0]*1.5, patchData['top_left_pos'][0]-patchData['patch_size'][0]*0.5])


# In[534]:

# # SAVE THE UPDATED PATCHES AND UPDATE THE FILMED OBJECT DICTIONARY TO INCLUDE THE CORRECT PATH AND ORIENTATION ANGLES
# patchesName = "thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-"
# updatedPatchesLoc = filmedSceneData[DICT_FILMED_SCENE_BASE_LOC]+patchesName+"{}.npy".format(filmedObjectData[DICT_FILMED_OBJECT_NAME])
# np.save(updatedPatchesLoc, updatedPatches)
# filmedObjectData[DICT_PATCHES_LOCATION] = updatedPatchesLoc

# orientationAngles = np.zeros(len(usedOrientationsAndSizes))
# for frameIdx in np.arange(len(usedOrientationsAndSizes)) :
#     worldPos = np.copy(trajectory.worldTrajectoryPoints[frameIdx, :])
#     worldMovingDir = np.copy(trajectory.worldTrajectoryDirections[frameIdx, :])
#     worldMoveNormalDir = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2, np.array([0.0, 0.0, 1.0]))), np.concatenate([worldMovingDir, [1]]))
#     worldMoveNormalDir = worldMoveNormalDir[:-1]/np.linalg.norm(worldMoveNormalDir[:-1])
#     worldOrientDir, billboardWidth, billboardHeight, dirMultiplier = usedOrientationsAndSizes[frameIdx]
#     ## check if the chosen orienation direction is the opposite of the moving direction in which case, the desired angle should be 0.0 (I will add the 180.0 to flip it later on when checking the adjust axis)
#     if np.sum(np.sqrt((worldMovingDir+worldOrientDir)**2)) < 1e-10 :
#         adjustAngle = 0.0
#     else :
#         adjustAngle = np.arccos(np.clip(np.dot(worldMovingDir, worldOrientDir), -1, 1))
#     ## if the adjust angle is not 0, it means I'm picking the normal to the trajectory as orientation and thus I can compute the adjust axis (which will tell me if I need to flip)
#     if adjustAngle != 0.0 :
#         adjustAxis = np.cross(worldOrientDir, worldMovingDir)
#         adjustAxis /= np.linalg.norm(adjustAxis)
#     ## otherwise the adjust axis depends on the dirMultiplier which tells me whether I need to flip the billboard oriented using the tangent to the trajectory
#     else :
#         adjustAxis = np.array([0.0, 0.0, dirMultiplier])
#     orientationAngles[frameIdx] = adjustAngle
#     ## I can achieve the same rotation as using the flipped z axis, by adding 180 degrees to the current rotation
#     if adjustAxis[-1] < 0.0 :
#         orientationAngles[frameIdx] += np.pi
# filmedObjectData[DICT_OBJECT_BILLBOARD_ORIENTATION] = orientationAngles
# ## the scale of the billboard is the same as maxBillboardHeight because the way I define billboards in my app is by keeping the height fixed to 1 and setting the width based on the aspect ratio of the 
# ## texture; the size of the texture is based on maxBillboardHeight and maxBillboardWidth and all patches are mapped to the exact same size texture; then I map the smallest billboard size for each frame to a rectangle
# ## centerd within the texture, with the same aspect ratio and scaled down to make sure that its height is the same relative to maxBillboardHeight; it's analogous to instead always mapping the max size billboard to the 
# ## texture rectangle (which would probably be easier in retrospect); the problem is then that I always map a billboard of height maxBillboardHeight and correct aspect ratio, to the texture rectangle with the same 
# ## aspect ratio, but if I just use the default GLBillboard, I artificially map the texture onto a bigger or smaller billboard than the one used when it was computed, resulting in bigger or smaller sprites; to solve this
# ## then, all I need to do is scale the whole GLBillboard using maxBillboardHeight which will uniformly scale the full billboard to have the correct height and aspect ratio; alternatively, I could have computed the
# ## max billboard size to get the aspect ratio and then used a billboard of height=1 and of width=ratio and mapped that to the texture rectangle and then I wouldn't have had to scale the GLBillboard
# filmedObjectData[DICT_OBJECT_BILLBOARD_SCALE] = maxBillboardHeight
# print(maxBillboardHeight)


# filmedObjectData[DICT_TRAJECTORY_POINTS] = trajectoryPoints + originalIntrinsics[:2, -1] - cameraIntrinsics[:2, -1]
# np.save(np.sort(glob.glob(filmedSceneLoc+"filmed_object-*.npy"))[filmedObjectIdx], filmedObjectData)


# In[389]:

# print(len(np.load(np.sort(glob.glob(filmedSceneLoc+"filmed_object-*.npy"))[filmedObjectIdx]).item()[DICT_TRAJECTORY_POINTS]))
# print(np.load(np.sort(glob.glob(filmedSceneLoc+"filmed_object-*.npy"))[filmedObjectIdx]).item())


# In[390]:

# print(len(filmedObjectData[DICT_TRAJECTORY_POINTS]))
# print(filmedObjectData)


# In[43]:

# height, width = medianImage.shape[:2]
# viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, medianImage.shape)
# verbose = False


# updatedPatches = {}

# doFindMaxBillboardSize = True
# if doFindMaxBillboardSize :
#     maxBillboardWidth = 0
#     maxBillboardHeight = 0
# for patchIdx in arange(len(np.sort(preloadedPatches.keys())[frameSubset[0]:frameSubset[1]-1]))[0:] :
#     patchKey = np.sort(preloadedPatches.keys())[frameSubset[0]:frameSubset[1]-1][patchIdx]-1528+1607
#     patchFrameName = "frame-{0:05}.png".format(patchKey+1)
#     patchImage = np.array(Image.open(filmedSceneLoc+patchFrameName)).astype(np.uint8)
#     patchImage = undistortImage(distortionParameter, distortionRatio, patchImage, originalIntrinsics)[0]
#     patchAlpha = np.array(Image.open(objectData[DICT_MASK_LOCATION]+patchFrameName)).astype(np.uint8)
#     patchAlpha = undistortImage(distortionParameter, distortionRatio, patchAlpha, originalIntrinsics)[0][:, :, -1]
    
# #     figure(); imshow(patchImage)
#     ## threshold the alpha based on bg diff 
#     if True :
#         visiblePixels = np.argwhere(patchAlpha != 0)
#         diffs = np.sqrt(np.sum((patchImage[visiblePixels[:, 0], visiblePixels[:, 1], :] - medianImage[visiblePixels[:, 0], visiblePixels[:, 1], :])**2, axis=1))
#         tmp = np.zeros([medianImage.shape[0], medianImage.shape[1]], np.uint8)
#         tmp[visiblePixels[:, 0], visiblePixels[:, 1]] = diffs
#         tmp = cv2.medianBlur(tmp, 7)
#         tmp = tmp/float(np.max(tmp))
#         med = np.median(tmp[visiblePixels[:, 0], visiblePixels[:, 1]])
#         tmp[tmp<med*0.8] = 0
#         tmp[tmp>0] = np.max(tmp)
# #         figure(); imshow(tmp, clim=(0.0, np.max(tmp)))
# #         figure(); imshow(patchAlpha)
#         patchAlpha = (tmp/np.max(tmp)*255).astype(np.uint8)
        

#     visiblePixels = np.argwhere(patchAlpha != 0)
#     topLeft = np.min(visiblePixels, axis=0)
#     patchSize = np.max(visiblePixels, axis=0) - topLeft + 1

#     colors = np.concatenate([patchImage[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)

#     patchData = {'top_left_pos':topLeft, 'sprite_colors':colors[:, [2, 1, 0, 3]],
#                  'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}

#     # patchData = preloadedPatches[np.sort(preloadedPatches.keys())[frameSubset[0]:frameSubset[1]-1][0]]
#     patchColors = np.zeros(np.concatenate([patchData['patch_size'], [4]]), dtype=np.uint8)
#     patchColors[patchData['visible_indices'][:, 0], patchData['visible_indices'][:, 1], :] = patchData['sprite_colors'][:, [2, 1, 0, 3]]
#     # figure(); imshow(patchColors)

#     ## find billboard width and height
#     worldPos = trajectory.worldTrajectoryPoints[patchIdx, :]
#     worldOrientDir = trajectory.worldTrajectoryDirections[patchIdx, :]
#     worldUpDir = np.array([0.0, 0.0, 1.0])
#     billboardWidth, billboardHeight = findBillboardSize(worldPos, worldOrientDir, worldUpDir, projectionMat, viewMat, np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]), verbose)


#     ########### ROTATE BILLBOARD TO ALIGN WITH MOVING DIRECTION AND PLACE AT POINT ON TRAJECTORY ###########
#     billboardModelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(np.pi/2.0, np.array([1.0, 0.0, 0.0])))
#     adjustAngle = np.arccos(np.clip(np.dot(np.array([-1.0, 0.0, 0.0]), worldOrientDir), -1, 1))
#     adjustAxis = np.cross(worldOrientDir, np.array([-1.0, 0.0, 0.0]))
#     adjustAxis /= np.linalg.norm(adjustAxis)
#     billboardModelMat = np.dot(quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis)), billboardModelMat)
#     billboardModelMat[:-1, -1] = worldPos

#     billboard = GLBillboard(np.zeros([10, 10*billboardWidth/billboardHeight, 4], np.uint8), billboardHeight, modelMat=billboardModelMat)

#     worldBillboardVertices = np.dot(billboard.modelMat, np.concatenate([billboard.vertices, np.ones([len(billboard.vertices), 1])], axis=1).T).T[:, :-1]

#     ## this could be done with one matrix computation but cba right now
#     screenBillboardVertices = np.zeros([len(worldBillboardVertices), 2])
#     for i, vertex in enumerate(worldBillboardVertices) :
#         screenBillboardVertices[i, :] = worldToScreenSpace(viewMat, projectionMat, vertex, width, height)
#     #     print(vertex, screenBillboardVertices[i, :])

#     footprintBillboardModelMat = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))
#     footprintBillboardModelMat[:-1, -1] = worldPos
#     footprintBillboard = GLBillboard(np.zeros([1000*0.081, 1000*0.18, 4], np.uint8), 0.265, modelMat=footprintBillboardModelMat)
#     worldFootprintBillboardVertices = np.dot(footprintBillboard.modelMat, np.concatenate([footprintBillboard.vertices, np.ones([len(footprintBillboard.vertices), 1])], axis=1).T).T[:, :-1]
#     ## this could be done with one matrix computation but cba right now
#     screenFootprintBillboardVertices = np.zeros([len(worldFootprintBillboardVertices), 2])
#     for i, vertex in enumerate(worldFootprintBillboardVertices) :
#         screenFootprintBillboardVertices[i, :] = worldToScreenSpace(viewMat, projectionMat, vertex, width, height)
        
#     texHeight = 0
#     texWidth = 0
#     if not doFindMaxBillboardSize :
#         if maxBillboardWidth > maxBillboardHeight :
#             texWidth = 512
#             texHeight = texWidth*maxBillboardHeight/maxBillboardWidth
#         else :
#             texHeight = 512
#             texWidth = texHeight*maxBillboardWidth/maxBillboardHeight#*billboardWidth/billboardHeight
    
#     ######################### SCALE COMPENSATION BASED ON BILLBOARD WIDTH IN WORLD SPACE #########################
#     if True :
#         if doFindMaxBillboardSize and billboardWidth > maxBillboardWidth :
#             maxBillboardWidth = billboardWidth
#             maxBillboardWidthOnHeight = billboardWidth/billboardHeight
#             print("NEW MAX WIDTH", patchIdx, maxBillboardWidth)
#         if doFindMaxBillboardSize and billboardHeight > maxBillboardHeight :
#             maxBillboardHeight = billboardHeight
#             print("NEW MAX HEIGHT", patchIdx, billboardHeight)
#         widthScale = billboardWidth/maxBillboardWidth
#         heightScale = billboardHeight/maxBillboardHeight
# #         patchScale = patchScale*(billboardWidth/billboardHeight)/maxBillboardWidthOnHeight
#         scaledTexHeight = texHeight*heightScale
#         scaledTexWidth = scaledTexHeight*billboardWidth/billboardHeight#texWidth*widthScale
#         print(patchIdx, patchKey, billboardWidth, billboardHeight, maxBillboardWidth, maxBillboardHeight, texHeight, texWidth, scaledTexHeight, scaledTexWidth)
    
#     ######################### SCALE COMPENSATION BASED ON WIDTH OF PATCH BILLBOARD PROJECTS TO IN CAMERA SPACE #########################
#     else :
#         billboardWidth = np.max(screenFootprintBillboardVertices[:, 0]) - np.min(screenFootprintBillboardVertices[:, 0])
#         billboardHeight = np.max(screenFootprintBillboardVertices[:, 1]) - np.min(screenFootprintBillboardVertices[:, 1])
#         if doFindMaxBillboardSize and billboardWidth > maxBillboardWidth :
#             maxBillboardWidth = billboardWidth
#             maxBillboardWidthOnHeight = billboardWidth/billboardHeight
#             print(patchIdx, maxBillboardWidth)
#         patchScale = billboardWidth/maxBillboardWidth
# #         patchScale = patchScale*(billboardWidth/billboardHeight)/maxBillboardWidthOnHeight
#         print(patchIdx, billboardWidth, maxBillboardWidth, patchScale)
#         scaledTexHeight = texHeight*patchScale
#         scaledTexWidth = texWidth*patchScale
    
# #     billboardHomography = cv2.findHomography(screenBillboardVertices[[2, 0, 1, 4], :], np.array([[0, 0], [texWidth, 0], [texWidth, texHeight], [0, texHeight]], dtype=float))[0]
#     billboardHomography = cv2.findHomography(screenBillboardVertices[[2, 0, 1, 4], :], np.array([[(texWidth-scaledTexWidth)/2.0, (texHeight-scaledTexHeight)/2.0],
#                                                                                                  [texWidth-(texWidth-scaledTexWidth)/2.0, (texHeight-scaledTexHeight)/2.0],
#                                                                                                  [texWidth-(texWidth-scaledTexWidth)/2.0, texHeight-(texHeight-scaledTexHeight)/2.0],
#                                                                                                  [(texWidth-scaledTexWidth)/2.0, texHeight-(texHeight-scaledTexHeight)/2.0]], dtype=float))[0]
#     tmp = cv2.warpPerspective(np.concatenate([patchImage, patchAlpha.reshape([patchAlpha.shape[0], patchAlpha.shape[1], 1])], axis=-1), billboardHomography, (int(np.ceil(texWidth)), int(np.ceil(texHeight))))
# #     figure(); imshow(tmp)
# #     print(tmp.shape, billboardWidth, billboardHeight, texWidth)
# #     print(screenBillboardVertices)
# #     print(patchIdx)
    
#     visiblePixels = np.argwhere(tmp[:, :, -1] != 0)
    
#     colors = np.concatenate([tmp[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)
# #     print(colors.shape)

#     updatedPatches[patchKey] = {'top_left_pos':np.zeros(2, int), 'sprite_colors':colors[:, [2, 1, 0, 3]],
#                                 'visible_indices': visiblePixels, 'patch_size': np.array([int(np.ceil(texHeight)), int(np.ceil(texWidth))], int)}


# In[197]:

# np.save("/home/ilisescu/PhD/data/havana/thresh_camera_adjusted_using_billboard_homography_scale-based-on-maxsize-world-billboard_preloaded_patches-white_bus1.npy", updatedPatches)


# In[35]:

# compositedImage = np.copy(medianImage)
# patchColors = np.zeros(np.concatenate([patchData['patch_size'], [4]]), dtype=np.uint8)
# patchColors[patchData['visible_indices'][:, 0], patchData['visible_indices'][:, 1], :] = patchData['sprite_colors'][:, [2, 1, 0, 3]]

# compositedImage[patchData['top_left_pos'][0]:patchData['top_left_pos'][0]+patchData['patch_size'][0],
#                 patchData['top_left_pos'][1]:patchData['top_left_pos'][1]+patchData['patch_size'][1], :] = (compositedImage[patchData['top_left_pos'][0]:patchData['top_left_pos'][0]+patchData['patch_size'][0],
#                                                                                                                             patchData['top_left_pos'][1]:patchData['top_left_pos'][1]+patchData['patch_size'][1], :]*
#                                                                                                            (1.0-patchColors[:, :, -1]/255.0).reshape([patchColors.shape[0], patchColors.shape[1], 1]) +
#                                                                                                            patchColors[:, :, :-1]*(patchColors[:, :, -1]/255.0).reshape([patchColors.shape[0], patchColors.shape[1], 1]))


# figure(); imshow(compositedImage)
# xlim([0, medianImage.shape[1]])
# ylim([medianImage.shape[0], 0])
# # xlim([(medianImage.shape[1]-1280)/2, 1280+(medianImage.shape[1]-1280)/2])
# # ylim([720+(medianImage.shape[0]-720)/2, (medianImage.shape[0]-720)/2])

# scatter(trajectory.cameraTrajectoryPoints[:, 0], trajectory.cameraTrajectoryPoints[:, 1], color=tuple(trajectory.drawColor/255.0), marker='o', facecolors='none', s=90)
# scatter(trajectoryPoints[:, 0], trajectoryPoints[:, 1], color='red', marker='x', s=90)
# plot([cameraPos[0], cameraDirPos[0]], [cameraPos[1], cameraDirPos[1]], color='yellow', linewidth=2)
# plot([cameraPos[0], cameraUpDirPos[0]], [cameraPos[1], cameraUpDirPos[1]], color='yellow', linewidth=2)
# plot([patchData['top_left_pos'][1], patchData['top_left_pos'][1], patchData['top_left_pos'][1]+patchData['patch_size'][1],
#       patchData['top_left_pos'][1]+patchData['patch_size'][1], patchData['top_left_pos'][1]], [patchData['top_left_pos'][0], patchData['top_left_pos'][0]+patchData['patch_size'][0],
#                                                                                                  patchData['top_left_pos'][0]+patchData['patch_size'][0], patchData['top_left_pos'][0],
#                                                                                                  patchData['top_left_pos'][0]], color='red', linewidth=2)

# scatter([cameraDirLeftIntersection[0], cameraDirRightIntersection[0], cameraUpDirTopIntersection[0], cameraUpDirBottomIntersection[0]],
#         [cameraDirLeftIntersection[1], cameraDirRightIntersection[1], cameraUpDirTopIntersection[1], cameraUpDirBottomIntersection[1]], color='blue', s=90)
# scatter([cameraClosestPointsToIntersection[:, 0]], [cameraClosestPointsToIntersection[:, 1]], color="yellow", marker="x", s=90)
# plot(screenBillboardVertices[[0, 1, 4, 2, 0], 0], screenBillboardVertices[[0, 1, 4, 2, 0], 1], color='magenta', linewidth=2)
# plot(screenFootprintBillboardVertices[[0, 1, 4, 2, 0], 0], screenFootprintBillboardVertices[[0, 1, 4, 2, 0], 1], color='cyan', linewidth=2)

# xlim([patchData['top_left_pos'][1]-patchData['patch_size'][1]*0.5, patchData['top_left_pos'][1]+patchData['patch_size'][1]*1.5])
# ylim([patchData['top_left_pos'][0]+patchData['patch_size'][0]*1.5, patchData['top_left_pos'][0]-patchData['patch_size'][0]*0.5])


# # CODE FOR GETTING CAMERA MATRICES FOR NUKE

# In[26]:

# filmedSceneLoc = "/media/ilisescu/Data1/PhD/data/theme_park_sunny/"
# filmedSceneData = np.load(filmedSceneLoc+"filmed_scene-theme_park_sunny.npy").item()
# cameraExtrinsics = filmedSceneData[DICT_CAMERA_EXTRINSICS]

# cameraIntrinsics = filmedSceneData[DICT_CAMERA_INTRINSICS]
# originalIntrinsics = np.copy(cameraIntrinsics)

# medianImage = np.array(Image.open(filmedSceneLoc+"median.png"), np.uint8)
# distortionParameter = filmedSceneData[DICT_DISTORTION_PARAMETER]
# distortionRatio = filmedSceneData[DICT_DISTORTION_RATIO]
# medianImage, cameraIntrinsics, distortionCoeff, _, _ = undistortImage(distortionParameter, distortionRatio, medianImage, cameraIntrinsics)
# viewMat, projectionMat = cvCameraToOpenGL(cameraExtrinsics, cameraIntrinsics, medianImage.shape[:2])


# In[21]:

# ## finding focal length to use in nuke
# np.dot(viewMat, np.linalg.inv(viewMat)[:, -1].reshape([4, 1]))
# print(cameraIntrinsics)
# print(originalIntrinsics)
# vFov = np.arctan2(1.0, projectionMat[1, 1])*2.0*180.0/np.pi
# vFocalLen = cameraIntrinsics[1, -1]*2/(2.0*np.tan(np.pi*vFov/360.0))
# print(vFov, vFocalLen, vFocalLen*20.25/cameraIntrinsics[1, -1]*2*(cameraIntrinsics[1, -1]*2/originalIntrinsics[1, -1]*2),
#       medianImage.shape, cameraIntrinsics[0, -1]*2/cameraIntrinsics[1, -1]*2, originalIntrinsics[0, -1]*2/originalIntrinsics[1, -1]*2)


# In[31]:

# ## NEED this to compute the location and projection of the footprint
# inverseT = np.linalg.inv(np.dot(originalIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]))
# print(inverseT)
# print(np.dot(inverseT, np.array([928.83824361, 449.23967789, 1.0]))/np.dot(inverseT, np.array([928.83824361, 449.23967789, 1.0]))[-1])
# print(cameraIntrinsics)


# In[22]:

# ## trying to get the model mat for a camera in Nuke's coordinate system
# print(np.linalg.inv(np.dot(np.array([[1, 0, 0, 0],
#                                      [0, np.cos(np.pi/2.0), -np.sin(np.pi/2.0), 0],
#                                      [0, np.sin(np.pi/2.0), np.cos(np.pi/2.0), 0],
#                                      [0, 0, 0, 1]]), cameraExtrinsics)))
# print(np.linalg.inv(viewMat))
# print(np.linalg.inv(cameraExtrinsics))
# print("\n")
# ## camera transform matrix is the inverse of the cameraExtrinsics which is defined in my custom coordinate system with z up
# ## rotate by 90 around z axis
# tMat1 = np.array([[np.cos(np.pi/2.0), -np.sin(np.pi/2.0), 0, 0],
#                   [np.sin(np.pi/2.0), np.cos(np.pi/2.0), 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]])
# ## rotate by -90 around x axis
# tMat2 = np.array([[1, 0, 0, 0],
#                   [0, np.cos(-np.pi/2.0), -np.sin(-np.pi/2.0), 0],
#                   [0, np.sin(-np.pi/2.0), np.cos(-np.pi/2.0), 0],
#                   [0, 0, 0, 1]])
# ## rotate by 90 around y axis
# tMat3 = np.array([[np.cos(np.pi/2.0), 0, np.sin(np.pi/2.0), 0],
#                   [0, 1, 0, 0],
#                   [-np.sin(np.pi/2.0), 0, np.cos(np.pi/2.0), 0],
#                   [0, 0, 0, 1]])
# print(np.arctan2(1.0, projectionMat[1, 1])*2.0*180.0/np.pi)
# # print()
# # print(np.dot(np.dot(tMat2, tMat1), np.linalg.inv(cameraExtrinsics)))
# # print()
# # print(np.dot(np.dot(tMat1, tMat3), np.linalg.inv(cameraExtrinsics)))
# # print()
# # print(np.dot(np.linalg.inv(np.dot(tMat2, tMat1)), np.linalg.inv(cameraExtrinsics)))
# # print()
# # print(np.dot(np.linalg.inv(np.dot(tMat1, tMat2)), np.linalg.inv(cameraExtrinsics)))
# # print()
# # print(np.dot(np.linalg.inv(cameraExtrinsics), np.linalg.inv(np.dot(tMat2, tMat1))))
# print()
# print(np.dot(np.dot(tMat1, np.dot(tMat2, np.dot(np.linalg.inv(cameraExtrinsics), np.linalg.inv(tMat2)))), np.linalg.inv(tMat1)))
# print()
# print(np.dot(np.dot(tMat1, tMat2), np.dot(np.linalg.inv(cameraExtrinsics), np.linalg.inv(np.dot(tMat1, tMat2)))))

# print()
# print(np.linalg.inv(viewMat))
# print(viewMat)
# # print(np.dot(np.linalg.inv(tMat2), np.linalg.inv(viewMat)))
# print(np.dot(tMat2, np.linalg.inv(viewMat)))


# # EXPERIMENTS FOR SMOOTHING TRAJECTORIES

# In[8]:

# def _vec2d_dist(p1, p2):
#     return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


# def _vec2d_sub(p1, p2):
#     return (p1[0]-p2[0], p1[1]-p2[1])


# def _vec2d_mult(p1, p2):
#     return p1[0]*p2[0] + p1[1]*p2[1]


# def ramerdouglas(line, dist):
#     """Does Ramer-Douglas-Peucker simplification of a curve with `dist`
#     threshold.

#     `line` is a list-of-tuples, where each tuple is a 2D coordinate

#     Usage is like so:

#     >>> myline = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]
#     >>> simplified = ramerdouglas(myline, dist = 1.0)
#     """

#     if len(line) < 3:
#         return line

#     (begin, end) = (line[0], line[-1]) if line[0] != line[-1] else (line[0], line[-2])

#     distSq = []
#     for curr in line[1:-1]:
#         tmp = (
#             _vec2d_dist(begin, curr) - _vec2d_mult(_vec2d_sub(end, begin), _vec2d_sub(curr, begin)) ** 2 / _vec2d_dist(begin, end))
#         distSq.append(tmp)

#     maxdist = max(distSq)
#     if maxdist < dist ** 2:
#         return [begin, end]

#     pos = distSq.index(maxdist)
#     return (ramerdouglas(line[:pos + 2], dist) + 
#             ramerdouglas(line[pos + 1:], dist)[1:])


# In[6]:

# def spline_4p( t, p_1, p0, p1, p2 ):
#     """ Catmull-Rom
#         (Ps can be numpy vectors or arrays too: colors, curves ...)
#     """
#         # wikipedia Catmull-Rom -> Cubic_Hermite_spline
#         # 0 -> p0,  1 -> p1,  1/2 -> (- p_1 + 9 p0 + 9 p1 - p2) / 16
#     # assert 0 <= t <= 1
#     return (
#           t*((2-t)*t - 1)   * p_1
#         + (t*t*(3*t - 5) + 2) * p0
#         + t*((4 - 3*t)*t + 1) * p1
#         + (t-1)*t*t         * p2 ) / 2
# tmp2 = [np.array([i[0], i[1]]) for i in tmp2]
# numTs = 10.0
# interpolatedPoints = []
# for j in np.arange(1, len(tmp2)-2 ):  # skip the ends
#     for t in np.arange(numTs):  # t: 0 .1 .2 .. .9
#         p = spline_4p(t/numTs, tmp2[j-1], tmp2[j], tmp2[j+1], tmp2[j+2] )
#         # draw p
#         interpolatedPoints.append(p)
        
# interpolatedPoints = np.array(interpolatedPoints)        
# plot(interpolatedPoints[:, 0], interpolatedPoints[:, 1], color="magenta")


# In[7]:

# # import scipy.interpolate as si

# points = [[i[0], i[1]] for i in tmp2] #[[0, 0], [0, 2], [2, 3], [4, 0], [6, 3], [8, 2], [8, 0]];
# points = np.array(points)
# x = points[:,0]
# y = points[:,1]

# t = range(len(points))
# ipl_t = np.linspace(0.0, len(points) - 1, 100)

# x_tup = si.splrep(t, x, k=3)
# y_tup = si.splrep(t, y, k=3)

# x_list = list(x_tup)
# xl = x.tolist()
# x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

# y_list = list(y_tup)
# yl = y.tolist()
# y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

# x_i = si.splev(ipl_t, x_list)
# y_i = si.splev(ipl_t, y_list)

# #==============================================================================
# # Plot
# #==============================================================================

# fig = plt.figure()

# ax = fig.add_subplot(231)
# plt.plot(t, x, '-og')
# plt.plot(ipl_t, x_i, 'r')
# plt.xlim([0.0, max(t)])
# plt.title('Splined x(t)')

# ax = fig.add_subplot(232)
# plt.plot(t, y, '-og')
# plt.plot(ipl_t, y_i, 'r')
# plt.xlim([0.0, max(t)])
# plt.title('Splined y(t)')

# ax = fig.add_subplot(233)
# plt.plot(x, y, '-og')
# plt.plot(x_i, y_i, 'r')
# plt.xlim([min(x) - 0.3, max(x) + 0.3])
# plt.ylim([min(y) - 0.3, max(y) + 0.3])
# plt.title('Splined f(x(t), y(t))')

# ax = fig.add_subplot(234)
# for i in range(7):
#     vec = np.zeros(11)
#     vec[i] = 1.0
#     x_list = list(x_tup)
#     x_list[1] = vec.tolist()
#     x_i = si.splev(ipl_t, x_list)
#     plt.plot(ipl_t, x_i)
# plt.xlim([0.0, max(t)])
# plt.title('Basis splines')
# plt.show()


# In[13]:

def smoothDirections(filterSize, directions, useUniformWeights = False) :
    coeff = special.binom(filterSize*2, range(0, filterSize*2 +1))
    coeff /= np.sum(coeff)
    ##
    if useUniformWeights :
        coeff = np.ones_like(coeff)/len(coeff)
        
    neighbourIdxs = np.arange(-filterSize, filterSize+1)
    smoothed = np.copy(directions)
    
    for i, point in enumerate(directions) :
        validIdxs = np.all(np.array([i+neighbourIdxs >= 0, i+neighbourIdxs < len(directions)]), axis=0)
        closenessToEdge = filterSize*2+1-len(np.argwhere(validIdxs).flatten())
        filterCoeffs = coeff[validIdxs]
        filterCoeffs /= np.sum(filterCoeffs)
        smoothed[i, :] = np.sum(directions[i+neighbourIdxs[validIdxs]]*filterCoeffs.reshape([len(filterCoeffs), 1]), axis=0)
        smoothed[i, :] /= np.linalg.norm(smoothed[i, :])
        if np.linalg.norm(smoothed[i, :]) != 1.0 and i > 0:
            print(i, np.linalg.norm(smoothed[i, :]), smoothed[i, :])
            smoothed[i, :] = smoothed[i-1, :]
    
    return smoothed

# print(trajectory.worldTrajectoryDirections)
# worldTrajectorySmoothDirections = smoothDirections(15, trajectory.worldTrajectoryDirections, True)
# print(worldTrajectorySmoothDirections)
# worldTrajectoryPointsDistances = np.ones(len(trajectory.worldTrajectoryPoints))*0.01
# worldTrajectoryPointsDistances[:-1] = np.linalg.norm(trajectory.worldTrajectoryPoints[:-1, :]-trajectory.worldTrajectoryPoints[1:, :], axis=1)
# worldTrajectoryPointsDistances = spimg.filters.gaussian_filter1d(worldTrajectoryPointsDistances, 11, axis=0)
# print(worldTrajectoryPointsDistances.shape)
# smoothedWorldTrajectoryPoints = np.zeros_like(trajectory.worldTrajectoryPoints)
# smoothedWorldTrajectoryPoints[0, :] = trajectory.worldTrajectoryPoints[0, :]
# for i in np.arange(1, len(smoothedWorldTrajectoryPoints)) :
#     ## the smooth trajectory is found by moving the previous point along the smooth direction by the amount of space between the previous point and the current in the original trajectory
# #     smoothedWorldTrajectoryPoints[i, :] = smoothedWorldTrajectoryPoints[i-1, :]+worldTrajectorySmoothDirections[i-1, :]*worldTrajectoryPointsDistances[i-1]

#     ## the smooth trajectory is found by moving the previous point along the smooth direction by the amount of space between the previous point in the smooth trajectory and the current one in the original trajectory
# #     smoothedWorldTrajectoryPoints[i, :] = smoothedWorldTrajectoryPoints[i-1, :]+worldTrajectorySmoothDirections[i-1, :]*np.linalg.norm(smoothedWorldTrajectoryPoints[i-1, :]-trajectory.worldTrajectoryPoints[i, :])

#     ## the smooth trajectory is found by projecting the current point in the original trajectory onto the smooth direction
#     a = smoothedWorldTrajectoryPoints[i-1, :]
#     b = smoothedWorldTrajectoryPoints[i-1, :]+worldTrajectorySmoothDirections[i-1, :]
#     p = trajectory.worldTrajectoryPoints[i, :]
#     ap = p-a
#     ab = b-a
#     smoothedWorldTrajectoryPoints[i, :] = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    
# # smoothWorldTrajectoryPointsDistances = np.ones(len(smoothedWorldTrajectoryPoints))*0.01
# # smoothWorldTrajectoryPointsDistances[:-1] = np.linalg.norm(smoothedWorldTrajectoryPoints[:-1, :]-smoothedWorldTrajectoryPoints[1:, :], axis=1)
# # smoothWorldTrajectoryPointsDistances = spimg.filters.gaussian_filter1d(smoothWorldTrajectoryPointsDistances, 11, axis=0)

# # for i in np.arange(1, len(smoothedWorldTrajectoryPoints)-1) :
# #     ## the smooth trajectory is found by moving the previous point along the smooth direction by the amount of space between the previous point and the current in the original trajectory
# #     smoothedWorldTrajectoryPoints[i, :] = smoothedWorldTrajectoryPoints[i-1, :]+worldTrajectorySmoothDirections[i-1, :]*smoothWorldTrajectoryPointsDistances[i-1]
    

# figure(); xlim(-1, 4); ylim(-6, -1)
# # figure(); xlim(2.3, 2.9); ylim(-1.9, -2.3)
# # figure(); xlim(-1, -0.5); ylim(-1.5, -1)
# scatter(trajectory.worldTrajectoryPoints[:, 0], trajectory.worldTrajectoryPoints[:, 1], color="blue", marker='o', facecolors='none')
# scatter(smoothedWorldTrajectoryPoints[:, 0], smoothedWorldTrajectoryPoints[:, 1], color="magenta", marker='x', s=100)
# for i in xrange(len(worldTrajectorySmoothDirections)) :
#     plot([trajectory.worldTrajectoryPoints[i, 0], trajectory.worldTrajectoryPoints[i, 0]+trajectory.worldTrajectoryDirections[i, 0]*0.01],
#          [trajectory.worldTrajectoryPoints[i, 1], trajectory.worldTrajectoryPoints[i, 1]+trajectory.worldTrajectoryDirections[i, 1]*0.01], color="red")
#     plot([trajectory.worldTrajectoryPoints[i, 0], trajectory.worldTrajectoryPoints[i, 0]+worldTrajectorySmoothDirections[i, 0]*worldTrajectoryPointsDistances[i]],
#          [trajectory.worldTrajectoryPoints[i, 1], trajectory.worldTrajectoryPoints[i, 1]+worldTrajectorySmoothDirections[i, 1]*worldTrajectoryPointsDistances[i]], color="magenta")
#     plot([trajectory.worldTrajectoryPoints[i, 0], smoothedWorldTrajectoryPoints[i, 0]],
#          [trajectory.worldTrajectoryPoints[i, 1], smoothedWorldTrajectoryPoints[i, 1]], "y--")


# # CODE FOR DOING THE ADJUSTMENT BASED ON THE TRAJECTORY AND THE STUFF I TALKED ABOUT IN MY TRANSFER VIVA

# In[316]:

# [-0.90505216  0.42530058]
# print(np.array([-0.90343641, 0.42872211])/np.array([1.4828125, 1.45694444])/np.linalg.norm(np.array([-0.90343641, 0.42872211])/np.array([1.4828125, 1.45694444])))
# print(1280/720.0, 1898/1050.0)
# print(832*2, 468*2)
# print(1664/936.0, 1280/720.0, 1898/1050.0)


# figure();
# xlim([0, 1664])
# ylim([936, 0])
# plot(np.array([(newFrameSize/2)[0], (newFrameSize/2)[0]+jack[0]*50]), np.array([(newFrameSize/2)[1], (newFrameSize/2)[1]+jack[1]*50]))

# figure();
# xlim([0, 1898])
# ylim([1050, 0])
# plot(np.array([(np.array([1898, 1050.0])/2)[0], (np.array([1898, 1050.0])/2)[0]+jack[0]*50]), np.array([(np.array([1898, 1050.0])/2)[1], (np.array([1898, 1050.0])/2)[1]+jack[1]*50]))


# In[242]:

# print(tmpTrajectoryCameraSpace-trajectoryPointsCameraSpace)
# figure(); plot(tmpTrajectoryCameraSpace[:, 0], tmpTrajectoryCameraSpace[:, 1])
# plot(trajectoryPointsCameraSpace[:, 0], trajectoryPointsCameraSpace[:, 1])


# In[16]:

# ## UNDISTORT EACH IMAGE WHERE A SPRITE IS PRESENT AND SAVE CORRECTED PATCHES ##

# ## load camera data
# trajectorySmoothness = 5
# data3D = np.load("tmp_trajectory_3D.npy").item()
# print(data3D.keys())
# cameraExtrinsics = data3D['extrinsics']
# cameraIntrinsics = data3D['intrinsics']
# distortionParameter = -0.19
# distortionRatio = -0.19
# distortionCoeff = np.array([distortionParameter, distortionParameter*distortionRatio, 0.0, 0.0, 0.0])
# originalIntrinsics = np.array([[702.736053, 0.0, 640.0],
#                                [0.0, 702.736053, 360.0],
#                                [0.0, 0.0, 1.0]])

# objectData = np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()
# print("LOADED", objectData[DICT_SEQUENCE_NAME])
# patches = np.load(objectData[DICT_PATCHES_LOCATION]).item()
# sortedPatchKeys = np.sort(patches.keys())


# ## get trajectory points and directions in both camera and world space
# frameSubset = [30, -50]
# # frameSubset = [0, 0]
# trajectoryPointsCameraSpace = np.array([objectData[DICT_BBOX_CENTERS][key] for key in sort(objectData[DICT_BBOX_CENTERS].keys())])[frameSubset[0]:frameSubset[1]-1, :]
# ## undistort points
# trajectoryPointsCameraSpace = cv2.undistortPoints(trajectoryPointsCameraSpace.reshape((1, len(trajectoryPointsCameraSpace), 2)), originalIntrinsics, distortionCoeff, P=cameraIntrinsics)[0, :, :]

# inverseT = np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]))
# trajectoryPointsWorldSpace = np.dot(inverseT, np.concatenate([trajectoryPointsCameraSpace, np.ones([len(trajectoryPointsCameraSpace), 1], np.float32)], axis=1).T)
# trajectoryPointsWorldSpace /= trajectoryPointsWorldSpace[-1, :]
# trajectoryPointsWorldSpace[-1, :] = 0
# trajectoryPointsWorldSpace = trajectoryPointsWorldSpace.T.astype(np.float32)
        
# ## smooth trajectory
# trajectoryPointsWorldSpace = np.array([spimg.filters.gaussian_filter1d(trajectoryPointsWorldSpace[:, 0], trajectorySmoothness, axis=0),
#                                        spimg.filters.gaussian_filter1d(trajectoryPointsWorldSpace[:, 1], trajectorySmoothness, axis=0), 
#                                        spimg.filters.gaussian_filter1d(trajectoryPointsWorldSpace[:, 2], trajectorySmoothness, axis=0)]).T.astype(np.float32)

# ## reproject points into image space after smoothing
# T = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]])
# trajectoryPointsCameraSpace = np.dot(T, np.concatenate([trajectoryPointsWorldSpace[:, :-1], np.ones([len(trajectoryPointsWorldSpace), 1])], axis=1).T)
# trajectoryPointsCameraSpace = (trajectoryPointsCameraSpace[:-1, :]/trajectoryPointsCameraSpace[-1, :]).T


# trajectoryDirectionsWorldSpace = np.array([trajectoryPointsWorldSpace[i, :]-trajectoryPointsWorldSpace[j, :] for i, j in zip(xrange(1, len(trajectoryPointsWorldSpace)),
#                                                                                                                              xrange(0, len(trajectoryPointsWorldSpace)-1))])
# trajectoryDirectionsWorldSpace /= np.linalg.norm(trajectoryDirectionsWorldSpace, axis=1).reshape([len(trajectoryDirectionsWorldSpace), 1])
# ## use direction of second to last point as the direction for the last point
# trajectoryDirectionsWorldSpace = np.concatenate([trajectoryDirectionsWorldSpace, trajectoryDirectionsWorldSpace[-1, :].reshape([1, trajectoryDirectionsWorldSpace.shape[1]])], axis=0)
# print("Trajectory points and directions", trajectoryPointsCameraSpace.shape, trajectoryPointsWorldSpace.shape, trajectoryDirectionsWorldSpace.shape)


# ## find directions from center of the car to the center of the camera
# cameraPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
# pointToCameraDirectionsWorldSpace = cameraPos.reshape([1, 3]) - trajectoryPointsWorldSpace
# pointToCameraDistances = np.linalg.norm(pointToCameraDirectionsWorldSpace, axis=1)
# pointToCameraDirectionsWorldSpace /= pointToCameraDistances.reshape([len(pointToCameraDistances), 1])

# spacing = 98#1
# subset = spacing+1#400
# # for each point in the trajectory
# preloadedPatches = {}
# for idx, direction in enumerate(trajectoryDirectionsWorldSpace[:subset:spacing, :]) :
#     i = idx*spacing
#     rotAxis = np.cross(np.array([1, 0, 0]), direction)
#     rotAxis /= np.linalg.norm(rotAxis)
#     rotAngle = np.arccos(np.dot(direction, np.array([1, 0, 0])))
    
    
#     ################ figure out how to turn the camera to look at the object ################
#     ## undo rotation of car wrt camera
#     M = quaternionTo4x4Rotation(angleAxisToQuaternion(rotAngle, rotAxis))
#     rotatedDir = np.dot(M, np.array([[pointToCameraDirectionsWorldSpace[i, 0], pointToCameraDirectionsWorldSpace[i, 1], pointToCameraDirectionsWorldSpace[i, 2], 1]]).T)
#     rotatedDir = rotatedDir[:-1, 0]/rotatedDir[-1, 0]
#     rotatedDir /= np.linalg.norm(rotatedDir)
    
#     ## this turns the camera towards the object
#     adjustCamPos, adjustCamNorm = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics))
#     adjustAxis = np.cross(-pointToCameraDirectionsWorldSpace[i, :], adjustCamNorm)
#     adjustAxis /= np.linalg.norm(adjustAxis)
#     adjustAngle = np.arccos(np.clip(np.dot(adjustCamNorm, -pointToCameraDirectionsWorldSpace[i, :]), -1, 1))
#     adjustM = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))

#     camMat = np.eye(4)
#     camMat[:-1, -1] = rotatedDir
#     camMat[:-1, :-1] = np.dot(adjustM[:-1, :-1], np.linalg.inv(cameraExtrinsics)[:-1, :-1])

#     ## this rotates camera to align with ground plane (and the car itself)
#     _, adjustCamRightVec2 = getWorldSpacePosAndNorm(camMat, np.array([[1, 0, 0, 1]], float).T)
#     _, adjustCamUpVec2 = getWorldSpacePosAndNorm(camMat, np.array([[0, -1, 0, 1]], float).T)
#     _, adjustCamNorm2 = getWorldSpacePosAndNorm(camMat)
#     adjustAxis2 = np.copy(adjustCamNorm2)
# #         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, adjustCamRightVec2-np.dot(adjustCamRightVec2, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0])), -1, 1)) ## aligns camera right vector to ground plane
# #         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, np.array([1, 0, 0], float)), -1, 1)) ## not sure what this does
#     if i < len(trajectoryPointsCameraSpace)-1 :
#         trajDir = trajectoryPointsCameraSpace[i, :]-trajectoryPointsCameraSpace[i+1, :]
#     else :
#         trajDir = trajectoryPointsCameraSpace[i-1, :]-trajectoryPointsCameraSpace[i, :]
# #     print(i, np.linalg.norm(trajDir))
#     trajDir /= np.linalg.norm(trajDir)
#     adjustAngle2 = np.arccos(np.clip(np.dot(trajDir, np.array([1, 0], float)), -1, 1)) ## align camera space direction to x axis (does it even make sense?)
#     if np.cross(trajDir, np.array([1, 0], float)) < 0 :
#         adjustAxis2 *= -1.0


#     adjustM2 = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle2, adjustAxis2))
#     camMat[:-1, :-1] = np.dot(M[:-1, :-1], np.dot(adjustM2[:-1, :-1], camMat[:-1, :-1]))
    
#     #########################################################################################
    
    
#     ################ rotate the camera to look at the car ################
#     camPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
#     rotatedCamTransform = rotateAboutPoint(np.linalg.inv(cameraExtrinsics), angleAxisToQuaternion(adjustAngle, adjustAxis), camPos)
# #     rotatedCamTransform = rotateAboutPoint(rotatedCamTransform, angleAxisToQuaternion(adjustAngle2, adjustAxis2), camPos)


#     _, camDir = getWorldSpacePosAndNorm(rotatedCamTransform, np.array([[0.0], [0.0], [1.0], [1.0]]))
#     desiredDist = np.linalg.norm(camPos)#1.0
#     t = camDir*(np.linalg.norm(trajectoryPointsWorldSpace[i, :]-camPos)-desiredDist)
#     tMat = np.array([[1, 0, 0, t[0]],
#                      [0, 1, 0, t[1]],
#                      [0, 0, 1, t[2]],
#                      [0, 0, 0, 1]])
#     ################################### can use this to do the scale normalization thing ###################################
#     tMat = np.eye(4)
#     rotatedExtrinsics = np.dot(np.linalg.inv(rotatedCamTransform), np.linalg.pinv(tMat))
# #     print("dist after tMat", np.linalg.norm(trajectoryPointsWorldSpace[i, :]-getWorldSpacePosAndNorm(np.linalg.inv(rotatedExtrinsics), posOnly=True)))
    
#     #########################################################################################
    
#     frameName = "frame-{0:05}.png".format(sortedPatchKeys[i+frameSubset[0]]+1)
#     frameImg = np.array(Image.open("/home/ilisescu/PhD/data/havana/"+frameName)).astype(np.uint8)
# #     figure(); imshow(frameImg)
# #     scatter(objectData[DICT_BBOX_CENTERS][sortedPatchKeys[i]][0], objectData[DICT_BBOX_CENTERS][sortedPatchKeys[i]][1])

#     frameSize = np.array([frameImg.shape[1], frameImg.shape[0]])
#     print(i, "dist", np.linalg.norm(trajectoryPointsWorldSpace[i, :]-camPos), desiredDist, frameName)

#     ## undistort image
#     sizeDelta = 0.3
#     newFrameSize = (frameSize*(1+sizeDelta)).astype(int)

#     map1, map2 = cv2.initUndistortRectifyMap(originalIntrinsics, distortionCoeff, None, cameraIntrinsics, tuple(newFrameSize), cv2.CV_16SC2)
#     undistortedUncropped = cv2.remap(frameImg, map1, map2, cv2.INTER_LINEAR)
#     figure(); imshow(undistortedUncropped)
#     scatter(trajectoryPointsCameraSpace[i, 0], trajectoryPointsCameraSpace[i, 1])
#     jack = trajectoryPointsCameraSpace[i+1, :]-trajectoryPointsCameraSpace[i, :]
#     jack /= np.linalg.norm(jack)
#     plot(np.array([trajectoryPointsCameraSpace[i, 0], trajectoryPointsCameraSpace[i, 0]+jack[0]*50]), np.array([trajectoryPointsCameraSpace[i, 1], trajectoryPointsCameraSpace[i, 1]+jack[1]*50]))

#     ## get grid points into world space and back into image space using the rotate extrinsics
#     gridDownsample = 1
#     imageGridPoints = np.indices(newFrameSize/gridDownsample).reshape([2, np.prod(newFrameSize/gridDownsample)]).T*gridDownsample
    
#     ## figure out mapping between original camera matrix and the new one that looks at the car
#     rotatedToWorld = np.linalg.inv(np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, [0, 1, 3]]))
#     worldToOriginal = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, :])

#     rotatedGridWorldSpace = np.dot(rotatedToWorld, np.concatenate([imageGridPoints, np.ones([len(imageGridPoints), 1], np.float64)], axis=1).T)
#     rotatedGridWorldSpace /= rotatedGridWorldSpace[-1, :]
#     rotatedGridWorldSpace[-1, :] = 0
#     rotatedGridWorldSpace = rotatedGridWorldSpace.T.astype(np.float64)

#     rotatedGridInOriginalCamera = np.dot(worldToOriginal, np.concatenate([rotatedGridWorldSpace, np.ones([len(rotatedGridWorldSpace), 1], np.float64)], axis=1).T)
#     rotatedGridInOriginalCamera = (rotatedGridInOriginalCamera[:-1, :]/rotatedGridInOriginalCamera[-1, :]).T
#     rotatedGridInOriginalCamera = rotatedGridInOriginalCamera.T.reshape([2, (newFrameSize/gridDownsample)[0], (newFrameSize/gridDownsample)[1]]).T.astype(np.float32)
#     mapPoints1, mapPoints2 = cv2.convertMaps(rotatedGridInOriginalCamera, None, cv2.CV_16SC2)
#     rotatedFrameImg = cv2.remap(undistortedUncropped, mapPoints1, mapPoints2, cv2.INTER_LINEAR)
#     figure(); imshow(rotatedFrameImg)
#     scatter((newFrameSize/2)[0], (newFrameSize/2)[1])
#     plot(np.array([(newFrameSize/2)[0], (newFrameSize/2)[0]+jack[0]*50]), np.array([(newFrameSize/2)[1], (newFrameSize/2)[1]+jack[1]*50]))
    
#     rotatedFrameAlpha = cv2.remap(cv2.remap(np.array(Image.open("/home/ilisescu/PhD/data/havana/blue_car1-maskedFlow-blended/"+frameName)).astype(np.uint8),
#                                             map1, map2, cv2.INTER_LINEAR), mapPoints1, mapPoints2, cv2.INTER_LINEAR)
# #     figure(); imshow(rotatedFrameAlpha)
    
#     ## find patchsize and top left such that the center of the image is in the center of the patch
#     visiblePixels = np.argwhere(rotatedFrameAlpha[:, :, -1] != 0)
#     imgCenter = np.array(rotatedFrameAlpha.shape[:2])/2
#     halfSize = np.max(np.array([imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter]), axis=0)
    
#     topLeft = imgCenter-halfSize #np.min(visiblePixels, axis=0)
#     patchSize = halfSize*2 + 1 #np.max(visiblePixels, axis=0) - topLeft + 1
    
#     colors = np.concatenate([rotatedFrameImg[visiblePixels[:, 0], visiblePixels[:, 1], :], np.ones([len(visiblePixels), 1])*255], axis=1).astype(np.uint8)
# #     print(colors.shape)

#     preloadedPatches[sortedPatchKeys[i]] = {'top_left_pos':topLeft, 'sprite_colors':colors[:, [2, 1, 0, 3]],
#                                             'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}


# In[92]:

# figure(); plot(np.linalg.norm(trajectoryPointsWorldSpace-camPos, axis=1))
# np.linalg.norm(trajectoryPointsWorldSpace-camPos, axis=1)


# In[237]:

# np.save("/home/ilisescu/PhD/data/havana/camera_adjusted_plus_scale_preloaded_patches-blue_car1.npy", preloadedPatches)
# print(trajectoryDirectionsWorldSpace)


# In[479]:

# # print(len(preloadedPatches))
# # figure(); imshow(rotatedFrameAlpha)
# # print(np.min(visiblePixels, axis=0))
# # print(len(sortedPatchKeys))
# # print(patches[sortedPatchKeys[i]])
# figure(); imshow(rotatedFrameImg)
# # print(i)


# In[511]:

# print(preloadedPatches.keys())
# patch = preloadedPatches[sortedPatchKeys[193]]
# img = np.zeros([patch['patch_size'][0], patch['patch_size'][1], 4], dtype=np.int8)
# img[patch['visible_indices'][:, 0], patch['visible_indices'][:, 1], :] = patch['sprite_colors']
# img = img[:, :, [2, 1, 0, 3]]
# figure(); imshow(img.astype(np.uint8))
# scatter((patch['patch_size']/2)[1], (patch['patch_size']/2)[0])


# In[430]:

# print(patch['patch_size']/2)


# In[420]:

# print(imgCenter, halfSize)
# print(np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0), "\n")

# print(topLeft, topLeft+patchSize)
# print(np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0), "\n")

# print(np.array([imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter]))
# print(imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter)
# print(np.max(np.array([imgCenter-np.min(visiblePixels, axis=0), np.max(visiblePixels, axis=0)-imgCenter]), axis=0))


# In[362]:

# ## check if the trajectory point projects to the center of the image using the rotatedExtrinsics
# print(trajectoryPointsCameraSpace[i, :])
# print(objectData[DICT_BBOX_CENTERS][sortedPatchKeys[frameIdx]])
# trajPoint = np.concatenate([trajectoryPointsWorldSpace[i, :], [1]]).reshape([1, 4])
# print(trajectoryPointsWorldSpace[i, :], trajPoint)
# projTrajPoint = np.dot(np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, :]), np.concatenate([trajectoryPointsWorldSpace[i, :], [1]]).reshape([1, 4]).T)
# projTrajPoint /= projTrajPoint[-1, 0]
# print(np.array(rotatedFrameAlpha.shape[:2])[::-1]/2, projTrajPoint.flatten())


# In[48]:

# frustumEdges = np.array(np.array([[0, 0, 0, 1],
#                                   [.25, .25, 1, 1],
#                                   [.25, -.25, 1, 1],
#                                   [-.25, -.25, 1, 1],
#                                   [-.25, .25, 1, 1]]))
# ## load 3D data
# data3D = np.load("tmp_trajectory_3D.npy").item()
# print(data3D.keys())
# cameraExtrinsics = data3D['extrinsics']
# cameraIntrinsics = data3D['intrinsics']
# ### NEED TO SAVE THIS IN THE 3D DATA BUT WHATEVS, FOR NOW IT'S HARDCODED FOR BLUE_CAR1
# # frameSubset = [30, -50]
# # frameSubset = [0, -1]

# # trajectoryPointsCameraSpace = data3D['trajectoryPointsCameraSpace']
# # trajectoryPointsWorldSpace = data3D['trajectoryPointsWorldSpace']
# # trajectoryDirectionsWorldSpace = data3D['trajectoryDirectionsWorldSpace']

# ## set up figure
# fig = figure()
# ax = fig.add_subplot(111, aspect='equal', projection='3d')
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)
# cols = cm.jet(np.arange(len(trajectoryPointsWorldSpace), dtype=float)/len(trajectoryPointsWorldSpace))

# ## plot car footprint and look direction as [1, 0, 0]
# if True :
#     footprint = np.array([[0.8, 0.5, 0.0],
#                           [0.8, -0.5, 0.0],
#                           [-0.8, -0.5, 0.0],
#                           [-0.8, 0.5, 0.0],
#                           [0.0, 0.0, 0.0],
#                           [0.8, 0.0, 0.0]]).T*0.5
#     for i, j in zip([0, 1, 2, 3, 4], [1, 2, 3, 0, 5]) :
#         ax.plot(np.array([footprint[0, i], footprint[0, j]]), np.array([footprint[1, i], footprint[1, j]]), np.array([footprint[2, i], footprint[2, j]]), c="magenta")

# ## plot normalized directions
# if False :
#     for i, direction in enumerate(trajectoryDirectionsWorldSpace) :
#         ax.plot(np.array([0, direction[0]]), np.array([0, direction[1]]), np.zeros(2), c=cols[i, :])
    
# ## plot trajectory
# if True :
#     ax.plot(trajectoryPointsWorldSpace[:, 0], trajectoryPointsWorldSpace[:, 1], np.zeros(len(trajectoryPointsWorldSpace)), c="cyan")

# ## find directions from center of the car to the center of the camera
# cameraPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
# pointToCameraDirectionsWorldSpace = cameraPos.reshape([1, 3]) - trajectoryPointsWorldSpace
# pointToCameraDistances = np.linalg.norm(pointToCameraDirectionsWorldSpace, axis=1)
# pointToCameraDirectionsWorldSpace /= pointToCameraDistances.reshape([len(pointToCameraDistances), 1])

# ## plot pointToCameraDistances
# if False :
#     for i, [direction, position] in enumerate(zip(pointToCameraDirectionsWorldSpace[:, :], trajectoryPointsWorldSpace[:, :])) :
#         ax.plot(np.array([position[0], direction[0]*pointToCameraDistances[i]+position[0]]),
#                 np.array([position[1], direction[1]*pointToCameraDistances[i]+position[1]]),
#                 np.array([position[2], direction[2]*pointToCameraDistances[i]+position[2]]), c=cols[i, :])
        

# ## find rotation to align trajectory directions to [1, 0, 0] and use it to rotate the pointToCameraDirections
# spacing = 10
# subset = 400
# scaledFrustumEdges = np.copy(frustumEdges.T)
# scaledFrustumEdges[0:2, :] *= .04
# scaledFrustumEdges[2, :] *= .05
# for idx, direction in enumerate(trajectoryDirectionsWorldSpace[:subset:spacing, :]) :
#     i = idx*spacing
#     lastI = np.copy(i)
#     rotAxis = np.cross(np.array([1, 0, 0]), direction)
#     rotAxis /= np.linalg.norm(rotAxis)
#     rotAngle = np.arccos(np.dot(direction, np.array([1, 0, 0])))
    
#     M = quaternionTo4x4Rotation(angleAxisToQuaternion(rotAngle, rotAxis))
#     rotatedDir = np.dot(M, np.array([[pointToCameraDirectionsWorldSpace[i, 0], pointToCameraDirectionsWorldSpace[i, 1], pointToCameraDirectionsWorldSpace[i, 2], 1]]).T)
#     rotatedDir = rotatedDir[:-1, 0]/rotatedDir[-1, 0]
#     rotatedDir /= np.linalg.norm(rotatedDir)
#     if False :
#         ax.plot(np.array([0, rotatedDir[0]]), np.array([0, rotatedDir[1]]), np.array([0, rotatedDir[2]]), c=cols[i, :])
        
#     if True :
#         ## plot footprint at current trajectory point
#         frameFootprint = np.array([[0.8, 0.5],
#                                    [0.8, -0.5],
#                                    [-0.8, -0.5],
#                                    [-0.8, 0.5],
#                                    [0.0, 0.0],
#                                    [0.8, 0.0]]).T*0.1
#         footprintRotAngle = -np.arccos(np.dot(direction, np.array([1, 0, 0])))
#         footprintRotMat = np.array([[np.cos(footprintRotAngle), -np.sin(footprintRotAngle)],
#                                     [np.sin(footprintRotAngle), np.cos(footprintRotAngle)]])
#         frameFootprint = np.dot(footprintRotMat, frameFootprint)
#         frameFootprint += trajectoryPointsWorldSpace[i, :-1].reshape([2, 1])
#         for fi, fj in zip([0, 1, 2, 3, 4], [1, 2, 3, 0, 5]) :
#             ax.plot(np.array([frameFootprint[0, fi], frameFootprint[0, fj]]), np.array([frameFootprint[1, fi], frameFootprint[1, fj]]), np.array([0.0, 0.0]), c="magenta")
    
#     if True :
#         ## this turns the camera towards the object
#         adjustCamPos, adjustCamNorm = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics))
#         adjustAxis = np.cross(-pointToCameraDirectionsWorldSpace[i, :], adjustCamNorm)
#         adjustAxis /= np.linalg.norm(adjustAxis)
#         adjustAngle = np.arccos(np.clip(np.dot(adjustCamNorm, -pointToCameraDirectionsWorldSpace[i, :]), -1, 1))
#         adjustM = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle, adjustAxis))
#         print(i, -rotatedDir, pointToCameraDirectionsWorldSpace[i, :], direction)
        
#         camMat = np.eye(4)
#         camMat[:-1, -1] = rotatedDir
#         camMat[:-1, :-1] = np.dot(adjustM[:-1, :-1], np.linalg.inv(cameraExtrinsics)[:-1, :-1])
# #         camMat[:-1, :-1] = np.dot(M[:-1, :-1], np.dot(adjustM2[:-1, :-1], np.dot(adjustM[:-1, :-1], np.linalg.inv(cameraExtrinsics)[:-1, :-1])))
        
#         ## this rotates camera to align with ground plane (and the car itself)
#         _, adjustCamRightVec2 = getWorldSpacePosAndNorm(camMat, np.array([[1, 0, 0, 1]], float).T)
#         _, adjustCamUpVec2 = getWorldSpacePosAndNorm(camMat, np.array([[0, -1, 0, 1]], float).T)
#         _, adjustCamNorm2 = getWorldSpacePosAndNorm(camMat)
#         adjustAxis2 = np.copy(adjustCamNorm2)
# #         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, adjustCamRightVec2-np.dot(adjustCamRightVec2, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0])), -1, 1)) ## aligns camera right vector to ground plane
# #         adjustAngle2 = np.arccos(np.clip(np.dot(adjustCamRightVec2, np.array([1, 0, 0], float)), -1, 1)) ## not sure what this does
#         trajDir = trajectoryPointsCameraSpace[i, :]-trajectoryPointsCameraSpace[i+1, :]
#         trajDir /= np.linalg.norm(trajDir)
#         adjustAngle2 = np.arccos(np.clip(np.dot(trajDir, np.array([1, 0], float)), -1, 1)) ## align camera space direction to x axis (does it even make sense?)
#         if np.cross(trajDir, np.array([1, 0], float)) < 0 :
#             adjustAxis2 *= -1.0
        
        
#         adjustM2 = quaternionTo4x4Rotation(angleAxisToQuaternion(adjustAngle2, adjustAxis2))
#         camMat[:-1, :-1] = np.dot(M[:-1, :-1], np.dot(adjustM2[:-1, :-1], camMat[:-1, :-1]))
        
#         camFrustum = np.dot(camMat, np.concatenate([scaledFrustumEdges, np.array([[0, 0, 1, 1]]).T], axis=1))
#         camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
#         for idxI, idxJ in zip([0, 0, 0, 0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 2, 3, 4, 1, 5]) :
#             ax.plot(np.array([camFrustum[0, idxI], camFrustum[0, idxJ]]), np.array([camFrustum[1, idxI], camFrustum[1, idxJ]]), np.array([camFrustum[2, idxI], camFrustum[2, idxJ]]), c=cols[i, :], linewidth=.5)
            
#         camPos, camUp = getWorldSpacePosAndNorm(camMat, np.array([[0.0, -1.0, 0.0, 1.0]]).T)
#         camUp *=0.05
#         ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]), c="green")
        
#         _, bob = getWorldSpacePosAndNorm(camMat, np.array([[1, 0, 0, 1]], float).T)
# #         print(adjustAngle2, bob, adjustCamRightVec2, adjustCamRightVec2-np.dot(adjustCamRightVec2, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0]),
# #               np.arccos(np.clip(np.dot(bob, bob-np.dot(bob, np.array([0.0, 0.0, 1.0]))*np.array([0.0, 0.0, 1.0])), -1, 1)))
    
# #     ax.scatter(-tmpDirections[lastI, 0], -tmpDirections[lastI, 1], -tmpDirections[lastI, 2])
        
        
# ## plot captured camera frustum
# if True :
#     camFrustum = np.dot(np.linalg.inv(cameraExtrinsics), frustumEdges.T)
#     camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
#     camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
#     for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
#         ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="blue")
#     ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]))
        
        
# ## draw sphere
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
# x=np.cos(u)*np.sin(v)
# y=np.sin(u)*np.sin(v)
# z=np.cos(v)
# ax.plot_wireframe(x, y, z, color="r", linewidth=.2)


# In[307]:

# objectData = np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()
# patches = np.load(objectData[DICT_PATCHES_LOCATION]).item()
# sortedPatchKeys = np.sort(patches.keys())

# frameIdx = lastI+frameSubset[0]
# patch = patches[sortedPatchKeys[frameIdx]]
# img = np.zeros([patch['patch_size'][0], patch['patch_size'][1], 4], dtype=np.int8)
# img[patch['visible_indices'][:, 0], patch['visible_indices'][:, 1], :] = patch['sprite_colors']
# img = img[:, :, [2, 1, 0, 3]]

# # figure(); imshow(img[:, :, :-1].astype(np.uint8))

# camPos = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), posOnly=True)
# rotatedCamTransform = rotateAboutPoint(np.linalg.inv(cameraExtrinsics), angleAxisToQuaternion(adjustAngle, adjustAxis), camPos)
# rotatedCamTransform = rotateAboutPoint(rotatedCamTransform, angleAxisToQuaternion(adjustAngle2, adjustAxis2), camPos)


# _, camDir = getWorldSpacePosAndNorm(rotatedCamTransform, np.array([[0.0], [0.0], [1.0], [1.0]]))
# t = camDir*1.3
# tMat = np.array([[1, 0, 0, t[0]],
#                  [0, 1, 0, t[1]],
#                  [0, 0, 1, t[2]],
#                  [0, 0, 0, 1]])
# ################################### can use this to do the scale normalization thing ###################################
# tMat = np.eye(4)
# rotatedExtrinsics = np.dot(np.linalg.inv(rotatedCamTransform), np.linalg.pinv(tMat))

# print(np.linalg.inv(rotatedExtrinsics))
# print(np.linalg.inv(cameraExtrinsics))

# camFrustum = np.dot(np.linalg.inv(rotatedExtrinsics), frustumEdges.T)
# camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
# camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(rotatedExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
# for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
#     ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="magenta")
# ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]), c="magenta")


# In[308]:

# distortionParameter = -0.19
# distortionRatio = -0.19
# distortionCoeff = np.array([distortionParameter, distortionParameter*distortionRatio, 0.0, 0.0, 0.0])

        
# originalIntrinsics = np.array([[702.736053, 0.0, 640.0],
#                                [0.0, 702.736053, 360.0],
#                                [0.0, 0.0, 1.0]])

# frameImg = np.array(Image.open("/home/ilisescu/PhD/data/havana/frame-{0:05}.png".format(sortedPatchKeys[frameIdx]+1))).astype(np.uint8)
# figure(); imshow(frameImg)
# scatter(objectData[DICT_BBOX_CENTERS][sortedPatchKeys[frameIdx]][0], objectData[DICT_BBOX_CENTERS][sortedPatchKeys[frameIdx]][1])

# frameSize = np.array([frameImg.shape[1], frameImg.shape[0]])

# ## undistort image
# sizeDelta = 0.3
# newFrameSize = (frameSize*(1+sizeDelta)).astype(int)

# map1, map2 = cv2.initUndistortRectifyMap(originalIntrinsics, distortionCoeff, None, cameraIntrinsics, tuple(newFrameSize), cv2.CV_16SC2)
# undistortedUncropped = cv2.remap(frameImg, map1, map2, cv2.INTER_LINEAR)
# figure(); imshow(undistortedUncropped)
# scatter(trajectoryPointsCameraSpace[lastI, 0], trajectoryPointsCameraSpace[lastI, 1])

# ## get grid points into world space and back into image space using the rotate extrinsics
# gridDownsample = 1
# imageGridPoints = np.indices(newFrameSize/gridDownsample).reshape([2, np.prod(newFrameSize/gridDownsample)]).T*gridDownsample

# if False :
#     print("using old way")
#     inverseT = np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]))
#     projectedImageGridPoints = np.dot(inverseT, np.concatenate([imageGridPoints, np.ones([len(imageGridPoints), 1], np.float64)], axis=1).T)
#     projectedImageGridPoints /= projectedImageGridPoints[-1, :]

#     projectedImageGridPoints[-1, :] = 0
#     projectedImageGridPoints = projectedImageGridPoints.T.astype(np.float64)

#     T = np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, :])
#     rotatedGridPoints = np.dot(T, np.concatenate([projectedImageGridPoints, np.ones([len(projectedImageGridPoints), 1], np.float64)], axis=1).T)
#     mapPoints1, mapPoints2 = cv2.convertMaps((rotatedGridPoints[:-1, :]/rotatedGridPoints[-1, :]).T.reshape([2, (newFrameSize/gridDownsample)[0], (newFrameSize/gridDownsample)[1]]).T.astype(np.float32), None, cv2.CV_16SC2)
#     rotatedGridPoints = np.round((rotatedGridPoints[:-1, :]/rotatedGridPoints[-1, :]).T).astype(int)


#     validCoords = np.all(np.concatenate([rotatedGridPoints >= 0,
#                                          (rotatedGridPoints[:, 0] < newFrameSize[0]).reshape([len(rotatedGridPoints), 1]),
#                                          (rotatedGridPoints[:, 1] < newFrameSize[1]).reshape([len(rotatedGridPoints), 1])], axis=1), axis=1)

#     rotatedFrameImg = np.zeros(undistortedUncropped.shape, np.uint8)
#     rotatedFrameImg[rotatedGridPoints[validCoords, 1], rotatedGridPoints[validCoords, 0], :] = undistortedUncropped[imageGridPoints[validCoords, 1], imageGridPoints[validCoords, 0], :]
# else :
#     print("using opencv")
#     rotatedToWorld = np.linalg.inv(np.dot(cameraIntrinsics, rotatedExtrinsics[:-1, [0, 1, 3]]))
#     worldToOriginal = np.dot(cameraIntrinsics, cameraExtrinsics[:-1, :])

#     rotatedGridWorldSpace = np.dot(rotatedToWorld, np.concatenate([imageGridPoints, np.ones([len(imageGridPoints), 1], np.float64)], axis=1).T)
#     rotatedGridWorldSpace /= rotatedGridWorldSpace[-1, :]
#     rotatedGridWorldSpace[-1, :] = 0
#     rotatedGridWorldSpace = rotatedGridWorldSpace.T.astype(np.float64)

#     rotatedGridInOriginalCamera = np.dot(worldToOriginal, np.concatenate([rotatedGridWorldSpace, np.ones([len(rotatedGridWorldSpace), 1], np.float64)], axis=1).T)
#     rotatedGridInOriginalCamera = (rotatedGridInOriginalCamera[:-1, :]/rotatedGridInOriginalCamera[-1, :]).T
#     rotatedGridInOriginalCamera = rotatedGridInOriginalCamera.T.reshape([2, (newFrameSize/gridDownsample)[0], (newFrameSize/gridDownsample)[1]]).T.astype(np.float32)
#     mapPoints1, mapPoints2 = cv2.convertMaps(rotatedGridInOriginalCamera, None, cv2.CV_16SC2)
#     rotatedFrameImg = cv2.remap(undistortedUncropped, mapPoints1, mapPoints2, cv2.INTER_LINEAR)
# figure(); imshow(rotatedFrameImg)

# scatter((newFrameSize/2)[0], (newFrameSize/2)[1])


# In[32]:

# modelMat = QtGui.QMatrix4x4(0.910911599349, 0.406799823245, 0.0689489809887, 1.04144313517,
#                             0.201452403084, -0.292662295856, -0.934754358041, -0.491083466897,
#                             -0.360079140556, 0.865368525191, -0.348540281954, 3.93563077185,
#                             0.0, 0.0, 0.0, 1.0).inverted()[0]

# cameraIntrinsics = np.array([[702.736053, 0.0, 640.0],
#                              [0.0, 702.736053, 360.0],
#                              [0.0, 0.0, 1.0]])
# cameraExtrinsics = np.array(modelMat.inverted()[0].data()).reshape([4, 4]).T
# print("README", cameraExtrinsics)

# frameSize = np.array([1280, 720])
# gridDownsample = 1
# projectedImageGridPoints = np.indices(frameSize/gridDownsample).reshape([2, np.prod(frameSize/gridDownsample)]).T*gridDownsample
# # projectedImageGridColors = img[projectedImageGridPoints[:, 1], projectedImageGridPoints[:, 0], :].astype(np.float32)/np.float32(255.0)
# tmp = np.copy(projectedImageGridPoints)

# inverseT = np.linalg.inv(np.dot(cameraIntrinsics, cameraExtrinsics[:-1, [0, 1, 3]]))
# projectedImageGridPoints = np.dot(inverseT, np.concatenate([projectedImageGridPoints, np.ones([len(projectedImageGridPoints), 1], np.float64)], axis=1).T)
# projectedImageGridPoints /= projectedImageGridPoints[-1, :]
# projectedImageGridPoints[-1, :] = 0
# projectedImageGridPoints = projectedImageGridPoints.T.astype(np.float64)


# In[251]:

# ## USED TO FIGURE OUT HOW TO ALIGN A CAMERA TO THE GROUND PLANE
# fig = figure()
# ax = fig.add_subplot(111, aspect='equal', projection='3d')
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)

# camFrustum = np.dot(np.linalg.inv(cameraExtrinsics), frustumEdges.T)
# camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
# camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
# _, camNorm = getWorldSpacePosAndNorm(np.linalg.inv(cameraExtrinsics), np.array([[0.0, 0.0, 1.0, 1.0]]).T)
# for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
#     ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="blue")
# ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]))

# frustumCenter = np.average(camFrustum[:, 1:], axis=1)
# ax.plot(np.array([frustumCenter[0], frustumCenter[0]+camUp[0]]), np.array([frustumCenter[1], frustumCenter[1]+camUp[1]]), np.array([frustumCenter[2], frustumCenter[2]+camUp[2]]), c="magenta")
# planeNorm = np.array([0.0, 0.0, 1.0])
# ax.plot(np.array([frustumCenter[0], frustumCenter[0]+planeNorm[0]]), np.array([frustumCenter[1], frustumCenter[1]+planeNorm[1]]), np.array([frustumCenter[2], frustumCenter[2]+planeNorm[2]]), c="cyan")
# ax.plot(np.array([frustumCenter[0], frustumCenter[0]+camNorm[0]]), np.array([frustumCenter[1], frustumCenter[1]+camNorm[1]]), np.array([frustumCenter[2], frustumCenter[2]+camNorm[2]]), c="blue")

# projNorm = planeNorm - np.dot(planeNorm, camNorm)*camNorm
# projNorm /= np.linalg.norm(projNorm)
# ax.plot(np.array([frustumCenter[0], frustumCenter[0]+projNorm[0]]), np.array([frustumCenter[1], frustumCenter[1]+projNorm[1]]), np.array([frustumCenter[2], frustumCenter[2]+projNorm[2]]), c="black")

# adjustAngle3 = np.arccos(np.clip(np.dot(projNorm, camUp), -1, 1))
# adjustAxis3 = np.cross(projNorm, camUp)
# adjustAxis3 /= np.linalg.norm(adjustAxis3)
# rotatedCamTransform = rotateAboutPoint(np.linalg.inv(cameraExtrinsics), angleAxisToQuaternion(adjustAngle3, adjustAxis3), camPos)
# camFrustum = np.dot(rotatedCamTransform, frustumEdges.T)
# camFrustum = camFrustum[:-1, :]/camFrustum[-1, :]
# camPos, camUp = getWorldSpacePosAndNorm(np.linalg.inv(rotatedExtrinsics), np.array([[0.0, -1.0, 0.0, 1.0]]).T)
# for i, j in zip([0, 0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 2, 3, 4, 1]) :
#     ax.plot(np.array([camFrustum[0, i], camFrustum[0, j]]), np.array([camFrustum[1, i], camFrustum[1, j]]), np.array([camFrustum[2, i], camFrustum[2, j]]), c="magenta")
# ax.plot(np.array([camPos[0], camPos[0]+camUp[0]]), np.array([camPos[1], camPos[1]+camUp[1]]), np.array([camPos[2], camPos[2]+camUp[2]]), c="magenta")

