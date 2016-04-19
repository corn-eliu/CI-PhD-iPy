# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# Python script to launch OpenMVG SfM tools on an image dataset
#
# usage : python tutorial_demo.py 
# 

# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "/home/ilisescu/openMVG/openMVG_build/Linux-x86_64-RELEASE"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/ilisescu/openMVG/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

import commands
import os
import subprocess
import sys

<<<<<<< HEAD
def get_parent_dir(directory):
    import os
    return os.path.dirname(directory)

input_eval_dir = "/media/ilisescu/Data2/PhD/raw videos/park/gopro_subset/"
# # Checkout an OpenMVG image dataset with Git
# if not os.path.exists(input_eval_dir):
#   pImageDataCheckout = subprocess.Popen([ "git", "clone", "https://github.com/openMVG/ImageDataset_SceauxCastle.git" ])
#   pImageDataCheckout.wait()

output_eval_dir = os.path.join(get_parent_dir(input_eval_dir), "tutorial_out")
input_eval_dir = os.path.join(input_eval_dir, "images")
if not os.path.exists(output_eval_dir):
  os.mkdir(output_eval_dir)

input_dir = input_eval_dir
output_dir = output_eval_dir
print ("Using input dir  : ", input_dir)
print ("      output_dir : ", output_dir)
=======
# dataFolder = "/home/ilisescu/PhD/data/"
dataFolder = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
dataSet = "theme_park_sunny/"

# <codecell>

figure(); imshow(np.load(dataFolder+"Videos/6489810.avi_distanceMatrix.npy"), interpolation='nearest')

# <codecell>

frameLocs = np.sort(glob.glob(dataFolder + dataSet + "/frame-*.png"))
frameSize = np.array(Image.open(frameLocs[0])).shape[0:2]
numOfFrames = len(frameLocs)
print numOfFrames, frameSize
medianImage = np.zeros((frameSize[0], frameSize[1], 3), dtype=np.uint8)

# <headingcell level=2>

# COMPUTE IMAGE MEDIAN

# <codecell>

allFrames = np.zeros((frameSize[0], frameSize[1], numOfFrames), dtype=np.uint8)
channel = 2
for i in xrange(len(frameLocs)) :
    allFrames[:, :, i] = np.array(Image.open(frameLocs[i]))[:, :, channel]
    if np.mod(i, 100) == 0 :
        sys.stdout.write('\r' + "Loaded image " + np.string_(i) + " (" + np.string_(len(frameLocs)) + ")")
        sys.stdout.flush()

# <codecell>

medianImage[:, :, channel] = np.median(allFrames, axis=-1)

# <codecell>

figure(); imshow(medianImage)

# <codecell>

Image.fromarray(np.array(medianImage, dtype=np.uint8)).save(dataFolder + dataSet + "median.png")

# <codecell>

figure(); imshow(np.array(tmp, dtype=np.uint8))

# <headingcell level=2>

# RENDER SPRITE ON BACKGROUND

# <codecell>

basePath = "/media/ilisescu/Data1/PhD/data/havana/"
bgImage = np.array(Image.open(basePath+"median.png"))
for i in np.arange(800, 800+476) :
    currentFrame = np.array(Image.open(basePath+"bus1/bus1-frame-{0:05d}.png".format(i)))
    spriteLoc = np.argwhere(currentFrame[:, :, -1] != 0)
    alphas = currentFrame[spriteLoc[:, 0], spriteLoc[:, 1], -1].reshape((len(spriteLoc), 1)).repeat(3, axis=-1)
    
    finalFrame = np.copy(bgImage)
    finalFrame[spriteLoc[:, 0], spriteLoc[:, 1], :] = (currentFrame[spriteLoc[:, 0], spriteLoc[:, 1], 0:-1]*(alphas/255.0) + 
                                                       bgImage[spriteLoc[:, 0], spriteLoc[:, 1], :]*((255-alphas)/255.0))

>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
    
matches_dir = os.path.join(output_dir, "matches")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

# Create the ouput/matches folder if not present
if not os.path.exists(matches_dir):
  os.mkdir(matches_dir)

print ("1. Intrinsics analysis") 
pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-d", camera_file_params, "-c", "3"] )
pIntrisics.wait()

print ("2. Compute features")
#pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT", "-f" , "1"] )
#pFeatures.wait()

print ("2. Compute matches")
#pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-f", "1"] )
#pMatches.wait()

reconstruction_dir = os.path.join(output_dir,"reconstruction_sequential")
print ("3. Do Incremental/Sequential reconstruction") #set manually the initial pair to avoid the prompt question
# pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_IncrementalSfM"),  "-i", matches_dir+"/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir, "-a", "100_7104.JPG", "-b", "100_7105.JPG"] )
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_IncrementalSfM"),  "-i", matches_dir+"/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir] ) #, "-a", "frame-03700.png", "-b", "frame-03800.png"] )
pRecons.wait()

print ("5. Colorize Structure")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.json", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
pRecons.wait()

print ("4. Structure from Known Poses (robust triangulation)")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),  "-i", reconstruction_dir+"/sfm_data.json", "-m", matches_dir, "-o", os.path.join(reconstruction_dir,"robust.ply")] )
pRecons.wait()

# Reconstruction for the global SfM pipeline
# - global SfM pipeline use matches filtered by the essential matrices
# - here we reuse photometric matches and perform only the essential matrix filering
print ("2. Compute matches (for the global SfM Pipeline)")
pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-r", "0.8", "-g", "e"] )
pMatches.wait()

reconstruction_dir = os.path.join(output_dir,"reconstruction_global")
print ("3. Do Global reconstruction")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GlobalSfM"),  "-i", matches_dir+"/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir] )
pRecons.wait()

print ("5. Colorize Structure")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.json", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
pRecons.wait()

print ("4. Structure from Known Poses (robust triangulation)")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),  "-i", reconstruction_dir+"/sfm_data.json", "-m", matches_dir, "-o", os.path.join(reconstruction_dir,"robust.ply")] )
pRecons.wait()


