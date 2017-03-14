# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import glob


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
DICT_SEMANTICS_NAMES = "semantics_names"
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

for seqLoc in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-*.npy")) :
    tmp = np.load(seqLoc).item()
    numFrames = len(tmp[DICT_FRAMES_LOCATIONS].keys())
#     print numFrames, seqLoc
    if DICT_LABELLED_FRAMES in tmp.keys() and DICT_NUM_EXTRA_FRAMES in tmp.keys() :
        for i in xrange(len(tmp[DICT_LABELLED_FRAMES])) :
            listToKeep = []
            for j in xrange(len(tmp[DICT_LABELLED_FRAMES][i])) :
                if tmp[DICT_LABELLED_FRAMES][i][j] >= 0 and tmp[DICT_LABELLED_FRAMES][i][j] < numFrames :
                    listToKeep.append(j)
                
#             print tmp[DICT_LABELLED_FRAMES][i], listToKeep,
            tmp[DICT_LABELLED_FRAMES][i] = [tmp[DICT_LABELLED_FRAMES][i][j] for j in listToKeep]
            tmp[DICT_NUM_EXTRA_FRAMES][i] = [tmp[DICT_NUM_EXTRA_FRAMES][i][j] for j in listToKeep]
#             print tmp[DICT_LABELLED_FRAMES][i], tmp[DICT_NUM_EXTRA_FRAMES][i]
    np.save(tmp[DICT_SEQUENCE_LOCATION], tmp)

