"""
    Python Script to track SMART processed sunspots, and assign likely identifying
    numbers. Uses functions from SS_tracker_module.py.

    Author: Sean Blake (blakese@tcd.ie), 2018
    
    Thanks to Sophie Murray and Tadhg Garton

    ----------------------------------------------------------------------------
    TRACKING METHOD:
    
    A list of sunspot objects are created from SMART processed .fits files for 
    some time t. Each sunspot is given an id number. At some later time t+1, 
    another list of sunspot numbers is created. 

    Every possible sunspot pair from the two lists are compared after, the older 
    sunspots have been rotated (to estimate new position due to solar rotation).

    If the rotated sunspots overlap with any of the new sunspots, they are probably
    the same sunspot, and the new sunspot number is updated with the old sunspot
    number.

    If the old, rotated sunspot does not overlap with any new sunspots, it is likely
    retired (or not detected by the SMART program. This could be updated in the
    future)

    If there is a new sunspot which does not overlap with any of the old sunspots,
    it is given a new sunspot number.

    ----------------------------------------------------------------------------

    GENERAL WORKFLOW:
    Point this code to three folders:

    input_folder = contains the SMART processed .fits and properties .json files.
    output_folder = where updated .json files (with new id numbers for sunspots)
                    will be written
    image_folder = where images will be saved

"""

from pylab import *
import sunpy.map
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from copy import deepcopy
import operator
import SS_tracker_module as SS
from sunpy.physics.differential_rotation import solar_rotate_coordinate

import astropy.units as u
import SS_tracker_module as SS

from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

###############################################################################
################################################################################

input_folder = "/home/blake/Drive/POSTDOC/SMART/json_test/input_data/"
output_folder = "/home/blake/Drive/POSTDOC/SMART/json_test/FINAL/output_data/"
image_folder = "/home/blake/Drive/POSTDOC/SMART/json_test/FINAL/output_images/"

filenames = sorted(os.listdir(input_folder))
filename_stems = [x[:14] for x in filenames if "_detections.fits" in x]
extensions = ["properties.json", "detections.fits", "map.fits"]

###############################################################################

# first image
# read in json
json_data = json.load(open(input_folder + filename_stems[0] + extensions[0]))
time1 = SS.datetime_from_json(json_data)

# read in map
yy = sunpy.map.Map(input_folder + filename_stems[0] + extensions[1]).data
master_num_of_ss, old_SS = SS.get_sunspot_data(yy, time1)

# write out json with updated 'trueid' numbers
true_id = {}
for index in range(len(old_SS)):
    true_id[str(index)] = int(old_SS[index].number)

json_data['posprop']['trueid'] = true_id
with open(output_folder + filename_stems[0] + extensions[0], 'w') as outfile:
    json.dump(json_data, outfile)

###############################################################################
# now for the rest of the images
count = 0
for filename_stem in filename_stems[1:]:
    # read in json
    json_data = json.load(open(input_folder + filename_stem + extensions[0]))
    time2 = SS.datetime_from_json(json_data)

    # read in map
    yy2 = sunpy.map.Map(input_folder + filename_stem + extensions[1]).data
    num_of_ss2, new_SS = SS.get_sunspot_data(yy2, time2)

    overlap_matrix = SS.make_overlap_matrix_V2(old_SS, new_SS)
    old_SS, new_SS, master_num_of_ss = SS.assign_numbers(old_SS, new_SS, overlap_matrix, master_num_of_ss)

    # write out json with updated 'trueid' numbers
    true_id = {}
    for index in range(len(new_SS)):
        true_id[str(index)] = int(new_SS[index].number)

    json_data['posprop']['trueid'] = true_id
    with open(output_folder + filename_stem + extensions[0], 'w') as outfile:
        json.dump(json_data, outfile)

    old_SS = deepcopy(new_SS)

    count += 1
    print(count)
    #if count > 50:
    #    break











