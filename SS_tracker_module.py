# TRY TO USE MASKS

from pylab import *
import sunpy.map
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from copy import deepcopy
import operator

import astropy.units 
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

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
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames


###############################################################################
###############################################################################

class SunSpot:
    def __init__(self, number, size, centroid):
        self.size = size
        self.number = number
        self.centroid = centroid
        self.mask = False
        self.x_points = []
        self.y_points = []
        self.timestamp = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None


def get_sunspot_data(yy, time1):
    master = []
    num_of_ss = np.max(yy.flatten())    # get number of different SS's
    centroids = []
    sizes = []
    numbers = []

    for i in np.arange(1, num_of_ss + 1):
        temp_sunspot = SunSpot(1, 1, 1)
        copy_yy = np.array(yy, copy = True)
        copy_yy[copy_yy != i] = 0   # points
        copy_yy[copy_yy == i] = 1

        indices_x, indices_y = np.where(yy == i)

        max_lat = np.max(indices_x)
        min_lat = np.min(indices_x)
        mean_lat = max_lat - (max_lat - min_lat)/2
        
        max_lon = np.max(indices_y)
        min_lon = np.min(indices_y)
        mean_lon = max_lon - (max_lon - min_lon)/2
        
        temp_sunspot.mask = copy_yy
        temp_sunspot.centroid = [mean_lon, mean_lat]
        temp_sunspot.size = len(indices_x)
        temp_sunspot.number = i
        temp_sunspot.x_points = indices_x
        temp_sunspot.y_points = indices_y
        temp_sunspot.timestamp = time1
        temp_sunspot.min_x = min_lon
        temp_sunspot.max_x = max_lon
        temp_sunspot.min_y = min_lat
        temp_sunspot.max_y = max_lat

        master.append(temp_sunspot)

    return num_of_ss, master

def euclidean_dist(ss1, ss2):
    lat1, lon1 = ss1.centroid
    lat2, lon2 = ss2.centroid

    return sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def distance_matrix(sunspots1, sunspots2):

    N1 = len(sunspots1)
    N2 = len(sunspots2)

    distance_matrix = np.zeros((N1, N2))

    for i in list(range(N1)):
        for j in list(range(N2)):

            distance_matrix[i, j] = euclidean_dist(sunspots1[i], sunspots2[j])

    return distance_matrix

def read_in_ss_file(filename):
    f = open(filename, 'r')
    data = f.readlines()
    f.close()

    output = []
    for i in data[1:]:
        ss_data = i.split("\t")
        sunspot_object = SunSpot(float(ss_data[0]), float(ss_data[1]), [float(ss_data[2]), float(ss_data[3])])
        output.append(sunspot_object)

    highest_number = float(data[0])

    return highest_number, output
################################################################################
# Functions for rotating the SS

def pixels_to_latlon(x_pos, y_pos):
    radius = 489.5
    x_pos_from_center = -1*(512.5 - x_pos)
    angle_1 = np.rad2deg(np.arccos(x_pos_from_center/radius))

    y_pos_from_center = -1*(512.5 - y_pos)
    angle_2 = np.rad2deg(np.arccos(y_pos_from_center/radius)) 

    return -1*angle_1, angle_2

def latlon_to_pixels(lon, lat):
    radius = 489.5
    x_pos_from_center = radius * np.cos(np.deg2rad(lon))
    x_pos = x_pos_from_center + 512.5
    
    y_pos_from_center = radius * np.cos(np.deg2rad(lat))
    y_pos = y_pos_from_center + 512.5
    
    return x_pos, y_pos

def rotate_SS(x_pos, y_pos, time1, time2):
    # Rotate centroid of SS for given times

    lon, lat = pixels_to_latlon(x_pos, y_pos)

    hgx = astropy.units.Quantity(lon - 90., astropy.units.deg)
    hgy = astropy.units.Quantity(lat - 90., astropy.units.deg)

    start_coord = SkyCoord(hgx, hgy, frame=frames.HeliographicStonyhurst, obstime = time1)
    rotated_coord = solar_rotate_coordinate(start_coord, time2)

    new_lat = rotated_coord.lat.value + 90.
    new_lon = rotated_coord.lon.value + 90.

    new_x, new_y = latlon_to_pixels(new_lon, new_lat)

    return new_x, new_y

################################################################################


def make_overlap_matrix(old_SS, new_SS):
    # make pixel-overlap matrix. #OLD = Y-axis, #NEW = X=axis
    overlap_matrix = np.zeros((len(old_SS), len(new_SS)))

    for i1, v1 in enumerate(old_SS):
        for i2, v2 in enumerate(new_SS):

            overlap = np.sum(v1.mask * v2.mask)
            overlap_matrix[i1][i2] = overlap

    return overlap_matrix

def make_overlap_matrix_V1(old_SS, new_SS):
    # make pixel-overlap matrix. #OLD = Y-axis, #NEW = X=axis
    overlap_matrix = np.zeros((len(old_SS), len(new_SS)))

    time1 = old_SS[0].timestamp
    time2 = new_SS[0].timestamp

    for i1, v1 in enumerate(old_SS):
        size1 = v1.size
        x_pos1, y_pos1 = v1.centroid
        x_pos2, y_pos2 = rotate_SS(x_pos1, y_pos1, time1, time2)
        horiz_extent = v1.max_x - v1.min_x
        difference = x_pos2 - x_pos1

        if size1 < 350:
            mask1 = np.zeros(shape(old_SS[0].mask))
            for kkk in arange(0.5, 3., .1):
                temp_mask = np.roll(v1.mask, int(difference * kkk), axis = 1)
                mask1 += temp_mask
        
            mask1[mask1 != 0] = 1   # points
        else:
            mask1 = np.roll(v1.mask, int(difference), axis = 1)

        for i2, v2 in enumerate(new_SS):
            mask2 = v2.mask

            overlap = np.sum(mask1 * mask2)
            overlap_matrix[i1][i2] = overlap

    return overlap_matrix

def make_overlap_matrix_V2(old_SS, new_SS):
    # make pixel-overlap matrix. #OLD = Y-axis, #NEW = X=axis
    overlap_matrix = np.zeros((len(old_SS), len(new_SS)))

    time1 = old_SS[0].timestamp
    time2 = new_SS[0].timestamp

    for i1, v1 in enumerate(old_SS):
        size1 = v1.size
        x_pos1, y_pos1 = v1.centroid
        x_pos2, y_pos2 = rotate_SS(x_pos1, y_pos1, time1, time2)
        horiz_extent = v1.max_x - v1.min_x
        difference = x_pos2 - x_pos1

        mask1 = np.roll(v1.mask, int(difference), axis = 1)

        for i2, v2 in enumerate(new_SS):
            mask2 = v2.mask

            overlap = np.sum(mask1 * mask2)
            overlap_matrix[i1][i2] = overlap

    return overlap_matrix

def assign_numbers(old_SS, new_SS, overlap_matrix, num_of_ss):
    # Now assign new sunspost numbers based off the old sunspots
    # if column is empty => new sunspot
    # if row is empty => old sunspot is retired
    new_SS_accounted_for, old_SS_accounted_for = [], []
    SS_claims = []

    # loop over new SS
    for icolumn, vcolumn in enumerate(overlap_matrix.T):
        if sum(vcolumn) == 0:   # doesnt overlap with any old SS -> new SS
            new_SS[icolumn].number = num_of_ss + 1
            num_of_ss += 1
            new_SS_accounted_for.append(icolumn)
            continue

        max_vcolumn = max(vcolumn)
        if sum(vcolumn) == max_vcolumn:    # only 1 in column
            row = list(vcolumn).index(max_vcolumn)

            if sum(overlap_matrix[row]) == max_vcolumn:  # only 1 in row
                # in this scenario, new_SS[icolumn] = old_SS[row]
                new_SS[icolumn].number = old_SS[row].number

                new_SS_accounted_for.append(icolumn)
                old_SS_accounted_for.append(row)
                continue

            else:   # two new sunspots have claim to an old sunspot
                SS_claims.append([row, icolumn, max_vcolumn])
                
        else: # more than 1 overlap in column -> 2 SS merging
            for irow, vrow in enumerate(vcolumn):
                if vrow != 0:
                    SS_claims.append([irow, icolumn, vrow])

    # Now to sort out competing claims
    # sort by intersection area
    SS_claims = sorted(SS_claims, key = operator.itemgetter(2), reverse = True)

    #print(SS_claims)
    #print("\n")

    for i in SS_claims:
        old_numb = i[0]
        new_numb = i[1]

        if (old_numb not in old_SS_accounted_for) and (new_numb not in new_SS_accounted_for):
            new_SS[new_numb].number = old_SS[old_numb].number

            old_SS_accounted_for.append(old_numb)
            new_SS_accounted_for.append(new_numb)

        elif (old_numb in old_SS_accounted_for) and (new_numb not in new_SS_accounted_for):
            new_SS[new_numb].number = num_of_ss + 1
            num_of_ss += 1
            new_SS_accounted_for.append(new_numb)
            old_SS_accounted_for.append(new_numb)

        elif (old_numb not in old_SS_accounted_for) and (new_numb in new_SS_accounted_for):
            pass

    #print("NEW ", new_SS_accounted_for)
    #print("old ", old_SS_accounted_for)
    #print(overlap_matrix)

    return old_SS, new_SS, num_of_ss

def write_out(out_filename, num_of_ss, old_SS):
    f = open(out_filename, 'w')
    f.write(str(num_of_ss) + "\n")

    for i in old_SS:
        mystr = str(i.number) + "\t" + str(i.size) + "\t" + str(i.centroid[0]) + "\t" + str(i.centroid[1]) + "\n"

        f.write(mystr)
    f.close()

###############################################################################
###############################################################################












