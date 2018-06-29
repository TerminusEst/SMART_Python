"""
    Python Script which contains functions for tracking sunspots. Used with Tracker.py
    Author: Sean Blake (blakese@tcd.ie)
    
    Thanks to Sophie Murray and Tadhg Garton
"""

from pylab import *
import sunpy.map
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from copy import deepcopy
import operator

import astropy.units 
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

import SS_tracker_module as SS
from sunpy.physics.differential_rotation import solar_rotate_coordinate

###############################################################################

class SunSpot:
    """The Sunspot class will contain all of the information needed for each sunspot. 
    Sunspot properties include:

    Parameters
    -----------
    number = initial identifier number. This will be updated to track the SS over time.
    size = size of sunspot in pixels
    centroid = x-y centroid in terms of pixels

    """

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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def datetime_from_json(data):
    """Extracting a datetime object from the SMART json file"""

    a = data['meta']['dateobs']
    year = int(a[:4])
    month = int(a[4:6])
    day = int(a[6:8])
    hour = int(a[9:11])
    time1 = datetime.datetime(year, month, day, hour)
    return time1

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def get_sunspot_data(yy, time1):
    """ This function takes processed .fits file, and returns a list of Sunspot objects, with properties.
    
    Parameters
    -----------
    yy = sunpy.map.Map("example_detections.fits").data
    
    time1 = datetime.datetime object from corresponding data
    
    Returns
    -----------
    num_of_ss = number of individual sunspots in the data
    
    master = a list of Sunspot objects (see above). Initially, the .number property
             of each of these subspots will be listed 1, 2, 3... etc.
    
    -----------------------------------------------------------------
    
    """
    master = []
    num_of_ss = np.max(yy.flatten())    # get number of different SS's
    centroids = []
    sizes = []
    numbers = []

    for i in np.arange(1, num_of_ss + 1):   # for each SS:
        temp_sunspot = SunSpot(1, 1, 1)
        copy_yy = np.array(yy, copy = True)
        copy_yy[copy_yy != i] = 0   # get only points == i
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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def euclidean_dist(ss1, ss2):
    """ Calculate distance between two points """
    lat1, lon1 = ss1.centroid
    lat2, lon2 = ss2.centroid

    return sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def distance_matrix(sunspots1, sunspots2):
    """ Create a euclidean distance matrix between to lists of sunspots 
        
    Parameters
    -----------
    sunspots1 = list of sunspots from time t-1, with estimated positions at time t due 
                to rotation
    
    sunspots2 = list of sunspots at some time t (i.e., the sunspots actually observed at time t) 
    

    Returns
    -----------
    highest_number = the highest numbered sunspot in either of the inputs (to prevent duplicate
                     numbering)
                     
    output = the distance matrix between the two input lists of sunspots.
             This distance matrix has rows = len(sunspots1), columns = len(sunspots2)

    -----------------------------------------------------------------
    """
    
    N1 = len(sunspots1)
    N2 = len(sunspots2)

    distance_matrix = np.zeros((N1, N2))

    for i in list(range(N1)):
        for j in list(range(N2)):

            distance_matrix[i, j] = euclidean_dist(sunspots1[i], sunspots2[j])

    return distance_matrix

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def pixels_to_latlon(x_pos, y_pos):
    """ convert pixels to approximate lat long """
    radius = 489.5
    x_pos_from_center = -1*(512.5 - x_pos)
    angle_1 = np.rad2deg(np.arccos(x_pos_from_center/radius))

    y_pos_from_center = -1*(512.5 - y_pos)
    angle_2 = np.rad2deg(np.arccos(y_pos_from_center/radius)) 

    return -1*angle_1, angle_2

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def latlon_to_pixels(lon, lat):
    """ converts lat long values to pixels """
    radius = 489.5
    x_pos_from_center = radius * np.cos(np.deg2rad(lon))
    x_pos = x_pos_from_center + 512.5
    
    y_pos_from_center = radius * np.cos(np.deg2rad(lat))
    y_pos = y_pos_from_center + 512.5
    
    return x_pos, y_pos

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def rotate_SS(x_pos, y_pos, time1, time2):
    # Rotate centroid of SS for given times
    """Rotates centroid of a sunspot for given times

    Parameters
    -----------
    x_pos, y_pos = centroid coordinates in pixel
    
    time1, time2 = datetime objects that indicate time of x_pos, y_pos (time1)
                   and time to rotate sunspot to (time2)

    Returns
    -----------
    new_x, new_y = estimated centroid position at time2 for sunspot

    -----------------------------------------------------------------

    """
    lon, lat = pixels_to_latlon(x_pos, y_pos)   # convert to latlon

    hgx = astropy.units.Quantity(lon - 90., astropy.units.deg)
    hgy = astropy.units.Quantity(lat - 90., astropy.units.deg)

    start_coord = SkyCoord(hgx, hgy, frame=frames.HeliographicStonyhurst, obstime = time1)
    rotated_coord = solar_rotate_coordinate(start_coord, time2)

    new_lat = rotated_coord.lat.value + 90.
    new_lon = rotated_coord.lon.value + 90.

    new_x, new_y = latlon_to_pixels(new_lon, new_lat)

    return new_x, new_y

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def make_overlap_matrix_V2(old_SS, new_SS):
    """ Make a overlap matrix between two lists of sunspots. Rotates old sunspots 
    based on time of new sunspots, and sees which sunspots overlap.
        
    Parameters
    -----------
    old_SS = list of sunspots at some time t-1
    
    sunspots2 = list of sunspots at some time t
    

    Returns
    -----------
    overlap_matrix = matrix of overlap (in pixels)

    -----------------------------------------------------------------
    """
    overlap_matrix = np.zeros((len(old_SS), len(new_SS)))   # empty overlap matrix

    time1 = old_SS[0].timestamp # the two times needed
    time2 = new_SS[0].timestamp

    for i1, v1 in enumerate(old_SS):    # for each old sunspot, rotate it
        size1 = v1.size
        x_pos1, y_pos1 = v1.centroid
        x_pos2, y_pos2 = rotate_SS(x_pos1, y_pos1, time1, time2)
        horiz_extent = v1.max_x - v1.min_x
        difference = x_pos2 - x_pos1

        mask1 = np.roll(v1.mask, int(difference), axis = 1)

        for i2, v2 in enumerate(new_SS):    # for each new sunspot, check if there is overlap
            mask2 = v2.mask

            overlap = np.sum(mask1 * mask2)
            overlap_matrix[i1][i2] = overlap

    return overlap_matrix

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def assign_numbers(old_SS, new_SS, overlap_matrix, num_of_ss):
    """ Update the numbers in a list of new sunspots, based on the overlap
        
    Parameters
    -----------
    old_SS = list of sunspots at some time t-1
    
    sunspots2 = list of sunspots at some time t
    
    overlap_matrix = matrix of overlap (in pixels)

    num_of_ss = highest number sunspot yet seen

    Returns
    -----------
    old_SS = list of sunspots at some time t-1

    new_SS = list of sunspots at time t, now with updated identifying numbers

    num_of_ss = highest number of sunspot yet seen (may have changed from input)

    -----------------------------------------------------------------
    """

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

    # Now to sort out competing claims (where there are two overlaps)
    # whichever old-new sunspot pair has the highest overlap will get the number
    # sort by intersection area
    SS_claims = sorted(SS_claims, key = operator.itemgetter(2), reverse = True)

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

    return old_SS, new_SS, num_of_ss

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
###############################################################################
###############################################################################
