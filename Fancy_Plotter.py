from pylab import *
import sunpy.map
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from copy import deepcopy
import operator
import SS_tracker_module as SS
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import matplotlib.image as mpimg
import json
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import matplotlib.animation as animation

################################################################################
################################################################################

def datetime_from_json(data):
    # convert timestring to datetime object

    a = data['meta']['dateobs']
    year = int(a[:4])
    month = int(a[4:6])
    day = int(a[6:8])
    hour = int(a[9:11])
    time1 = datetime.datetime(year, month, day, hour)
    return time1

def datetime_from_file_string(a):
    # convert timestring to datetime object

    year = int(a[:4])
    month = int(a[4:6])
    day = int(a[6:8])
    hour = int(a[9:11])
    time1 = datetime.datetime(year, month, day, hour)
    return time1

###############################################################################
input_folder = "/home/blake/Drive/POSTDOC/SMART/json_test/input_data/"
json_folder = "/home/blake/Drive/POSTDOC/SMART/json_test/json_data/"
movie_folder = "/home/blake/Drive/POSTDOC/SMART/json_test/movie_folder/"

filenames = sorted(os.listdir(json_folder))
filename_dates = [datetime_from_file_string(x) for x in filenames]

start_date, end_date = datetime.datetime(2011, 9, 1, 0, 0, 0, 0), datetime.datetime(2012, 9, 4, 1, 1)

date_strings = []
for index, value in enumerate(filename_dates):
    if start_date < value < end_date:
        date_strings.append([filenames[index][:13], value])

# get properties
sunspot_data = {}
for date_string in date_strings:
    json_filename = json_folder + date_string[0] + "_properties.json"
    data = json.load(open(json_filename))

    for key, value in data['posprop']['trueid'].items():
        if str(value) in sunspot_data:
            sunspot_data[str(value)][0].append(date_string[1])
            sunspot_data[str(value)][1].append(data['magprop']['totarea'][str(key)])            
        else:
            sunspot_data[str(value)] = [[date_string[1]], [data['magprop']['totarea'][str(key)]]           ]


#----------------------------------------
# get detection outlines as outline_edges

ims = []
fig = plt.figure(1)

count = 1
for date_string in date_strings:
    filename = input_folder + date_string[0] + "_detections.fits"
    yy1 = sunpy.map.Map(filename).data

    # get outlines of sunspot detections
    outline_edges = np.zeros((1024, 1024))
    num_of_ss = np.max(yy1.flatten())

    for i in np.arange(1, num_of_ss + 1):
        yy_copy = np.copy(yy1)

        mask = yy_copy==i
        yy_copy[~mask] = 0

        # rank 2 structure with full connectivity
        struct = ndimage.generate_binary_structure(2, 2)
        erode = ndimage.binary_erosion(mask, struct)
        edges = mask ^ erode

        outline_edges += edges
    outline_edges = np.ma.masked_where(outline_edges == 0, outline_edges)

    #----------------------------------------
    # get actual image of sun
    filename = input_folder + date_string[0] + "_map.fits"
    yy2 = sunpy.map.Map(filename).data

    #----------------------------------------
    # read in numbers and centroids from json
    # read in json
    json_data = json.load(open(json_folder + date_string[0] + "_properties.json"))
    time2 = datetime_from_json(json_data)
    number_json = list(json_data['posprop']['trueid'].keys())
    number_json_values = [json_data['posprop']['trueid'][i] for i in number_json]


    json_centx, json_centy = [], []
    for i in number_json:
        json_centx.append(json_data['posprop']['xcenarea'][i])
        json_centy.append(json_data['posprop']['ycenarea'][i])

    #----------------------------------------
    # plotting shite

    ax1 = subplot2grid((5, 4), (0, 0), colspan = 4, rowspan = 3)
    im1 = imshow(yy2)
    im2 = imshow(outline_edges, cmap = "Greys", interpolation = "nearest")

    plot(json_centx, json_centy, 'or')
    for x, y, numb in zip(json_centx, json_centy, number_json_values):
            text(x, y, str(numb), fontsize = 24, color = "red")


    title(date_string[0], fontsize = 24)

    ax2 = subplot2grid((5, 4), (3, 0), colspan = 4, rowspan = 2)

    for key, value in sunspot_data.items():
        
        ax2.plot(value[0], value[1])

    axvline(date_string[1], lw = 3, linestyle = "dashed", color = "black")

    count += 1
    savefig(movie_folder + str(count) + ".png", dpi = 100, figsize = (80, 40))
    plt.clf()
    
# aborted attempts to get the above to animate
#ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
#ani.save('sdo_aia.mp4', writer='ffmpeg')














































