# ----------------------------------------- #
#         Bad Channel Identification        #
# ----------------------------------------- #

# ----------------------------------------- #
#                 Overview                  #
# ----------------------------------------- #
#  This code can be used to plot raw EEG    #
#  data in order to identify bad channels.  #
# ----------------------------------------- #
#      a.n.j.p.m.haas@gmail.com (2021)      #
# ----------------------------------------- #

# =============== SETTINGS ================ #

# What is the name of the file that contains
#  the EEG data that you would like to plot?
fileName = '01_F_26_R_AD.vhdr'

# ================= CODE ================== #

### -------------- Step A --------------- ###

# We import the Python modules we need.
import mne
from os import path

### -------------- Step B --------------- ###

# In total, there were 37 participants.
#  Their data is stored in two directories.
mainDirectory = '../..'
subDirectory1 = '/Data/Batch 1 (P01-P20) [2019]/'
subDirectory2 = '/Data/Batch 2 (P21-P37) [2021]/'

# The data for the first 20 participants is stored
#  in subdirectory 1. The data for the remaining 17
#  participants is stored in subdirectory 2. Where can
#  we find the data file that we are looking for?
if int(fileName[0:2]) <= 20:
    filePath = mainDirectory + subDirectory1 + fileName
else:
    filePath = mainDirectory + subDirectory2 + fileName

### -------------- Step C --------------- ###

# We check whether the specified
#  file actually exists.
if not path.exists(filePath):
    print("\nThe following file could not be found: \'{}\'.".format(filePath))
    exit()

### -------------- Step D --------------- ###

# We load the data. Since we make use
#  of BrainVision data, we should apply
#  a non-standard read function here.
raw = mne.io.read_raw_brainvision(filePath)

### -------------- Step E --------------- ###

# We plot the raw EEG data.
raw.plot(block=True)