# --------------------------------- #
#    Renaming BrainVision Files     #
# --------------------------------- #

# --------------------------------- #
#             Overview              #
# --------------------------------- #
#  Each time one makes a recording  #
#  with the BrainVision Recorder    #
#  software in the RecogNize lab,   #
#  three files are generated: one   #
#  with the extension .vhdr, one    #
#  with the extension .eeg and one  #
#  with the extension .vmrk. One    #
#  should be careful when renaming  #
#  those files, since they contain  #
#  references to each other. The    #
#  code in this file can be used    #
#  to rename BrainVision files in   #
#  a safe, quick and easy manner.   #
# --------------------------------- #
#  a.n.j.p.m.haas@gmail.com (2021)  #
# --------------------------------- #

# ============ SETTINGS =========== #

# What is the name of the .vhdr file
#  that you would like to rename? The
#  corresponding .eeg file and the
#  corresponding .vmrk file will also
#  be renamed (in a similar manner).
fileName = 'oldFileName.vhdr'

# Where is that .vhdr file stored?
#  The corresponding .eeg file and
#  the corresponding .vmrk file should
#  be stored in the same directory.
fileLocation = '../../Data/Batch 1 (P01-P20) [2019]/'

# What would you like the renamed
#  .vhdr file and its .eeg and
#  .vmrk siblings to be called?
#  The three new files will be
#  stored in the same directory
#  as the three original files.
newFileName = 'newFileName.vhdr'

# ============= CODE ============== #

### ---------- Step A ----------- ###

# We import the Python modules we need.
import mne
from os import path
from mne_bids.copyfiles import copyfile_brainvision

### ---------- Step B ----------- ###

# What is the path to the original .vhdr file?
originalFilePath = fileLocation + fileName

### ---------- Step C ----------- ###

# We check whether the specified file
#  and its siblings actually exist.
#  Does the specified .vhdr file exist?
if not path.exists(originalFilePath):
    print("\n[ERROR] The following file could not be found: \'{}\'.".format(originalFilePath))
    exit()

# Does the corresponding
#  .eeg file also exist?
originalFilePath_eeg = originalFilePath[:len(originalFilePath) - 4] + 'eeg'
if not path.exists(originalFilePath_eeg):
    print("\n[ERROR] The following file could not be found: \'{}\'.".format(originalFilePath_eeg))
    exit()

# Does the corresponding
#  .vmrk file also exist?
originalFilePath_vmrk = originalFilePath[:len(originalFilePath) - 4] + 'vmrk'
if not path.exists(originalFilePath_vmrk):
    print("\n[ERROR] The following file could not be found: \'{}\'.".format(originalFilePath_vmrk))
    exit()

### ---------- Step D ----------- ###

# We indicate where we want the
#  renamed files to be stored
#  and what they should be called.
newFilePath = originalFilePath[:len(originalFilePath) - len(fileName)] + newFileName

### ---------- Step E ----------- ###

# We rename the files. Copies are
#  made of the original files. The
#  copies bear the new names.
copyfile_brainvision(originalFilePath, newFilePath, verbose=True)

### ---------- Step F ----------- ###

# We make sure that no files were corrupted.
print("\n----------------------------------------------------------------------------------------------------------------------------------")
print("Check 1/2 - Unless this text is followed by an error message (runtime warnings do not count), the original files are still intact.")
print("----------------------------------------------------------------------------------------------------------------------------------\n")
raw = mne.io.read_raw_brainvision(originalFilePath)
print("\n----------------------------------------------------------------------------------------------------------------------------------------")
print("Check 2/2 - Unless this text is followed by an error message (runtime warnings do not count), the new (renamed) files are ready for use.")
print("----------------------------------------------------------------------------------------------------------------------------------------\n")
raw = mne.io.read_raw_brainvision(newFilePath)
print("\n-------------------------------------------------------------------")
print("The code was executed successfully. Three new files were generated:")
print("1. {}".format(newFilePath))
print("2. {}".format(newFilePath[:len(newFilePath) - 4] + 'eeg'))
print("3. {}".format(newFilePath[:len(newFilePath) - 4] + 'vmrk'))
print("-------------------------------------------------------------------")