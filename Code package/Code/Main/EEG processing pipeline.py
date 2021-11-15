# ----------------------------------- #
#       EEG Processing Pipeline       #
# ----------------------------------- #

# ----------------------------------- #
#              Overview               #
# ----------------------------------- #
#  This code was written in 2021 by   #
#  Alexander Haas, under the super-   #
#  vision of dr. Rob van der Lubbe,   #
#  as part of his MSc thesis project  #
#  at the University of Twente. It    #
#  can be used to process the EEG     #
#  data of individuals who have been  #
#  subjected to dr. Van der Lubbe's   #
#  2019 implementation of the Add-n   #
#  task. This code was written with   #
#  one specific research question     #
#  in mind. Large parts of the code   #
#  can easily be recycled, however.   #
#  Readers should feel free to do so. #
# ----------------------------------- #
#   a.n.j.p.m.haas@gmail.com (2021)   #
# ----------------------------------- #

# ============ SETTINGS ============= #

# To process the EEG data of only a
#  few (rather than all) participants,
#  set 'limitedFocus' to 'True' and
#  enter the identification numbers of
#  the participants whose data you want
#  to examine in 'selectedParticipants'.
limitedFocus = False
selectedParticipants = [1, 2, 3]

# To ensure that a participant's data
#  will never be processed (because it
#  is too noisy, for example), please
#  enter their identification number
#  in 'excludedParticipants' below.
excludedParticipants = [22, 34]

# To avoid generating new ICA solutions
#  for all participants (which is very
#  time-consuming) at step 2.2.10 and to
#  make use of old ICA solutions instead,
#  please set 'completeICA' to 'False'.
completeICA = False

# Different scholars have different
#  views on what constitutes theta
#  activity. To accommodate for this,
#  you can set 'thetaRange' yourself
#  here. The first value (in Hz) is
#  the lower limit. The second value
#  (also in Hz) is the upper limit.
thetaRange = [4.0, 7.0]

# =============== CODE =============== #

### ******************************** ###
###            ~ Part 1 ~            ###
###         Essential set-up         ###
### ******************************** ###

### ----------- Step 1.1 ----------- ###

# We import the Python modules we need.
import mne
import os
from os import path
from pathlib import Path
from mne.preprocessing import ICA
import pandas as pd
import numpy as np
import pickle

### ----------- Step 1.2 ----------- ###

# We import some useful information
#  that we stored in other files to
#  avoid cluttering up this code file.
mainDirectory = '../..'

# There were 37 participants in total.
#  For each participant, we have one .vhdr
#  file. The paths to all .vhdr files can
#  be found in '/Miscellaneous/File paths.txt'.
#  For the sake of convenience, let us load
#  those paths into an array called 'files'.
files = []
document = open('../../Miscellaneous/File paths.txt', 'r')
document = document.readlines()
for fileName in document:
    files.append(mainDirectory + fileName.strip())

# Were there any bad channels when we
#  recorded our EEG data? To find out,
#  we can use the code provided in
#  '/Code/Other/Bad channel identifica-
#  tion.py'. My findings (per subject)
#  can be found in '/Miscellaneous/Bad
#  channels.txt'. Let us load them into
#  an array called 'badChannelsPerSubject'.
badChannelsPerSubject = []
document = open('../../Miscellaneous/Bad channels.txt', 'r')
document = document.readlines()
for badChannelSet in document:
    badChannelsPerSubject.append(badChannelSet.strip()[5:].split())

# We will use independent component
#  analysis (ICA) to remove noise from
#  our data. I already identified which
#  components should be removed. My
#  findings (per subject) can be found in
#  '/Miscellaneous/Unwanted components.txt'.
#  Let us load them into an array called
#  'unwantedComponentsPerSubject'.
unwantedComponentsPerSubject = []
document = open('../../Miscellaneous/Unwanted components.txt', 'r')
document = document.readlines()
for unwantedComponentSet in document:
    components = unwantedComponentSet.strip()[5:].split()
    unwantedComponentsPerSubject.append(components)

### ******************************** ###
###            ~ Part 2 ~            ###
###    Subject-level computations    ###
### ******************************** ###

### ----------- Step 2.1 ----------- ###

# For all participant-condition-electrode
#  combinations, we want to calculate power
#  spectral density scores for a wide range
#  of sampling frequencies (0 Hz - 250 Hz).
#  We will do that in this part of the code.
#  We will store the power spectral density
#  scores in an array called 'powerScoresPer-
#  Subject'. We will store the sampling fre-
#  quencies (which will be identical for all
#  participant-condition-electrode tuples)
#  in an array called 'samplingFrequencies'.
powerScoresPerSubject = []
samplingFrequencies = []

### ----------- Step 2.2 ----------- ###

# Let's have a look at all files, and hence
#  all subjects, one by one in a special loop.
for file in files:

    ### ---------- Step 2.2.1 ---------- ###

    # We can derive the identification number
    #  of this subject from the name of their
    #  .vhdr file. We will do that immediately.
    participantNumber = file[-17:-15]

    ### ---------- Step 2.2.2 ---------- ###

    # Did we set 'limitedFocus' to 'True'?
    if limitedFocus:
        # Did the current subject make the selection?
        if int(participantNumber) not in selectedParticipants:
            #  We move on to the next participant.
            continue

    # Should we skip this participant?
    if int(participantNumber) in excludedParticipants:
        # We move on to the next participant.
        continue

    ### ---------- Step 2.2.3 ---------- ###

    # Does the .vhdr file actually exist?
    if not path.exists(file):
        print(print("[ERROR] The file \'{}\' could not be found".format(file)))
        exit()

    # We recorded our data with the Brain-
    #  Vision Recorder software. The files
    #  we refer to in the array we called
    #  'files' (step 1.2) are .vhdr files.
    #  For each participant, the BrainVision
    #  Recorder software we used also gene-
    #  rated two other (eponymous) files that
    #  are also important: a .eeg file and a
    #  .vmrk file. The .vhdr file merely con-
    #  tains metadata. The .eeg file contains
    #  the raw data that we need. The .vmrk
    #  file contains information about events
    #  (e.g. button presses and stimulus on-
    #  sets) that occurred during the experi-
    #  ment. Let us check whether the .eeg
    #  file and the .vmrk file, which should
    #  be located in the same folder as the
    #  corresponding .vhdr file, also exist.
    eegFile = file[:len(file) - 4] + 'eeg'
    vmrkFile = file[:len(file) - 4] + 'vmrk'
    if not path.exists(eegFile):
        print(print("[ERROR] The file \'{}\' could not be found".format(eegFile)))
        exit()
    elif not path.exists(vmrkFile):
        print(print("[ERROR] The file \'{}\' could not be found".format(vmrkFile)))
        exit()

    ### ---------- Step 2.2.4 ---------- ###

    # We load the data. Since we make use
    #  of BrainVision data, we should apply
    #  a non-standard read function here.
    raw = mne.io.read_raw_brainvision(file, preload=True)

    # We can inspect the loaded data.
    if False:
        print(raw.info)
        print(raw.info.ch_names)

    ### ~~~~~~~~ Pre-processing ~~~~~~~~ ###

    ### ---------- Step 2.2.5 ---------- ###

    # We have 32 EEG channels and 2 MISC
    #  channels. The two MISC channels are
    #  labeled 'hEOG' and 'vEOG'. They only
    #  contain useful data for the first 20
    #  participants or so. We want to treat
    #  each data file in a similar manner,
    #  so let us simply discard the two
    #  MISC channels for all participants.
    raw.drop_channels(['hEOG', 'vEOG'])

    ### ---------- Step 2.2.6 ---------- ###

    # When we recorded our data, we used TP8
    #  as our reference electrode. It would
    #  be better to make use of an average
    #  reference, however, since that would
    #  reduce a potential bias towards brain
    #  activity in the left hemisphere. We
    #  add TP8 to our set of electrodes and
    #  then calculate an average reference.
    mne.add_reference_channels(raw, ref_channels=['TP8'], copy=False)
    raw.set_eeg_reference(ref_channels='average')

    ### ---------- Step 2.2.7 ---------- ###

    # We should indicate how the EEG electrodes
    #  were positioned on the subject's head (i.e.
    #  what electrode montage we used). We
    #  made use of the so-called 10-20 system.
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

    # We can visualise our electrode montage.
    if False:
        raw.plot_sensors(kind='topomap', ch_type='eeg', block=True)
        raw.plot_sensors(kind='3d', ch_type='eeg', block=True)

    ### ---------- Step 2.2.8 ---------- ###

    # During the experiment, many events
    #  (e.g. button presses and stimulus
    #  onsets) occurred. Let us extract all
    #  event information for the current
    #  participant from the data and store
    #  it in an array called 'events'.
    events, notNeeded = mne.events_from_annotations(raw)

    # Each type of event is described by a
    #  so-called 'stimulus code'. Different
    #  types of events are described by
    #  different stimulus codes. Let us in-
    #  dicate what each stimulus code means.
    event_dictionary = \
        {'Required1': 1, 'Required2': 2,
         'Required3': 3, 'Required4': 4,
         'Required5': 5, 'Required6': 6,
         'Required7': 7, 'Required8': 8,
         'Required9': 9, 'Pressed0': 210,
         'Pressed1': 201, 'Pressed2': 202,
         'Pressed3': 203, 'Pressed4': 204,
         'Pressed5': 205, 'Pressed6': 206,
         'Pressed7': 207, 'Pressed8': 208,
         'Pressed9': 209,  'PressedSB': 211,
         'Add0_StimulusAppears': 100,
         'Add0_StimulusDisappears': 150,
         'Add1_StimulusAppears': 101,
         'Add1_StimulusDisappears': 151,
         'Add2_StimulusAppears': 102,
         'Add2_StimulusDisappears': 152,
         'Practice': 155}

    # There is one stimulus code that we did
    #  not include in the above dictionary
    #  yet: '10' (meaning: 'Required0'). That
    #  code was (by accident) not included in
    #  the data for participant 1. It was in-
    #  cluded in the data for the remaining 36
    #  participants, however, so we will add
    #  the code to their event dictionaries.
    if int(participantNumber) != 1:
        event_dictionary['Required0'] = 10

    # We can visualise which events
    #  occurred at which points in time.
    if False:
        figure = mne.viz.plot_events(events,
                                     event_id=event_dictionary,
                                     sfreq=raw.info['sfreq'],
                                     first_samp=raw.first_samp)

    ### ---------- Step 2.2.9 ---------- ###

    # Were there any bad channels when
    #  we recorded this participant's
    #  brain activity? At step 1.2 above,
    #  we loaded this information into an
    #  array called 'badChannelsPerSubject'.
    #  Let us extract the information that
    #  we need from that array and link it
    #  to the current subject's EEG data.
    #  We will interpolate the bad channels
    #  later, at step 2.2.14 of this code.
    raw.info['bads'] = badChannelsPerSubject[int(participantNumber) - 1]

    ### ---------- Step 2.2.10 --------- ###

    # Eye blinks, eye movements, heartbeats
    #  and environmental factors may have
    #  caused there to be artefacts (bits
    #  of noise) in our data. We can try to
    #  get rid of those artefacts by means
    #  of a technique known as independent
    #  component analysis (ICA). If we set
    #  'completeICA' to 'True' earlier, we
    #  will now generate a new ICA solution
    #  for this participant. Please note that
    #  this may take some time (~90 seconds).
    if completeICA:

        #  We first make a copy of our data, which
        #  we will use to create an ICA solution.
        raw_copy = raw.copy()

        # We need to remove all major frequency
        #  drifts from the copy of our data, since
        #  such frequency drifts can make it hard
        #  to create an ICA solution. The reason
        #  we made a copy of our data earlier, is
        #  that we do not want to remove any drifts
        #  from our original data yet at this point.
        raw_copy.load_data().filter(l_freq=0.1, h_freq=30)

        # We will now create the ICA solution for
        #  the current participant's data. We make
        #  use of the 'FastICA' algorithm, since I
        #  found (after several trial sessions) that
        #  this algorithm tends to converge faster
        #  than its key competitors: the 'infomax'
        #  algorithm and the 'Picard' algorithm.
        #  Since 'FastICA' does not converge for
        #  participant 27 (for unknown reasons),
        #  we will use 'Picard' for that subject.
        algorithm = 'fastica'
        if int(participantNumber) == 27:
            algorithm = 'picard'
        numberOfComponents = raw.info['nchan']-len(raw.info['bads'])-1
        ica = ICA(n_components=numberOfComponents, random_state=91, method=algorithm)
        ica.fit(raw_copy)

        # Let us store the ICA solution in a folder
        #  called '/Output/ICA solutions' so we can
        #  use it again in the future.
        with open('../../Output/ICA solutions/P' +
                  participantNumber + '.data', 'wb') as filehandle:
            pickle.dump(ica, filehandle)

    # If we set 'completeICA' to 'False' earlier,
    #  we will not generate a new ICA solution for
    #  this participant. Instead, we will make use
    #  of a solution that we already found earlier.
    if not completeICA:

        # Let us load the ICA solution from
        #  the folder we stored it in earlier.
        with open('../../Output/ICA solutions/P' +
                  participantNumber + '.data', 'rb') as filehandle:
            ica = pickle.load(filehandle)

    # When we created our ICA solution for the
    #  current participant, we essentially
    #  tried to split up that participant's
    #  EEG data into various independent parts
    #  (or 'components'). We can now examine
    #  all of those components one by one.
    if False:
        raw.load_data()
        ica.plot_components()
        ica.plot_sources(raw, block=True)

    # We may want to get rid of some of the
    #  components we have identified, such
    #  as components that seem to have captured
    #  ECG or EOG (rather than EEG) activity. I
    #  already identified all unwanted components
    #  for each subject. We loaded this information
    #  into an array called 'unwantedComponentsPer-
    #  Subject' at step 1.2. Let us extract the
    #  information that we need from that array.
    ica.exclude = [int(i) for i in unwantedComponentsPerSubject[int(participantNumber) - 1]]

    # We are now ready to apply the ICA solution
    #  that we created to our original data. This
    #  essentially means that we are now ready to
    #  reconstruct our original EEG data, this
    #  time with much less noise. Let us do this.
    ica.apply(raw)

    ### ---------- Step 2.2.11 --------- ###

    # We now filter all major frequency
    #  drifts from our data, to further
    #  enhance the data's overall quality.
    raw.load_data().filter(l_freq=0.1, h_freq=30)

    ### ~~~~~~~~~~~ Epoching ~~~~~~~~~~~ ###

    ### ---------- Step 2.2.12 --------- ###

    # An epoch is a segment of EEG data
    #  that is centered around an event.
    #  Let us extract epochs from this
    #  subject's data. By default, one
    #  epoch will be created for each
    #  event. We cannot change this, even
    #  though we are only interested
    #  in epochs centered around a spe-
    #  cific type of event: 'Add[N]_Stim-
    #  ulusAppears' with N ∈ {0, 1, 2}.
    #  We can determine how long each
    #  epoch should be, however. The
    #  settings that are used here were
    #  chosen because they seem suitable
    #  for the epochs we are interested
    #  in. For details, please see
    #  sections 2 and 4 of my thesis.
    epochs = mne.Epochs(raw, events,
                        event_id=event_dictionary,
                        tmin=-0.5, tmax=4.0, preload=True)

    ### ---------- Step 2.2.13 --------- ###

    # We already tried to clean our data
    #  in various ways, but unfortunately
    #  there may still be some artefacts
    #  left. We will now throw away all
    #  epochs that contain major artefacts.
    #  If the difference between the
    #  highest recorded amplitude and
    #  the lowest recorded amplitude in
    #  an epoch is larger than 150 µV, we
    #  reject that epoch. Brain activity
    #  fluctuations are unlikely to cause
    #  such large amplitude fluctuations.
    reject_criteria = dict(eeg=150e-6)

    # If the difference between the
    #  highest recorded amplitude and
    #  the lowest recorded amplitude in
    #  an epoch is smaller than 0.1 µV,
    #  we reject that epoch. Brain ac-
    #  tivity fluctuations typically
    #  give rise to larger amplitude
    #  fluctuations, so it seems there
    #  has been a measurement error.
    flat_criteria = dict(eeg=1e-7)

    # Now that we have specified our epoch
    #  rejection criteria, let us get rid of
    #  all epochs that meet those criteria.
    originalNumberOfEpochs = len(epochs)
    epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)

    # We can print some statistics
    #  about how many epochs were dropped.
    if False:
        epochs.plot_drop_log()
        remainingNumberOfEpochs = len(epochs)
        percentageDropped = (originalNumberOfEpochs - remainingNumberOfEpochs)/(originalNumberOfEpochs/100)
        print("Percentage of epochs that were dropped: {}".format(percentageDropped))

    ### ---------- Step 2.2.14 --------- ###

    # At step 2.2.9, we marked all of
    #  the bad channels for this subject.
    #  Instead of simply dropping those
    #  channels and pretending they were
    #  never part of our electrode montage,
    #  we will try to repair them by looking
    #  at the EEG data that was recorded by
    #  other (good) channels in the same
    #  area. This technique is known as
    #  interpolation. We make use of the
    #  so-called spherical spline method.
    epochs.interpolate_bads()

    ### ---------- Step 2.2.15 --------- ###

    # We can visualise the epochs around,
    #  for example, all events that are
    #  of type 'Add0_StimulusAppears'.
    if False:
        eventsToHighlight_simple = [100, 150, 1, 2, 3, 4, 5, 6, 7, 8, 9, 210,
                                    201, 202, 203, 204, 205, 206, 207, 208, 209]
        eventsToHighlight_complex = mne.pick_events(events, include=eventsToHighlight)
        colourSettings = dict(Add0_StimulusAppears='red', Add0_StimulusDisappears='blue',
                              Required1='green', Required2='green', Required3='green',
                              Required4='green', Required5='green', Required6='green',
                              Required7='green', Required8='green', Required9='green',
                              Pressed0='purple', Pressed1='purple', Pressed2='purple',
                              Pressed3='purple', Pressed4='purple', Pressed5='purple',
                              Pressed6='purple', Pressed7='purple', Pressed8='purple', Pressed9='purple')
        epochs['Add0_StimulusAppears'].plot(events=eventsToHighlight_complex, event_id=event_dictionary,
                                            n_epochs=3, block=True, event_color=colourSettings)

    ### ~~~~~~~~~ Power scores ~~~~~~~~~ ###

    # We will now calculate the power scores
    #  for this participant, one condition at
    #  a time. We will store the scores in an
    #  array called 'powerScoresPerCondition'
    #  and the sampling frequencies in an array
    #  called 'samplingFrequenciesPerCondition'.
    #  Please note that the latter is actually
    #  a bit redundant: we use the same sampling
    #  frequencies across all three conditions.
    powerScoresPerCondition = []
    samplingFrequenciesPerCondition = []

    for condition in range(0, 3):

        epochsForThisCondition = epochs['Add' + str(condition) + '_StimulusAppears']
        powerScoresForThisCondition, samplingFrequenciesForThisCondition = \
            mne.time_frequency.psd_multitaper(epochsForThisCondition, picks=['eeg'])
        powerScoresForThisCondition = np.mean(powerScoresForThisCondition, axis=0)

        # We store the power scores and sampling
        #  frequencies for this condition in
        #  the arrays we initialised earlier.
        powerScoresPerCondition.append(powerScoresForThisCondition)
        samplingFrequenciesPerCondition.append(samplingFrequenciesForThisCondition)

        # We can visualise the topographical
        #  distribution of theta activity for
        #  the current participant-condition tuple.
        if False:
            mne.viz.topomap.plot_psds_topomap(
                psds=powerScoresForThisCondition, freqs=samplingFrequenciesForThisCondition,
                bands=[(thetaRange[0],thetaRange[1],'Theta')], dB=False, normalize=False,
                show=True, ch_type='eeg', pos=epochsForThisCondition[0][0].info)

    # We normalise the power
    #  scores for this participant.
    sumOfAllPowerScores = \
        powerScoresPerCondition[0].sum(axis=-1, keepdims=True) + \
        powerScoresPerCondition[1].sum(axis=-1, keepdims=True) + \
        powerScoresPerCondition[2].sum(axis=-1, keepdims=True)
    powerScoresPerCondition[0] /= sumOfAllPowerScores
    powerScoresPerCondition[1] /= sumOfAllPowerScores
    powerScoresPerCondition[2] /= sumOfAllPowerScores

    # We store the power scores and sampling
    #  frequencies for this participant in
    #  the arrays we initialised earlier for
    #  this specific purpose at step 2.1.
    powerScoresPerSubject.append(powerScoresPerCondition)
    samplingFrequencies.append(samplingFrequenciesPerCondition)

### ******************************** ###
###            ~ Part 3 ~            ###
###     Sample-level computations    ###
### ******************************** ###

### ----------- Step 3.1 ----------- ###

# We now have power spectral density scores
#  for all participant-condition-electrode
#  combinations. We want to combine those
#  scores into a single array, which
#  contains one (average) power score per
#  condition-electrode-frequency combination.
#  We start by creating an array that
#  contains a list of power scores for each
#  condition-electrode-frequency combination.
powerScores_FullSample_notAveragedPerFrequency = [[[[] for i in range(1126)] for j in range(32)] for k in range(3)]
for condition in range(0, 3):
    for participant in range (0, len(powerScoresPerSubject)):
        for electrode in range (0,32):
            for frequency in range(0, 1126):
                powerScores_FullSample_notAveragedPerFrequency[condition][electrode][frequency].\
                    append(powerScoresPerSubject[participant][condition][electrode][frequency])

# We now create an array that contains a
#  single (average) power score for each
#  condition-electrode-frequency combination.
powerScores_FullSample_averagedPerFrequency = [[[] for j in range(32)] for k in range(3)]
for condition in range(0, 3):
    for electrode in range (0,32):
        for frequency in range(0, 1126):
            listOfPowerScores = powerScores_FullSample_notAveragedPerFrequency[condition][electrode][frequency]
            averagePowerScore = sum(listOfPowerScores) / len(listOfPowerScores)
            powerScores_FullSample_averagedPerFrequency[condition][electrode].append(averagePowerScore)

### ----------- Step 3.2 ----------- ###

# Let us now extract a single average theta
#  power score per electrode, per condition,
#  per participant from all of our data.
#  We will store the new scores in an array
#  called 'powerScores_perParticipant_theta'.
powerScores_perParticipant_theta = []

thetaScoreIndices = []
for frequencyIndex in range(0, len(samplingFrequencies[0][0])):
    if thetaRange[0] <= samplingFrequencies[0][0][frequencyIndex] <= thetaRange[1]:
        thetaScoreIndices.append(frequencyIndex)

for participant in powerScoresPerSubject:
    conditions = []
    for condition in participant:
        electrodes = []
        for electrode in condition:
            thetaScores = []
            for scoreIndex in range(0, len(electrode)):
                if scoreIndex in thetaScoreIndices:
                    thetaScores.append(electrode[scoreIndex])
            averageElectrodeScore_theta = sum(thetaScores) / len(thetaScores)
            electrodes.append(averageElectrodeScore_theta)
        conditions.append(electrodes)
    powerScores_perParticipant_theta.append(conditions)

### ******************************** ###
###            ~ Part 4 ~            ###
###     A focus on theta activity    ###
### ******************************** ###

### ----------- Step 4.1 ----------- ###

# We can now use the calculated power
#  scores to generate three theta topoplots:
#  one for each condition. We save the three
#  theta topoplots as PDF files in a folder
#  called '/Output/Theta topoplots'.
for condition in range(0, 3):
    powerScoresForThisCondition = np.array(powerScores_FullSample_averagedPerFrequency[condition])
    samplingFrequenciesForThisCondition = np.array(samplingFrequencies[0][0])
    fig = mne.viz.topomap.plot_psds_topomap(
        psds=powerScoresForThisCondition, freqs=samplingFrequenciesForThisCondition,
        bands=[(thetaRange[0],thetaRange[1],'Theta')], dB=False, normalize=False,
        show=True, ch_type='eeg', pos=epochsForThisCondition[0][0].info)
    fig.savefig(fname="../../Output/Theta topoplots/Add-" + str(condition) + ".pdf", format='pdf')

### ----------- Step 4.2 ----------- ###

# We extract the theta power scores for
#  further processing in SPSS. We store the
#  extracted theta power scores in a folder
#  called '/Output/Theta power scores'. We
#  do so twice, in two different formats:
#  the 'wide' format and the 'long' format.
#  We start by creating a 'wide' table. We
#  save it in the above-mentioned folder as
#  an Excel-file called 'Wide format.xlsx'.
pythonTable_wide = []
for participantNumber in range (0, len(powerScores_perParticipant_theta)):
    realParticipantNumber = participantNumber + 1
    for excludedParticipant in excludedParticipants:
        if realParticipantNumber >= excludedParticipant:
            realParticipantNumber += 1
    newRow = [realParticipantNumber]
    for conditionNumber in range (0, len(powerScores_perParticipant_theta[participantNumber])):
        for electrodeNumber in range (0, len(powerScores_perParticipant_theta[participantNumber][conditionNumber])):
            thetaPowerAverage = powerScores_perParticipant_theta[participantNumber][conditionNumber][electrodeNumber]
            newRow.append(thetaPowerAverage)
    pythonTable_wide.append(newRow)
columns = ['Participant']
numberOfConditions = len(powerScores_perParticipant_theta[0])
numberOfElectrodes = len(powerScores_perParticipant_theta[0][0])
for conditionNumber in range(0, numberOfConditions):
    for electrodeNumber in range(0, numberOfElectrodes):
        columns.append('Add' + str(conditionNumber) + '_Electrode' + str(electrodeNumber + 1))
pandasTable_wide = pd.DataFrame(pythonTable_wide, columns=columns)
pandasTable_wide.to_excel("../../Output/Theta power scores/Wide format.xlsx")

# We continue by  creating a 'long' table.
#  We save it (in the same folder) as an
#  Excel-file called 'Long format.xlsx'.
pythonTable_long = []
for participantNumber in range (0, len(powerScores_perParticipant_theta)):
    realParticipantNumber = participantNumber + 1
    for excludedParticipant in excludedParticipants:
        if realParticipantNumber >= excludedParticipant:
            realParticipantNumber += 1
    for conditionNumber in range (0, len(powerScores_perParticipant_theta[participantNumber])):
        for electrodeNumber in range (0, len(powerScores_perParticipant_theta[participantNumber][conditionNumber])):
            thetaPowerAverage = powerScores_perParticipant_theta[participantNumber][conditionNumber][electrodeNumber]
            newRow = [realParticipantNumber, conditionNumber, electrodeNumber+1, thetaPowerAverage]
            pythonTable_long.append(newRow)
pandasTable_long = pd.DataFrame(pythonTable_long, columns=['Participant', 'Condition', 'Electrode', 'Theta power score'])
pandasTable_long.to_excel("../../Output/Theta power scores/Long format.xlsx")

### ----------- Step 4.3 ----------- ###

# The extracted theta power scores were
#  subjected to a two-way repeated measures
#  analysis of variance (ANOVA) in SPSS with
#  'Condition' and 'Electrode' as the two
#  within-subjects factors. For details,
#  please see section 2 of my thesis. The
#  interaction between the factors 'Condition'
#  and 'Electrode' was insignificant, which
#  implies that the three theta distributions
#  that were plotted at step 4.1 above do not
#  significantly differ from each other. The
#  opposite could not immediately have been
#  concluded if a significant interaction had
#  been found between 'Condition' and 'Electrode':
#  such a hypothetical interaction could have been
#  caused by overall amplitude differences across
#  the conditions, rather than by topographical
#  differences. To exclude this possibility, the
#  data would have to be analysed again after
#  being rescaled. See, for instance, Jing et al.
#  (2006) (DOI: 10.1016/j.jneumeth.2005.08.002).
#  Their rescaling method is implemented below.
#  I did not have to apply it myself, but I hope
#  that others will be able to benefit from this
#  implementation in the future. In order to run
#  the code, please set 'rescaleData' to 'True'.
rescaleData = False

if rescaleData:
    # The rescaling method of Jing et al. (2006)
    #  allows us to compare only two conditions
    #  with each other at a time, so we will look
    #  at each pair of conditions A, B (with A≠B)
    #  one by one here. There are three such pairs.
    conditionPairs = [[0, 1], [0, 2], [1, 2]]

    for conditionPair in conditionPairs:
        # Which conditions are we going
        #  to compare with each other in
        #  this iteration of the loop?
        conditionA = conditionPair[0]
        conditionB = conditionPair[1]
        skippedCondition = ({0, 1, 2} - set(conditionPair)).pop()

        ### ---------- Step 4.3.1 ---------- ###

        # We will now apply the rescaling method.
        #  We will store the rescaled power score
        #  averages in a new array called 'power-
        #  Scores_perParticipant_theta_rescaled'.
        powerScores_perParticipant_theta_rescaled = []

        for participant in powerScores_perParticipant_theta:

            # Let's load the participant's current
            #  power scores for both conditions.
            powerScoresConditionA = participant[conditionA]
            powerScoresConditionB = participant[conditionB]

            # First, we calculate the scaling factors
            #  for both conditions (pA and pB).
            numerator_part1 = 0
            for electrodeNumber in range(0, 32):
                numerator_part1 += powerScoresConditionA[electrodeNumber]*powerScoresConditionB[electrodeNumber]
            numerator_part2 = sum(powerScoresConditionA) * sum(powerScoresConditionB)
            numerator = 32 * numerator_part1 - numerator_part2
            denominatorA_part1 = 0
            for electrodeNumber in range(0, 32):
                denominatorA_part1 += powerScoresConditionA[electrodeNumber]**2
            denominatorA_part2 = sum(powerScoresConditionA)**2
            denominatorA = 32 * denominatorA_part1 - denominatorA_part2
            denominatorB_part1 = 0
            for electrodeNumber in range(0, 32):
                denominatorB_part1 += powerScoresConditionB[electrodeNumber]**2
            denominatorB_part2 = sum(powerScoresConditionB)**2
            denominatorB = 32 * denominatorB_part1 - denominatorB_part2
            pA = numerator / denominatorA
            pB = numerator / denominatorB

            # Next, we calculate the offsets
            #  for both conditions (cA and cB).
            cA = -1 * pA * (sum(powerScoresConditionA)/len(powerScoresConditionA)) + \
                 (sum(powerScoresConditionB)/len(powerScoresConditionB))
            cB = -1 * pB * (sum(powerScoresConditionB)/len(powerScoresConditionB)) + \
                 (sum(powerScoresConditionA)/len(powerScoresConditionA))

            # Finally, we calculate the rescaled
            #  power scores for both conditions.
            powerScoresConditionA_rescaled = []
            for electrodeNumber in range(0, 32):
                oldScore = powerScoresConditionA[electrodeNumber]
                newScore = oldScore * pA + cA
                powerScoresConditionA_rescaled.append(newScore)
            powerScoresConditionB_rescaled = []
            for electrodeNumber in range(0, 32):
                oldScore = powerScoresConditionB[electrodeNumber]
                newScore = oldScore * pB + cB
                powerScoresConditionB_rescaled.append(newScore)

            # We store the participant's rescaled
            #  power scores in the array 'powerScores_
            #  perParticipant_theta_rescaled' that we
            #  initialised earlier for this purpose.
            newScores = [[],[],[]]
            newScores[conditionA] = powerScoresConditionA_rescaled
            newScores[conditionB] = powerScoresConditionB_rescaled
            newScores[skippedCondition] = [-1]*32
            powerScores_perParticipant_theta_rescaled.append(newScores)

        ### ---------- Step 4.3.2 ---------- ###

        # Now that we have rescaled all theta power
        #  scores for the pair of conditions that we
        #  are currently looking at, we can extract
        #  the rescaled scores for further processing
        #  in SPSS. We store the extracted data in a
        #  folder called '/Output/Rescaled theta power
        #  scores/Add-[condition A] and Add-[condition
        #  B]'. We do so twice, in two formats: the
        #  'wide' format and the 'long' format. We
        #  start by creating a 'wide' table, which
        #  we save in the above-mentioned folder as
        #  an Excel-file called 'Wide format.xlsx'.
        pythonTable_wide = []
        for participantNumber in range (0, len(powerScores_perParticipant_theta_rescaled)):
            realParticipantNumber = participantNumber + 1
            for excludedParticipant in excludedParticipants:
                if realParticipantNumber >= excludedParticipant:
                    realParticipantNumber += 1
            newRow = [realParticipantNumber]
            for conditionNumber in range (0, len(powerScores_perParticipant_theta_rescaled[participantNumber])):
                for electrodeNumber in range (0, len(powerScores_perParticipant_theta_rescaled[participantNumber][conditionNumber])):
                    thetaPowerAverage = powerScores_perParticipant_theta_rescaled[participantNumber][conditionNumber][electrodeNumber]
                    newRow.append(thetaPowerAverage)
            pythonTable_wide.append(newRow)
        columns = ['Participant']
        numberOfConditions = len(powerScores_perParticipant_theta_rescaled[0])
        numberOfElectrodes = len(powerScores_perParticipant_theta_rescaled[0][0])
        for conditionNumber in range(0, numberOfConditions):
            for electrodeNumber in range(0, numberOfElectrodes):
                columns.append('Add' + str(conditionNumber) + '_Electrode' + str(electrodeNumber + 1))
        Path("../../Output/Rescaled theta power scores/Add-" +
             str(conditionA) + " and Add-" + str(conditionB)).mkdir(parents=True, exist_ok=True)
        pandasTable_wide = pd.DataFrame(pythonTable_wide, columns=columns)
        pandasTable_wide.to_excel("../../Output/Rescaled theta power scores/Add-" +
                                  str(conditionA) + " and Add-" + str(conditionB) + "/Wide format.xlsx")

        # We continue by  creating a 'long' table.
        #  We save it (in the same folder) as an
        #  Excel-file called 'Long format.xlsx'.
        pythonTable_long = []
        for participantNumber in range (0, len(powerScores_perParticipant_theta_rescaled)):
            realParticipantNumber = participantNumber + 1
            for excludedParticipant in excludedParticipants:
                if realParticipantNumber >= excludedParticipant:
                    realParticipantNumber += 1
            for conditionNumber in range (0, len(powerScores_perParticipant_theta_rescaled[participantNumber])):
                for electrodeNumber in range (0, len(powerScores_perParticipant_theta_rescaled[participantNumber][conditionNumber])):
                    thetaPowerAverage = powerScores_perParticipant_theta_rescaled[participantNumber][conditionNumber][electrodeNumber]
                    newRow = [realParticipantNumber, conditionNumber, electrodeNumber+1, thetaPowerAverage]
                    pythonTable_long.append(newRow)
        columnNames = ['Participant', 'Condition', 'Electrode', 'Rescaled theta power score']
        pandasTable_long = pd.DataFrame(pythonTable_long, columns=columnNames)
        pandasTable_long.to_excel("../../Output/Rescaled theta power scores/Add-" +
                                  str(conditionA) + " and Add-" + str(conditionB) + "/Long format.xlsx")