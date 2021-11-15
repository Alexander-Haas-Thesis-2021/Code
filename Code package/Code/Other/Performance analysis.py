# --------------------------------------- #
#           Performance Analysis          #
# --------------------------------------- #

# --------------------------------------- #
#                 Overview                #
# --------------------------------------- #
#  This code can be used to analyse how   #
#  well the participants performed on the #
#  implementation of the Add-n task that  #
#  was developed by dr. Rob van der Lubbe #
#  at the University of Twente in 2019.   #
# --------------------------------------- #
#     a.n.j.p.m.haas@gmail.com (2021)     #
# --------------------------------------- #

# ================ CODE ================= #

### ------------- Step A -------------- ###

# We import the Python modules we need.
import mne
import sys
from os import path
from pathlib import Path

### ------------- Step B -------------- ###

# We import some useful information
#  that we stored in other files to
#  avoid cluttering up this code file.
mainDirectory = '../..'
subDirectory1 = '/Data/Batch 1 (P01-P20) [2019]/'
subDirectory2 = '/Data/Batch 2 (P21-P37) [2021]/'

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

### ------------- Step C -------------- ###

# We analyse the participants' performance
#  on the Add-n task. We look at the
#  participants one by one in a loop.
for file in files:

    # We check whether the specified file actually exists.
    if not path.exists(file):
        print("\nThe following file could not be found and therefore will not be analysed: \'{}\'.".format(file))
        continue

    # We load the raw data. Since we make use of BrainVision
    #  data, we should apply a non-standard read function. For this
    #  to work, the .vhdr file should be in the same directory as its
    #  corresponding .eeg file and its corresponding .vmrk file.
    raw = mne.io.read_raw_brainvision(file)

    # During the experiment, we kept track of everything that happened. Relevant happenings,
    #  such as button presses, are called 'events'. Let us extract all events from the data.
    events, notNeeded = mne.events_from_annotations(raw)

    # We can explore the structure of the array called 'events'.
    if False:
        print("*** Events for participant {} ***".format(file[-17:-15]))
        for event in events:
            print(event)
        print("*********************************\n")

    # For each condition, we want to store how many of the digits that were entered
    #  by the participant were correct. We initialise three variables for this purpose.
    add0_Correct = 0
    add1_Correct = 0
    add2_Correct = 0

    # We look at all events that were recorded for the participant. We compare the correct digits
    #  (which are included in the data as events) with the digits entered by the participant. We
    #  update the appropriate variable ('add0_Correct', 'add1_Correct' or 'add2_Correct') if a digit
    #  entered by the participant was correct. (The first two events are '99999' and '10001' for each
    #  participant. Since these events are irrelevant, we skip them here by starting our loop at i=2.)
    #  This code was designed to be able to deal with the idiosyncratic structure of the array called
    #  'events'. Familiarity with that structure shall be assumed here, and annotation shall be minimal.
    for i in range(2, len(events)):

        # We are looking for events that mark the start of an Add-0 trial (code 100), an Add-1 trial
        #  (code 101) or an Add-2 trial (code 102). Upon encountering such an event, we take a closer
        #  look at the events that follow. Those events tell us which digits would have been correct for
        #  the trial we have stumbled upon, and which digits were entered by the participant for that
        #  trial. We want to store this information, so we can subsequently assess how many of the digits
        #  that were entered by the participant were correct. We initialise two arrays for this purpose.
        correctDigits = []
        enteredDigits = []

        if file[-17:-15] == '01':
            # There is an issue with the first file. Whenever '0' was part of the correct solution, its
            #  corresponding stimulus code ('10') was not included as part of the correct solution in
            #  the event data. In the Add-0 condition, for example, if the correct solution was '0 8 4 7'
            #  (or '8 0 4 7', or '8 4 0 7', or '8 4 7 0') the correct solution was '8 4 7' (instead of '10
            #  8 4 7', or '8 10 4 7', or '8 4 10 7', or '8 4 7 10', respectively) according to the event data.
            #  Fortunately, this bug seems to have been corrected after the first session in the lab in 2019.
            #  (I decided to give the first participant the benefit of the doubt whenever a '0' seemed to have
            #  been part of the correct solution. If the correct solution was '7 5 6' according to the event data,
            #  for example, I would not just award a full point for the answer '210 207 205 206', but also for
            #  the answers '207 210 205 206', '207 205 210 206' and '207 205 206 210'.)
            add0_Correct = 159
            add1_Correct = 137
            add2_Correct = 149
            break

        # Did we just stumble upon an event that marks the start of an Add-0 trial?
        if events[i][2] == 100:
            if file[-17:-15] == '10' and i == 143:
                # The file for participant 10 contains some strange deviations around event 143 (code 100).
                continue
            # What were the correct digits for this trial?
            jump = 0
            while True:
                jump = jump + 1
                if 1 <= events[i+jump][2] <= 10:
                    correctDigits.append(events[i + jump][2])
                    continue
                elif len(correctDigits) != 4 or events[i + jump][2] == 211:
                    continue
                else:
                    break
            # What digits were entered by the participant?
            while True:
                if 201 <= events[i+jump][2] <= 210:
                    enteredDigits.append(events[i + jump][2])
                    if jump+1+i >= len(events):
                        break
                    jump = jump + 1
                    continue
                elif events[i+jump][2] == 211:
                    if jump+1+i >= len(events):
                        break
                    jump = jump + 1
                    continue
                else:
                    break
            # How many of the digits that the participant entered were correct?
            numberOfCorrectAnswers = 0
            if correctDigits[0] == (enteredDigits[0] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[1] == (enteredDigits[1] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[2] == (enteredDigits[2] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[3] == (enteredDigits[3] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            add0_Correct = add0_Correct + numberOfCorrectAnswers

        # Did we just stumble upon an event that marks the start of an Add-1 trial?
        elif events[i][2] == 101:
            # What were the correct digits for this trial?
            jump = 0
            while True:
                jump = jump + 1
                if 1 <= events[i+jump][2] <= 10:
                    correctDigits.append(events[i + jump][2])
                    continue
                elif len(correctDigits) != 4 or events[i + jump][2] == 211:
                    continue
                else:
                    break
            # What digits were entered by the participant?
            while True:
                if 201 <= events[i+jump][2] <= 210:
                    enteredDigits.append(events[i + jump][2])
                    if jump+1+i >= len(events):
                        break
                    jump = jump + 1
                    continue
                elif events[i+jump][2] == 211:
                    if jump+1+i >= len(events):
                        break
                    jump = jump + 1
                    continue
                else:
                    break
            # How many of the digits that the participant entered were correct?
            numberOfCorrectAnswers = 0
            if correctDigits[0] == (enteredDigits[0] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[1] == (enteredDigits[1] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[2] == (enteredDigits[2] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[3] == (enteredDigits[3] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            add1_Correct = add1_Correct + numberOfCorrectAnswers

        # Did we just stumble upon an event that marks the start of an Add-2 trial?
        elif events[i][2] == 102:
            # What were the correct digits for this trial?
            jump = 0
            while True:
                jump = jump + 1
                if 1 <= events[i+jump][2] <= 10:
                    correctDigits.append(events[i + jump][2])
                    continue
                elif len(correctDigits) != 4 or events[i + jump][2] == 211:
                    continue
                else:
                    break
            # What digits were entered by the participant?
            while True:
                if 201 <= events[i+jump][2] <= 210:
                    enteredDigits.append(events[i + jump][2])
                    if jump+1+i >= len(events):
                        break
                    jump = jump + 1
                    continue
                elif events[i+jump][2] == 211:
                    if jump+1+i >= len(events):
                        break
                    jump = jump + 1
                    continue
                else:
                    break
            # How many of the digits that the participants entered were correct?
            numberOfCorrectAnswers = 0
            if correctDigits[0] == (enteredDigits[0] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[1] == (enteredDigits[1] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[2] == (enteredDigits[2] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            if correctDigits[3] == (enteredDigits[3] - 200):
                numberOfCorrectAnswers = numberOfCorrectAnswers + 1
            add2_Correct = add2_Correct + numberOfCorrectAnswers

        # Apparently, we did not stumble upon an event that marks the start of an
        #  Add-0 trial, an Add-1 trial or an Add-2 trial. Let's keep looking.
        else:
            continue

    # We looked at all events now. How did the participant perform? The participant faced each condition twice.
    #  Each condition consisted of 20 trials. The participant was asked to enter four digits per trial. In
    #  total, therefore, the participant could enter at most 160 digits correctly per condition. To calculate
    #  the participant's relative number of correct digits for a specific condition, we should divide the
    #  participant's absolute number of correct digits for that condition by 1.6 (since 2*20*4 equals 160).

    # Let's print our results to a file called 'Overview.txt'. First, let's add some metadata to that file.
    if file[-17:-15] == '01':
        Path("../../Output/Participant performance").mkdir(parents=True, exist_ok=True)
        with open('../../Output/Participant performance/Overview.txt', 'w') as outputFile:
            original_out = sys.stdout
            sys.stdout = outputFile
            print("[PERFORMANCE OVERVIEW - GENERATED BY 'PERFORMANCE ANALYSIS.PY']")
            sys.stdout = original_out

    # Next, let's add the results of our performance analysis to 'Overview.txt'.
    with open('../../Output/Participant performance/Overview.txt', 'a') as outputFile:
        original_out = sys.stdout
        sys.stdout = outputFile
        denominator = 1.6
        print("\n---------- P{} ----------".format(file[-17:-15]))
        print("How did participant {} perform?".format(file[-17:-15]))
        print("> Condition Add-0: {} of the digits entered by "
              "the participant were correct ({}%)".format(add0_Correct, add0_Correct/denominator))
        print("> Condition Add-1: {} of the digits entered by "
              "the participant were correct ({}%)".format(add1_Correct, add1_Correct/denominator))
        print("> Condition Add-2: {} of the digits entered by "
              "the participant were correct ({}%)".format(add2_Correct, add2_Correct/denominator))
        if file[-17:-15] == "19" or file[-17:-15] == "26":
            print("* Warning: some trials of the experiment were accidentally not recorded for this participant")
        sys.stdout = original_out

# If at least one file was successfully analysed, we will print a 'mission accomplished' message to the console.
for file in files:
    if path.exists(file):
        print("\n------------------------------------------------------------------------------------------------------------------------------")
        print("The code was executed successfully. Please see '.../Output/Participant performance/Overview.txt' for the outcomes.")
        print("------------------------------------------------------------------------------------------------------------------------------")
        exit()