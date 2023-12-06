#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.1.3),
    on Fri Nov 17 16:01:41 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from task_code
# add another output table.
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# info about the experiment session

expInfo = {'participant': '', 'session': '001'}
expInfo['date'] = data.getDateStr()  # add a simple timestamp

result_name = filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expInfo['date'], 'results')
resultTbl = data.ExperimentHandler(name='musicbox_practice', version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\behrenslab-fmrib\\Documents\\Psychopy\\musicbox\\experiment\\3x3_practice_v2.py',
    savePickle=True, saveWideText=True,
    dataFileName=result_name)
    
    
# function to get a gamma-function jitter.
import numpy as np
def jitter(exp_step_no, mean_subpath_length, var_per_step=1.5):
    # first randomly sample from a gamma distribution
    subpath_length = np.random.standard_gamma(mean_subpath_length)
    while (subpath_length < 3) or (subpath_length > 12):
        subpath_length = np.random.standard_gamma(mean_subpath_length)
    # then make an array for each step + reward you expect to take
    step_size_dummy = np.random.randint(1, (exp_step_no+1)*var_per_step, size = exp_step_no + 1)
    # make the last one, the reward, twice as long as the average step size
    ave_step = np.mean(step_size_dummy[0:-1])
    step_size_dummy[-1] = ave_step * 2
    # then multiply the fraction of all step lenghts with the actual subpath length
    stepsizes = np.empty(exp_step_no + 1)
    for i in range(exp_step_no + 1):
        stepsizes[i] = (step_size_dummy[i]/(sum(step_size_dummy))) * subpath_length
    # stepsizes[-1] will be the reward length. If more steps than len(stepsizes[0:-1]), randomly sample.
    return stepsizes


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2023.1.3'
expName = '3x3_fMRI_part1'  # from the Builder filename that created this script
expInfo = {
    'participant': '',
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Volumes/NO NAME/latest_experiment/3x3_fMRI_part1.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1333,1000], fullscr=False, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='dell_compneuro_laptop', color=[0.7412, 0.4431, 0.0588], colorSpace='rgb',
    backgroundImage='', backgroundFit='none',
    blendMode='avg', useFBO=True, 
    units='norm')
win.mouseVisible = True
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "test_keys" ---
# Run 'Begin Experiment' code from key_test_code
from psychopy.core import wait as wait
msg = "waiting for the 4 button presses.."
all_keys_are_pressed = False
treasurehunt_welcome_img = visual.ImageStim(
    win=win,
    name='treasurehunt_welcome_img', 
    image='images/map_w_txt.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
welcome_txt = visual.TextStim(win=win, name='welcome_txt',
    text='',
    font='Open Sans',
    pos=(0, 0.1), height=0.1, wrapWidth=None, ori=0.0, 
    color='black', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
key_resp_test = keyboard.Keyboard()
feedback_testkeys = visual.TextStim(win=win, name='feedback_testkeys',
    text='',
    font='Open Sans',
    pos=(0, -0.4), height=0.08, wrapWidth=None, ori=0.0, 
    color='black', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# --- Initialize components for Routine "scan_trigger" ---
scan_trigger_key = keyboard.Keyboard()
wait_scanner_trigger_txt = visual.TextStim(win=win, name='wait_scanner_trigger_txt',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='black', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# --- Initialize components for Routine "show_rewards" ---
# Run 'Begin Experiment' code from show_rew_code
op_rewB = 0
op_rewA = 0
op_rewC = 0
op_rewD = 0
sand_pirate = visual.ImageStim(
    win=win,
    name='sand_pirate', 
    image='images/sand_3x3grid_pirate.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
reward_A = visual.ImageStim(
    win=win,
    name='reward_A', 
    image='images/coin.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.1, 0.13),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
reward_B = visual.ImageStim(
    win=win,
    name='reward_B', 
    image='images/coin.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.1, 0.13),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
reward_C = visual.ImageStim(
    win=win,
    name='reward_C', 
    image='images/coin.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.1, 0.13),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-4.0)
reward_D = visual.ImageStim(
    win=win,
    name='reward_D', 
    image='images/coin.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.1, 0.13),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-5.0)
warning_txt = visual.TextStim(win=win, name='warning_txt',
    text='You thought this was easy? Now find the gold in reversed order!',
    font='Open Sans',
    pos=(-0.515, 0), height=0.1, wrapWidth=0.38, ori=0.0, 
    color='black', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-6.0);

# --- Initialize components for Routine "task" ---
# Run 'Begin Experiment' code from task_code
from psychopy.core import wait as wait
import random 

sand_box = visual.ImageStim(
    win=win,
    name='sand_box', 
    image='images/sand_3x3grid_box.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
foot = visual.ImageStim(
    win=win,
    name='foot', 
    image='images/footprints_buttons.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.2, 0.26),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
reward = visual.ImageStim(
    win=win,
    name='reward', 
    image='images/coin.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.1, 0.13),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
nav_key_task = keyboard.Keyboard()
break_key = keyboard.Keyboard()
progressbar_background = visual.Rect(
    win=win, name='progressbar_background',
    width=(0.1, 0.41)[0], height=(0.1, 0.41)[1],
    ori=0.0, pos=(-0.5, -0.21), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.0902, -0.4588, -0.8510], fillColor=[0.0902, -0.4588, -0.8510],
    opacity=None, depth=-6.0, interpolate=True)
progress_bar = visual.Rect(
    win=win, name='progress_bar',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=(-0.5, -0.21), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=[0.7098, 0.2941, -0.7490],
    opacity=None, depth=-7.0, interpolate=True)
reward_progress = visual.ImageStim(
    win=win,
    name='reward_progress', 
    image='images/coin.png', mask=None, anchor='center',
    ori=0.0, pos=(-0.65, -0.21), size=(0.1, 0.13),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-8.0)
plus_coin_txt = visual.TextStim(win=win, name='plus_coin_txt',
    text='+1 coin',
    font='Open Sans',
    pos=(-0.65, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='black', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-9.0);
reward_A_feedback = visual.TextStim(win=win, name='reward_A_feedback',
    text='"Arrr! Ye\'ve struck gold 1 of 4 with yer keen eye!"',
    font='Open Sans',
    pos=(0.518, 0), height=0.07, wrapWidth=0.38, ori=0.0, 
    color='white', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-10.0);
TR_key = keyboard.Keyboard()

# --- Initialize components for Routine "feedback_screen" ---
# Run 'Begin Experiment' code from code
feedback_msg = ''
background_feedback = visual.ImageStim(
    win=win,
    name='background_feedback', 
    image='images/pirate_withou_text.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
feedback_text = visual.TextStim(win=win, name='feedback_text',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=0.85, ori=0.0, 
    color=[-0.1765, -0.1765, -0.1765], colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "test_keys" ---
continueRoutine = True
# update component parameters for each repeat
key_resp_test.keys = []
key_resp_test.rt = []
_key_resp_test_allKeys = []
# keep track of which components have finished
test_keysComponents = [treasurehunt_welcome_img, welcome_txt, key_resp_test, feedback_testkeys]
for thisComponent in test_keysComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "test_keys" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # Run 'Each Frame' code from key_test_code
    keys_pressed = key_resp_test.keys
    if all_keys_are_pressed == True:
        wait(3)
        continueRoutine = False
        
        
    if len(keys_pressed) > 0:
        all_keys_are_pressed = set(['1', '2', '3', '4']).issubset(set(keys_pressed))
        if all_keys_are_pressed == True:
            msg = "all buttons pressed!"
    
    
    # *treasurehunt_welcome_img* updates
    
    # if treasurehunt_welcome_img is starting this frame...
    if treasurehunt_welcome_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        treasurehunt_welcome_img.frameNStart = frameN  # exact frame index
        treasurehunt_welcome_img.tStart = t  # local t and not account for scr refresh
        treasurehunt_welcome_img.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(treasurehunt_welcome_img, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'treasurehunt_welcome_img.started')
        # update status
        treasurehunt_welcome_img.status = STARTED
        treasurehunt_welcome_img.setAutoDraw(True)
    
    # if treasurehunt_welcome_img is active this frame...
    if treasurehunt_welcome_img.status == STARTED:
        # update params
        pass
    
    # *welcome_txt* updates
    
    # if welcome_txt is starting this frame...
    if welcome_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome_txt.frameNStart = frameN  # exact frame index
        welcome_txt.tStart = t  # local t and not account for scr refresh
        welcome_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_txt, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'welcome_txt.started')
        # update status
        welcome_txt.status = STARTED
        welcome_txt.setAutoDraw(True)
    
    # if welcome_txt is active this frame...
    if welcome_txt.status == STARTED:
        # update params
        welcome_txt.setText('Welcome to the treasure hunt experiment!\n\nPlease press every key of the button box once.', log=False)
    
    # *key_resp_test* updates
    waitOnFlip = False
    
    # if key_resp_test is starting this frame...
    if key_resp_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_test.frameNStart = frameN  # exact frame index
        key_resp_test.tStart = t  # local t and not account for scr refresh
        key_resp_test.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_test, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'key_resp_test.started')
        # update status
        key_resp_test.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_test.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_test.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_test.getKeys(keyList=['1','2','3','4'], waitRelease=False)
        _key_resp_test_allKeys.extend(theseKeys)
        if len(_key_resp_test_allKeys):
            key_resp_test.keys = [key.name for key in _key_resp_test_allKeys]  # storing all keys
            key_resp_test.rt = [key.rt for key in _key_resp_test_allKeys]
            key_resp_test.duration = [key.duration for key in _key_resp_test_allKeys]
    
    # *feedback_testkeys* updates
    
    # if feedback_testkeys is starting this frame...
    if feedback_testkeys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        feedback_testkeys.frameNStart = frameN  # exact frame index
        feedback_testkeys.tStart = t  # local t and not account for scr refresh
        feedback_testkeys.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(feedback_testkeys, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'feedback_testkeys.started')
        # update status
        feedback_testkeys.status = STARTED
        feedback_testkeys.setAutoDraw(True)
    
    # if feedback_testkeys is active this frame...
    if feedback_testkeys.status == STARTED:
        # update params
        feedback_testkeys.setText(msg, log=False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in test_keysComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "test_keys" ---
for thisComponent in test_keysComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if key_resp_test.keys in ['', [], None]:  # No response was made
    key_resp_test.keys = None
thisExp.addData('key_resp_test.keys',key_resp_test.keys)
if key_resp_test.keys != None:  # we had a response
    thisExp.addData('key_resp_test.rt', key_resp_test.rt)
    thisExp.addData('key_resp_test.duration', key_resp_test.duration)
thisExp.nextEntry()
# the Routine "test_keys" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
wait_5_times = data.TrialHandler(nReps=5.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='wait_5_times')
thisExp.addLoop(wait_5_times)  # add the loop to the experiment
thisWait_5_time = wait_5_times.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisWait_5_time.rgb)
if thisWait_5_time != None:
    for paramName in thisWait_5_time:
        exec('{} = thisWait_5_time[paramName]'.format(paramName))

for thisWait_5_time in wait_5_times:
    currentLoop = wait_5_times
    # abbreviate parameter names if possible (e.g. rgb = thisWait_5_time.rgb)
    if thisWait_5_time != None:
        for paramName in thisWait_5_time:
            exec('{} = thisWait_5_time[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "scan_trigger" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from scanner_trigger_code
    time_scanner_prompt_start = globalClock.getTime()
    scan_trigger_key.keys = []
    scan_trigger_key.rt = []
    _scan_trigger_key_allKeys = []
    # keep track of which components have finished
    scan_triggerComponents = [scan_trigger_key, wait_scanner_trigger_txt]
    for thisComponent in scan_triggerComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "scan_trigger" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from scanner_trigger_code
        time_scanner_prompt_end = globalClock.getTime()
        
        thisExp.addData('time_scanner_prompt_start', time_scanner_prompt_start)
        thisExp.addData('time_scanner_prompt_end', time_scanner_prompt_end)
        
        
        if len(scan_trigger_key.getKeys(keyList = '5')) > 0:
            Scanner_trigger = globalClock.getTime()
            thisExp.nextEntry()
            thisExp.addData(f"TR_received_no{len(scan_trigger_key.getKeys(keyList = '5'))}", Scanner_trigger)
        
        # *scan_trigger_key* updates
        waitOnFlip = False
        
        # if scan_trigger_key is starting this frame...
        if scan_trigger_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            scan_trigger_key.frameNStart = frameN  # exact frame index
            scan_trigger_key.tStart = t  # local t and not account for scr refresh
            scan_trigger_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(scan_trigger_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'scan_trigger_key.started')
            # update status
            scan_trigger_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(scan_trigger_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(scan_trigger_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if scan_trigger_key.status == STARTED and not waitOnFlip:
            theseKeys = scan_trigger_key.getKeys(keyList=['5'], waitRelease=False)
            _scan_trigger_key_allKeys.extend(theseKeys)
            if len(_scan_trigger_key_allKeys):
                scan_trigger_key.keys = [key.name for key in _scan_trigger_key_allKeys]  # storing all keys
                scan_trigger_key.rt = [key.rt for key in _scan_trigger_key_allKeys]
                scan_trigger_key.duration = [key.duration for key in _scan_trigger_key_allKeys]
                # a response ends the routine
                continueRoutine = False
        
        # *wait_scanner_trigger_txt* updates
        
        # if wait_scanner_trigger_txt is starting this frame...
        if wait_scanner_trigger_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            wait_scanner_trigger_txt.frameNStart = frameN  # exact frame index
            wait_scanner_trigger_txt.tStart = t  # local t and not account for scr refresh
            wait_scanner_trigger_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wait_scanner_trigger_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wait_scanner_trigger_txt.started')
            # update status
            wait_scanner_trigger_txt.status = STARTED
            wait_scanner_trigger_txt.setAutoDraw(True)
        
        # if wait_scanner_trigger_txt is active this frame...
        if wait_scanner_trigger_txt.status == STARTED:
            # update params
            wait_scanner_trigger_txt.setText('Waiting for scanner ...', log=False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in scan_triggerComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "scan_trigger" ---
    for thisComponent in scan_triggerComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if scan_trigger_key.keys in ['', [], None]:  # No response was made
        scan_trigger_key.keys = None
    wait_5_times.addData('scan_trigger_key.keys',scan_trigger_key.keys)
    if scan_trigger_key.keys != None:  # we had a response
        wait_5_times.addData('scan_trigger_key.rt', scan_trigger_key.rt)
        wait_5_times.addData('scan_trigger_key.duration', scan_trigger_key.duration)
    # the Routine "scan_trigger" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 5.0 repeats of 'wait_5_times'


# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('task_cond_pt2_3x3.xlsx'),
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "show_rewards" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from show_rew_code
    time_start = globalClock.getTime()
    
    op_rewB = 0
    op_rewA = 0
    op_rewC = 0
    op_rewD = 0
    
    
    if type == 'backw':
        show_backwards_r = 1
    elif type == 'forw':
        show_backwards_r = 0
    reward_A.setPos((rew_x_A, rew_y_A))
    reward_B.setPos((rew_x_B, rew_y_B))
    reward_C.setPos((rew_x_C, rew_y_C))
    reward_D.setPos((rew_x_D, rew_y_D))
    warning_txt.setOpacity(show_backwards_r)
    # keep track of which components have finished
    show_rewardsComponents = [sand_pirate, reward_A, reward_B, reward_C, reward_D, warning_txt]
    for thisComponent in show_rewardsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "show_rewards" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 12.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from show_rew_code
        t_curr = globalClock.getTime()
        tim = t_curr - time_start
        if tim < 1.5:
            op_rewA = 1
        if (tim > 1.5) and (tim < 3):
            op_rewB = 1
            op_rewA = 0
        if (tim > 3) and (tim < 4.5):
            op_rewB = 0
            op_rewC = 1
        if (tim > 4.5) and (tim < 6):
            op_rewC = 0
            op_rewD = 1
        if (tim > 6) and (tim < 7):
            op_rewD = 0
            op_rewA = 1
        if (tim > 7) and (tim < 8):
            op_rewB = 1
            op_rewA = 0
        if (tim > 8) and (tim < 9):
            op_rewB = 0
            op_rewC = 1
        if tim > 9:
            op_rewC = 0
            op_rewD = 1
        
        # *sand_pirate* updates
        
        # if sand_pirate is starting this frame...
        if sand_pirate.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sand_pirate.frameNStart = frameN  # exact frame index
            sand_pirate.tStart = t  # local t and not account for scr refresh
            sand_pirate.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sand_pirate, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sand_pirate.started')
            # update status
            sand_pirate.status = STARTED
            sand_pirate.setAutoDraw(True)
        
        # if sand_pirate is active this frame...
        if sand_pirate.status == STARTED:
            # update params
            pass
        
        # if sand_pirate is stopping this frame...
        if sand_pirate.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sand_pirate.tStartRefresh + 12-frameTolerance:
                # keep track of stop time/frame for later
                sand_pirate.tStop = t  # not accounting for scr refresh
                sand_pirate.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sand_pirate.stopped')
                # update status
                sand_pirate.status = FINISHED
                sand_pirate.setAutoDraw(False)
        
        # *reward_A* updates
        
        # if reward_A is starting this frame...
        if reward_A.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_A.frameNStart = frameN  # exact frame index
            reward_A.tStart = t  # local t and not account for scr refresh
            reward_A.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_A, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'reward_A.started')
            # update status
            reward_A.status = STARTED
            reward_A.setAutoDraw(True)
        
        # if reward_A is active this frame...
        if reward_A.status == STARTED:
            # update params
            reward_A.setOpacity(op_rewA, log=False)
        
        # if reward_A is stopping this frame...
        if reward_A.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_A.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_A.tStop = t  # not accounting for scr refresh
                reward_A.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_A.stopped')
                # update status
                reward_A.status = FINISHED
                reward_A.setAutoDraw(False)
        
        # *reward_B* updates
        
        # if reward_B is starting this frame...
        if reward_B.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_B.frameNStart = frameN  # exact frame index
            reward_B.tStart = t  # local t and not account for scr refresh
            reward_B.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_B, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'reward_B.started')
            # update status
            reward_B.status = STARTED
            reward_B.setAutoDraw(True)
        
        # if reward_B is active this frame...
        if reward_B.status == STARTED:
            # update params
            reward_B.setOpacity(op_rewB, log=False)
        
        # if reward_B is stopping this frame...
        if reward_B.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_B.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_B.tStop = t  # not accounting for scr refresh
                reward_B.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_B.stopped')
                # update status
                reward_B.status = FINISHED
                reward_B.setAutoDraw(False)
        
        # *reward_C* updates
        
        # if reward_C is starting this frame...
        if reward_C.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_C.frameNStart = frameN  # exact frame index
            reward_C.tStart = t  # local t and not account for scr refresh
            reward_C.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_C, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'reward_C.started')
            # update status
            reward_C.status = STARTED
            reward_C.setAutoDraw(True)
        
        # if reward_C is active this frame...
        if reward_C.status == STARTED:
            # update params
            reward_C.setOpacity(op_rewC, log=False)
        
        # if reward_C is stopping this frame...
        if reward_C.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_C.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_C.tStop = t  # not accounting for scr refresh
                reward_C.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_C.stopped')
                # update status
                reward_C.status = FINISHED
                reward_C.setAutoDraw(False)
        
        # *reward_D* updates
        
        # if reward_D is starting this frame...
        if reward_D.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_D.frameNStart = frameN  # exact frame index
            reward_D.tStart = t  # local t and not account for scr refresh
            reward_D.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_D, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'reward_D.started')
            # update status
            reward_D.status = STARTED
            reward_D.setAutoDraw(True)
        
        # if reward_D is active this frame...
        if reward_D.status == STARTED:
            # update params
            reward_D.setOpacity(op_rewD, log=False)
        
        # if reward_D is stopping this frame...
        if reward_D.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_D.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_D.tStop = t  # not accounting for scr refresh
                reward_D.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_D.stopped')
                # update status
                reward_D.status = FINISHED
                reward_D.setAutoDraw(False)
        
        # *warning_txt* updates
        
        # if warning_txt is starting this frame...
        if warning_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            warning_txt.frameNStart = frameN  # exact frame index
            warning_txt.tStart = t  # local t and not account for scr refresh
            warning_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(warning_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'warning_txt.started')
            # update status
            warning_txt.status = STARTED
            warning_txt.setAutoDraw(True)
        
        # if warning_txt is active this frame...
        if warning_txt.status == STARTED:
            # update params
            pass
        
        # if warning_txt is stopping this frame...
        if warning_txt.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > warning_txt.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                warning_txt.tStop = t  # not accounting for scr refresh
                warning_txt.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'warning_txt.stopped')
                # update status
                warning_txt.status = FINISHED
                warning_txt.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in show_rewardsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "show_rewards" ---
    for thisComponent in show_rewardsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-12.000000)
    
    # set up handler to look after randomisation of conditions etc
    rep_runs = data.TrialHandler(nReps=reps_per_run, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='rep_runs')
    thisExp.addLoop(rep_runs)  # add the loop to the experiment
    thisRep_run = rep_runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRep_run.rgb)
    if thisRep_run != None:
        for paramName in thisRep_run:
            exec('{} = thisRep_run[paramName]'.format(paramName))
    
    for thisRep_run in rep_runs:
        currentLoop = rep_runs
        # abbreviate parameter names if possible (e.g. rgb = thisRep_run.rgb)
        if thisRep_run != None:
            for paramName in thisRep_run:
                exec('{} = thisRep_run[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "task" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from task_code
        rew_visible = 0
        A_visible = 0
        index_rew = [0,0]
        no_keys_pressed = [0]
        
        steps_taken_this_subpath = 0
        
        foot_disappear = 1
        progress_disappear = 1
        progress2_disappear = 1
        
        
        # flush out all potentially recorded button press events
        nav_key_task.clearEvents()
        loop_no = 0
        
        if type == 'forw':
            rew_x = [rew_x_A, rew_x_B, rew_x_C, rew_x_D]
            rew_y = [rew_y_A, rew_y_B, rew_y_C, rew_y_D]
            curr_x = rew_x_D
            curr_y = rew_y_D
            all_exp_step_nos = [exp_step_DA, exp_step_AB, exp_step_BC, exp_step_CD]
            curr_exp_step_no = all_exp_step_nos[index_rew[-1]]
        
        elif type == 'backw':
            rew_x = [rew_x_D, rew_x_C, rew_x_B, rew_x_A]
            rew_y = [rew_y_D, rew_y_C, rew_y_B, rew_y_A]
            curr_x = rew_x_A
            curr_y = rew_y_A
            all_exp_step_nos = [exp_step_DA, exp_step_CD, exp_step_BC, exp_step_AB]
            curr_exp_step_no = all_exp_step_nos[index_rew[-1]]
            
        
        secs_per_step = jitter(curr_exp_step_no, 5.75, var_per_step=1.5)
        
        print(f'these are the new reward locations: xABCD {rew_x}, yABCD {rew_y}')
        
        
        states = ['A', 'B', 'C', 'D']
        field_nos = ['','','','']
        msg = ''
        
        thisExp.nextEntry()
        thisExp.addData('next_task', globalClock.getTime())
        
        resultTbl.nextEntry()
        resultTbl.addData('next_task', globalClock.getTime())
        resultTbl.addData('round_no', Round)
        resultTbl.addData('task_config', Config)
        
        resultTbl.addLoop(rep_runs)  # add the loop to the experiment
        
        # i have 2000ms. every 10 ms, I want to move this thing.
        # this means that I will move the thing 200 times.
        
        interval = 0.05
        move_timer = []
        move_timer.append(globalClock.getTime())
        move_counter = 0
        time_at_press = globalClock.getTime()
        key1_press_trigger = 0
        key2_press_trigger = 0
        key3_press_trigger = 0
        key4_press_trigger = 0
        
        progress_bar_on = 0
        progress_timer = []
        progress_timer.append(globalClock.getTime())
        reward_waiting = random.uniform(1, 3)
        
        nav_key_task.keys = []
        nav_key_task.rt = []
        _nav_key_task_allKeys = []
        break_key.keys = []
        break_key.rt = []
        _break_key_allKeys = []
        TR_key.keys = []
        TR_key.rt = []
        _TR_key_allKeys = []
        # keep track of which components have finished
        taskComponents = [sand_box, foot, reward, nav_key_task, break_key, progressbar_background, progress_bar, reward_progress, plus_coin_txt, reward_A_feedback, TR_key]
        for thisComponent in taskComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from task_code
            loop_no += 1
            #TRs = TR_count.getKeys(keyList = ['5'], clear = False)
            #if len(TRs) > 0:
            #    print(TRs[-1].name)
            
            # first check if you have found the final reward.
            if index_rew[-1] == len(states):
                # reward_waiting = random.uniform(1, 3)
                reward_waiting = secs_per_step[-1]
                thisExp.addData('reward_delay', reward_waiting)
                resultTbl.addData('reward_delay', reward_waiting)
                wait(reward_waiting)
                thisExp.addData('t_reward_afterwait', globalClock.getTime())
                resultTbl.addData('t_reward_afterwait', globalClock.getTime())
                break 
            
            # activate these only if just starting or if a step has been made
            if loop_no == 1: 
                curr_state = states[index_rew[-1]]
                curr_rew_x = rew_x[index_rew[-1]] 
                curr_rew_y = rew_y[index_rew[-1]]
                
                # create new jitter for the first subpath.
                curr_exp_step_no = all_exp_step_nos[index_rew[-1]]
                secs_per_step = jitter(curr_exp_step_no, 5.75, var_per_step=1)
                resultTbl.addData('jitters_subpath', secs_per_step)
                thisExp.addData('jitters_subpath', secs_per_step)
                
                thisExp.addData('start_ABCD_screen', globalClock.getTime())
                resultTbl.addData('start_ABCD_screen', globalClock.getTime())
                thisExp.addData('curr_loc_x', curr_x)
                thisExp.addData('curr_loc_y', curr_y)
                resultTbl.addData('curr_loc_x', curr_x)
                resultTbl.addData('curr_loc_y', curr_y)
                resultTbl.addData('curr_rew_x', curr_rew_x)
                resultTbl.addData('curr_rew_y', curr_rew_y)
                resultTbl.addData('repeat', rep_runs.thisRepN)
                resultTbl.addData('type', type)
                #if len(TRs) > 0:
                    #thisExp.nextEntry()
                    #thisExp.addData('first_task_TR', TRs[0].rt)
                    #print(TRs.rt[0])
            
            
            
            if (key1_press_trigger == 0) and (key2_press_trigger == 0) and (key3_press_trigger == 0) and (key4_press_trigger == 0) and (progress_bar_on == 0):
                keys = nav_key_task.getKeys(keyList = ['1', '2', '3', '4'], clear = False)
                no_keys_pressed.append(len(keys))
            
            if len(keys) > 0:
                # first check if you last found a reward. this means that you can't 
                # react to key presses for a bit.
                if index_rew[-1] > index_rew[-2]:
                    # enter loop where you framewise increase the progress_height 
                    # until the reward_wait time is elapsed.
                    if progress_bar_on == 0:
                        #reward_waiting = random.uniform(1,3)
                        reward_waiting = secs_per_step[-1]
                        thisExp.addData('reward_delay', reward_waiting)
                        resultTbl.addData('reward_delay', reward_waiting)
                        last_height = height_progress
                        time_rew_found = globalClock.getTime()
                        progress_update = time_rew_found + interval
                        progress_bar_on = 1
                    if progress_timer[-1] < time_rew_found+reward_waiting:
                        progress_timer.append(globalClock.getTime())
                        if progress_timer[-1] >= progress_update:
                            #it can be max 0.43 long. I have 3 repeats and 4 rews. > increase by 0.035 each rew
                            # I now have variable repeats. reps_per_run 
                            height_progress += ((0.41/(4*reps_per_run))/(reward_waiting/interval))
                            progress_update += interval #increment by 100ms
                    if progress_timer[-1] >= time_rew_found+reward_waiting:
                        height_progress = last_height + (0.41/(4*reps_per_run))
                        progress_bar_on = 0
                        print('done progress bar update')
                        # and save that you made a step.
                        thisExp.addData('t_reward_afterwait', globalClock.getTime())
                        resultTbl.addData('t_reward_afterwait', globalClock.getTime())
                        # and make it invisible again
                        rew_visible = 0
                        A_visible = 0
                        # and add the same last value, so it wouldn't enter this loop again.
                        index_rew.append(index_rew[-1])
                        # and update the reward!! now off to find the next one.
                        curr_state = states[index_rew[-1]]
                        curr_rew_x = rew_x[index_rew[-1]] 
                        curr_rew_y = rew_y[index_rew[-1]]
                        
                        # and move on to the next subpath_length! 
                        curr_exp_step_no = all_exp_step_nos[index_rew[-1]]
                        secs_per_step = jitter(curr_exp_step_no, 5.75, var_per_step=1.5)
                        steps_taken_this_subpath = 0
                        resultTbl.addData('jitters_subpath', secs_per_step)
                        thisExp.addData('jitters_subpath', secs_per_step)
                        
                        print(f'Reward found and waited! These should all be 0: key 1:{key1_press_trigger} key 2:{key2_press_trigger} key3:{key3_press_trigger} key 4:{key4_press_trigger}')
                        print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                # if you didnt last found a reward, you can check for key presses.   
                # then check if a new key has been pressed or if we are still updating the position.
                elif (no_keys_pressed[-1] > no_keys_pressed[-2]) or (key1_press_trigger == 1) or (key2_press_trigger == 1) or (key3_press_trigger == 1) or (key4_press_trigger == 1):
                    # check which key has been pressed
                    #define how long the steps are supposed to be
                    if (keys[-1].name == '1') or (key1_press_trigger == 1):
                        if (curr_x> -21/100):
                            key1_press_trigger = 1
                            if move_counter == 0:
                                last_x = curr_x
                                time_at_press = globalClock.getTime()
                                thisExp.addData('t_step_press_global', time_at_press)
                                resultTbl.addData('t_step_press_global', time_at_press)
                                thisExp.addData('t_step_press_curr_run', keys[-1].rt)
                                resultTbl.addData('t_step_press_curr_run', keys[-1].rt)
                                update_timer = time_at_press + interval
                                move_counter = 1
                                
                                if steps_taken_this_subpath < curr_exp_step_no:
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps and optimal stepno is {curr_exp_step_no}.')
                                else:
                                    isi = random.uniform(0.6, 1.5)
                                
                                thisExp.nextEntry()
                                resultTbl.nextEntry()
                                thisExp.addData('length_step', isi)
                                resultTbl.addData('length_step', isi)
                                print(isi)
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_x -= ((21/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                                    if curr_x < -21/100:
                                        curr_x = -21/100
                            if move_timer[-1] >= time_at_press+isi:
                                curr_x = last_x - 21/100
                                move_counter = 0
                                key1_press_trigger = 0
                                print('done moving 1')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('curr_loc_x', curr_x)
                                resultTbl.addData('curr_loc_y', curr_y)
                                resultTbl.addData('curr_rew_x', curr_rew_x)
                                resultTbl.addData('curr_rew_y', curr_rew_y)
                                resultTbl.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('repeat', rep_runs.thisRepN)
                                resultTbl.addData('type', type)
                                resultTbl.addData('state', states[index_rew[-1]])
                    # check which key has been pressed
                    if (keys[-1].name == '2') or (key2_press_trigger == 1):
                        if curr_y < 29/100:
                            key2_press_trigger = 1
                            if move_counter == 0:
                                last_y = curr_y
                                time_at_press = globalClock.getTime()
                                thisExp.addData('t_step_press_global', time_at_press)
                                resultTbl.addData('t_step_press_global', time_at_press)
                                thisExp.addData('t_step_press_curr_run', keys[-1].rt)
                                resultTbl.addData('t_step_press_curr_run', keys[-1].rt)
                                update_timer = time_at_press + interval
                                move_counter = 1
                                if steps_taken_this_subpath < curr_exp_step_no:
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps, {curr_exp_step_no}.')
                                else:
                                    isi = random.uniform(0.6, 1.5)
                                    print(curr_exp_step_no)
                                    
                                thisExp.nextEntry()
                                resultTbl.nextEntry()
                                thisExp.addData('length_step', isi)
                                resultTbl.addData('length_step', isi)
                                print(isi)
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_y += ((29/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                                    if curr_y > 29/100:
                                        curr_y = 29/100
                            if move_timer[-1] >= time_at_press+isi:
                                curr_y = last_y + 29/100
                                move_counter = 0
                                key2_press_trigger = 0
                                print('done moving 2')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('curr_loc_x', curr_x)
                                resultTbl.addData('curr_loc_y', curr_y)
                                resultTbl.addData('curr_rew_x', curr_rew_x)
                                resultTbl.addData('curr_rew_y', curr_rew_y)
                                resultTbl.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('type', type)
                                resultTbl.addData('state', states[index_rew[-1]])
                    # check which key had been pressed
                    if (keys[-1].name == '3') or (key3_press_trigger == 1):
                        if curr_y > -29/100:
                            key3_press_trigger = 1
                            if move_counter == 0:
                                last_y = curr_y
                                time_at_press = globalClock.getTime()
                                thisExp.addData('t_step_press_curr_run', keys[-1].rt)
                                resultTbl.addData('t_step_press_curr_run', keys[-1].rt)
                                thisExp.addData('t_step_press_global', time_at_press)
                                resultTbl.addData('t_step_press_global', time_at_press)
                                update_timer = time_at_press + interval
                                move_counter = 1
                                if steps_taken_this_subpath < curr_exp_step_no:
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps.')
                                else:
                                    isi = random.uniform(0.6, 1.5)
                                    
                                thisExp.nextEntry()
                                resultTbl.nextEntry()
                                thisExp.addData('length_step', isi)
                                resultTbl.addData('length_step', isi)
                                print(isi)
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_y -= ((29/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                                    if curr_y < -29/100:
                                        curr_y = -29/100
                            if move_timer[-1] >= time_at_press+isi:
                                curr_y = last_y - 29/100
                                move_counter = 0
                                key3_press_trigger = 0
                                print('done moving 3')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('curr_loc_x', curr_x)
                                resultTbl.addData('curr_loc_y', curr_y)
                                resultTbl.addData('curr_rew_x', curr_rew_x)
                                resultTbl.addData('curr_rew_y', curr_rew_y)
                                resultTbl.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('type', type)
                                resultTbl.addData('state', states[index_rew[-1]])
                    # check which keys have been pressed
                    if (keys[-1].name == '4') or (key4_press_trigger == 1):
                        if curr_x < 21/100:
                            key4_press_trigger = 1
                            if move_counter == 0:
                                last_x = curr_x
                                time_at_press = globalClock.getTime()
                                thisExp.addData('t_step_press_global', time_at_press)
                                resultTbl.addData('t_step_press_global', time_at_press)
                                thisExp.addData('t_step_press_curr_run', keys[-1].rt)
                                resultTbl.addData('t_step_press_curr_run', keys[-1].rt)
                                update_timer = time_at_press + interval
                                move_counter = 1
                                if steps_taken_this_subpath < curr_exp_step_no:
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps.')
                                else:
                                    isi = random.uniform(0.6, 1.5)
            
                                thisExp.nextEntry()
                                resultTbl.nextEntry()
                                thisExp.addData('length_step', isi)
                                resultTbl.addData('length_step', isi)
                                print(isi)
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_x += ((21/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                                    if curr_x > 21/100:
                                        curr_x = 21/100
                            if move_timer[-1] >= time_at_press+isi:
                                curr_x = last_x + 21/100
                                move_counter = 0
                                key4_press_trigger = 0
                                print('done moving 4')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('curr_loc_x', curr_x)
                                resultTbl.addData('curr_loc_y', curr_y)
                                resultTbl.addData('curr_rew_x', curr_rew_x)
                                resultTbl.addData('curr_rew_y', curr_rew_y)
                                resultTbl.addData('t_step_end_global', globalClock.getTime())
                                resultTbl.addData('type', type)
                                resultTbl.addData('state', states[index_rew[-1]])
                    # then check if reward location and curr loc are the same
                    if (curr_x == curr_rew_x) and (curr_y == curr_rew_y):
                        # display something if it's A
                        if index_rew[-1] == 0:
                            A_visible = 1
                        # go to next reward
                        index_rew.append(index_rew[-1]+1)
                        print('found reward')
                        # show the reward!
                        rew_visible = 1
                        thisExp.addData('rew_loc_x', curr_rew_x)
                        thisExp.addData('rew_loc_y', curr_rew_y)
                        thisExp.addData('t_reward_start', globalClock.getTime())
                        resultTbl.addData('rew_loc_x', curr_rew_x)
                        resultTbl.addData('rew_loc_y', curr_rew_y)
                        resultTbl.addData('t_reward_start', globalClock.getTime())
                        progress1 = 0
                        steps_taken_this_subpath = 0
            
            # check if the safety key has been pressed
            safety_key = break_key.getKeys(keyList = ['0'], clear = False)
            if len(safety_key) > 0:
                curr_x = 0
                curr_y = 0
                
                
            # then let everything update all the shapes.
            
            # *sand_box* updates
            
            # if sand_box is starting this frame...
            if sand_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sand_box.frameNStart = frameN  # exact frame index
                sand_box.tStart = t  # local t and not account for scr refresh
                sand_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sand_box, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sand_box.started')
                # update status
                sand_box.status = STARTED
                sand_box.setAutoDraw(True)
            
            # if sand_box is active this frame...
            if sand_box.status == STARTED:
                # update params
                pass
            
            # *foot* updates
            
            # if foot is starting this frame...
            if foot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                foot.frameNStart = frameN  # exact frame index
                foot.tStart = t  # local t and not account for scr refresh
                foot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(foot, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'foot.started')
                # update status
                foot.status = STARTED
                foot.setAutoDraw(True)
            
            # if foot is active this frame...
            if foot.status == STARTED:
                # update params
                foot.setPos((curr_x, curr_y), log=False)
            
            # *reward* updates
            
            # if reward is starting this frame...
            if reward.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reward.frameNStart = frameN  # exact frame index
                reward.tStart = t  # local t and not account for scr refresh
                reward.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reward, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward.started')
                # update status
                reward.status = STARTED
                reward.setAutoDraw(True)
            
            # if reward is active this frame...
            if reward.status == STARTED:
                # update params
                reward.setOpacity(rew_visible, log=False)
                reward.setPos((curr_rew_x, curr_rew_y), log=False)
            
            # *nav_key_task* updates
            waitOnFlip = False
            
            # if nav_key_task is starting this frame...
            if nav_key_task.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nav_key_task.frameNStart = frameN  # exact frame index
                nav_key_task.tStart = t  # local t and not account for scr refresh
                nav_key_task.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nav_key_task, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'nav_key_task.started')
                # update status
                nav_key_task.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(nav_key_task.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(nav_key_task.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if nav_key_task.status == STARTED and not waitOnFlip:
                theseKeys = nav_key_task.getKeys(keyList=['1', '2', '3', '4'], waitRelease=False)
                _nav_key_task_allKeys.extend(theseKeys)
                if len(_nav_key_task_allKeys):
                    nav_key_task.keys = [key.name for key in _nav_key_task_allKeys]  # storing all keys
                    nav_key_task.rt = [key.rt for key in _nav_key_task_allKeys]
                    nav_key_task.duration = [key.duration for key in _nav_key_task_allKeys]
            
            # *break_key* updates
            waitOnFlip = False
            
            # if break_key is starting this frame...
            if break_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_key.frameNStart = frameN  # exact frame index
                break_key.tStart = t  # local t and not account for scr refresh
                break_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_key.started')
                # update status
                break_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(break_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(break_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if break_key.status == STARTED and not waitOnFlip:
                theseKeys = break_key.getKeys(keyList=['0'], waitRelease=False)
                _break_key_allKeys.extend(theseKeys)
                if len(_break_key_allKeys):
                    break_key.keys = _break_key_allKeys[-1].name  # just the last key pressed
                    break_key.rt = _break_key_allKeys[-1].rt
                    break_key.duration = _break_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *progressbar_background* updates
            
            # if progressbar_background is starting this frame...
            if progressbar_background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                progressbar_background.frameNStart = frameN  # exact frame index
                progressbar_background.tStart = t  # local t and not account for scr refresh
                progressbar_background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(progressbar_background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'progressbar_background.started')
                # update status
                progressbar_background.status = STARTED
                progressbar_background.setAutoDraw(True)
            
            # if progressbar_background is active this frame...
            if progressbar_background.status == STARTED:
                # update params
                pass
            
            # *progress_bar* updates
            
            # if progress_bar is starting this frame...
            if progress_bar.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                progress_bar.frameNStart = frameN  # exact frame index
                progress_bar.tStart = t  # local t and not account for scr refresh
                progress_bar.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(progress_bar, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'progress_bar.started')
                # update status
                progress_bar.status = STARTED
                progress_bar.setAutoDraw(True)
            
            # if progress_bar is active this frame...
            if progress_bar.status == STARTED:
                # update params
                progress_bar.setSize((0.1, height_progress), log=False)
            
            # *reward_progress* updates
            
            # if reward_progress is starting this frame...
            if reward_progress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reward_progress.frameNStart = frameN  # exact frame index
                reward_progress.tStart = t  # local t and not account for scr refresh
                reward_progress.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reward_progress, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_progress.started')
                # update status
                reward_progress.status = STARTED
                reward_progress.setAutoDraw(True)
            
            # if reward_progress is active this frame...
            if reward_progress.status == STARTED:
                # update params
                reward_progress.setOpacity(rew_visible, log=False)
            
            # *plus_coin_txt* updates
            
            # if plus_coin_txt is starting this frame...
            if plus_coin_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                plus_coin_txt.frameNStart = frameN  # exact frame index
                plus_coin_txt.tStart = t  # local t and not account for scr refresh
                plus_coin_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(plus_coin_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'plus_coin_txt.started')
                # update status
                plus_coin_txt.status = STARTED
                plus_coin_txt.setAutoDraw(True)
            
            # if plus_coin_txt is active this frame...
            if plus_coin_txt.status == STARTED:
                # update params
                plus_coin_txt.setOpacity(rew_visible, log=False)
            
            # *reward_A_feedback* updates
            
            # if reward_A_feedback is starting this frame...
            if reward_A_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reward_A_feedback.frameNStart = frameN  # exact frame index
                reward_A_feedback.tStart = t  # local t and not account for scr refresh
                reward_A_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reward_A_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reward_A_feedback.started')
                # update status
                reward_A_feedback.status = STARTED
                reward_A_feedback.setAutoDraw(True)
            
            # if reward_A_feedback is active this frame...
            if reward_A_feedback.status == STARTED:
                # update params
                reward_A_feedback.setOpacity(A_visible, log=False)
            
            # *TR_key* updates
            
            # if TR_key is starting this frame...
            if TR_key.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                TR_key.frameNStart = frameN  # exact frame index
                TR_key.tStart = t  # local t and not account for scr refresh
                TR_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(TR_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('TR_key.started', t)
                # update status
                TR_key.status = STARTED
                # keyboard checking is just starting
                TR_key.clock.reset()  # now t=0
            if TR_key.status == STARTED:
                theseKeys = TR_key.getKeys(keyList=['5'], waitRelease=False)
                _TR_key_allKeys.extend(theseKeys)
                if len(_TR_key_allKeys):
                    TR_key.keys = [key.name for key in _TR_key_allKeys]  # storing all keys
                    TR_key.rt = [key.rt for key in _TR_key_allKeys]
                    TR_key.duration = [key.duration for key in _TR_key_allKeys]
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in taskComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task" ---
        for thisComponent in taskComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if nav_key_task.keys in ['', [], None]:  # No response was made
            nav_key_task.keys = None
        rep_runs.addData('nav_key_task.keys',nav_key_task.keys)
        if nav_key_task.keys != None:  # we had a response
            rep_runs.addData('nav_key_task.rt', nav_key_task.rt)
            rep_runs.addData('nav_key_task.duration', nav_key_task.duration)
        # check responses
        if break_key.keys in ['', [], None]:  # No response was made
            break_key.keys = None
        rep_runs.addData('break_key.keys',break_key.keys)
        if break_key.keys != None:  # we had a response
            rep_runs.addData('break_key.rt', break_key.rt)
            rep_runs.addData('break_key.duration', break_key.duration)
        # check responses
        if TR_key.keys in ['', [], None]:  # No response was made
            TR_key.keys = None
        rep_runs.addData('TR_key.keys',TR_key.keys)
        if TR_key.keys != None:  # we had a response
            rep_runs.addData('TR_key.rt', TR_key.rt)
            rep_runs.addData('TR_key.duration', TR_key.duration)
        # the Routine "task" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed reps_per_run repeats of 'rep_runs'
    
    
    # --- Prepare to start Routine "feedback_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    feedback_msg= f'This round, you were better than {feedback}% of the candidates. Try to be even more precise and walk only the same routes to impress me! Yer gold will be more plenty as well...'
    feedback_text.setText(feedback_msg)
    # keep track of which components have finished
    feedback_screenComponents = [background_feedback, feedback_text]
    for thisComponent in feedback_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "feedback_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_feedback* updates
        
        # if background_feedback is starting this frame...
        if background_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_feedback.frameNStart = frameN  # exact frame index
            background_feedback.tStart = t  # local t and not account for scr refresh
            background_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_feedback, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'background_feedback.started')
            # update status
            background_feedback.status = STARTED
            background_feedback.setAutoDraw(True)
        
        # if background_feedback is active this frame...
        if background_feedback.status == STARTED:
            # update params
            pass
        
        # if background_feedback is stopping this frame...
        if background_feedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > background_feedback.tStartRefresh + 3.5-frameTolerance:
                # keep track of stop time/frame for later
                background_feedback.tStop = t  # not accounting for scr refresh
                background_feedback.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'background_feedback.stopped')
                # update status
                background_feedback.status = FINISHED
                background_feedback.setAutoDraw(False)
        
        # *feedback_text* updates
        
        # if feedback_text is starting this frame...
        if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            feedback_text.frameNStart = frameN  # exact frame index
            feedback_text.tStart = t  # local t and not account for scr refresh
            feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'feedback_text.started')
            # update status
            feedback_text.status = STARTED
            feedback_text.setAutoDraw(True)
        
        # if feedback_text is active this frame...
        if feedback_text.status == STARTED:
            # update params
            pass
        
        # if feedback_text is stopping this frame...
        if feedback_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > feedback_text.tStartRefresh + 3.5-frameTolerance:
                # keep track of stop time/frame for later
                feedback_text.tStop = t  # not accounting for scr refresh
                feedback_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_text.stopped')
                # update status
                feedback_text.status = FINISHED
                feedback_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedback_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "feedback_screen" ---
    for thisComponent in feedback_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.500000)
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'trials'

# Run 'End Experiment' code from task_code
resultTbl.saveAsWideText(result_name+'.csv', delim='auto')
resultTbl.saveAsPickle(result_name)

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
