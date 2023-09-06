#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.1.1),
    on September 04, 2023, at 15:28
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
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



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.1.1'
expName = '3x3_practice_v2'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
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
    originPath='C:\\Users\\behrenslab-fmrib\\Documents\\Psychopy\\musicbox\\experiment\\3x3_practice_v2_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1440], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='dell_compneuro_laptop', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='norm')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# Setup ioHub
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

# Initialize components for Routine "intro_0"
intro_0Clock = core.Clock()
ship_intro_0 = visual.ImageStim(
    win=win,
    name='ship_intro_0', 
    image='images/pirates.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
text = visual.TextStim(win=win, name='text',
    text='Ahoy, landlubber!',
    font='Open Sans',
    pos=(0, -0.6), height=0.2, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
ship_box_intro_0 = visual.ImageStim(
    win=win,
    name='ship_box_intro_0', 
    image='images/ship_box.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2, 2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
text_intro_0 = visual.TextStim(win=win, name='text_intro_0',
    text='You have grown weary of life on land and decided to embark on an adventure, aiming to become a part of the renowned pirate crew aboard the majestic ship "Sea Serpent\'s Fury." However, you face a minor setback, as there are several other candidates also vying to join the crew at the same time. To assess your treasure-hunting capabilities, the Captain Seraphina Storme has arranged a test for you and the other candidates.\n\nPress 1 to continue!',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
next_intro_0 = keyboard.Keyboard()

# Initialize components for Routine "intro_1"
intro_1Clock = core.Clock()
ship_box_intro_1 = visual.ImageStim(
    win=win,
    name='ship_box_intro_1', 
    image='images/ship_box.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
text_intro_1 = visual.TextStim(win=win, name='text_intro_1',
    text='On a secluded and sandy island, the captain will evaluate how well you can locate treasures hidden by the pirates. Depending on your performance, your chances of fulfilling your dream of becoming a true pirate will increase.\n\nYou will be provided with one set of coordinates for the treasures, which you must find in a specific order. \n\nPress 1 to continue!',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
next_intro_1 = keyboard.Keyboard()

# Initialize components for Routine "intro_2"
intro_2Clock = core.Clock()
treasure_box_intro_2 = visual.ImageStim(
    win=win,
    name='treasure_box_intro_2', 
    image='images/map_w_txt.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
text_intro_2 = visual.TextStim(win=win, name='text_intro_2',
    text='To navigate through the sand, you can use the keys 1, 2, 3, and 4.\nPlease use your right hand so that your index finger (1) will take you left, your middle finger (2) moves you north, your ring finger (3) leads you south, and your little finger (4) guides you right.\n\nPress 1 to start the practice!',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
next_intro_2 = keyboard.Keyboard()
navigation_img = visual.ImageStim(
    win=win,
    name='navigation_img', 
    image='images/footprints_buttons.png', mask=None, anchor='center',
    ori=0.0, pos=(0, -0.6), size=(0.4, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)

# Initialize components for Routine "up_down_practice"
up_down_practiceClock = core.Clock()
parrot_updown_empty = visual.ImageStim(
    win=win,
    name='parrot_updown_empty', 
    image='images/parrot_no_txt.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
help_txt_updown = visual.TextStim(win=win, name='help_txt_updown',
    text="'Arrrr! You look like my favorite future pirate. I will help you!'",
    font='Open Sans',
    pos=(0, -0.5), height=0.05, wrapWidth=1.0, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
updown_pract = visual.ImageStim(
    win=win,
    name='updown_pract', 
    image='images/updown_parrot.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
feedback_updown_txt = visual.TextStim(win=win, name='feedback_updown_txt',
    text='',
    font='Open Sans',
    pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);
nav_vert_key = keyboard.Keyboard()
foot_vert = visual.ImageStim(
    win=win,
    name='foot_vert', 
    image='images/footprints_buttons.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.2, 0.26),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-6.0)

# Initialize components for Routine "left_right_practice"
left_right_practiceClock = core.Clock()
left_right_pract = visual.ImageStim(
    win=win,
    name='left_right_pract', 
    image='images/leftright_parrot.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2, 2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
foot_leftright = visual.ImageStim(
    win=win,
    name='foot_leftright', 
    image='images/footprints_buttons.png', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=(0.2, 0.26),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
intro_leftright = visual.TextStim(win=win, name='intro_leftright',
    text="'Arrrr! Well done! Now walk to the left and to the right.'",
    font='Open Sans',
    pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
feedback_leftright = visual.TextStim(win=win, name='feedback_leftright',
    text='',
    font='Open Sans',
    pos=(0, -0.5), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);
nav_horz_key = keyboard.Keyboard()

# Initialize components for Routine "summary_intro"
summary_introClock = core.Clock()
summary_parrot = visual.ImageStim(
    win=win,
    name='summary_parrot', 
    image='images/summary_parrot.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
next_summary_key = keyboard.Keyboard()

# Initialize components for Routine "new_rewards_prep"
new_rewards_prepClock = core.Clock()
new_rewards = visual.ImageStim(
    win=win,
    name='new_rewards', 
    image='images/sand_3x3grid_pirate_new.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)

# Initialize components for Routine "parrot_tip"
parrot_tipClock = core.Clock()
parrot = visual.ImageStim(
    win=win,
    name='parrot', 
    image='images/parrot_tip.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)

# Initialize components for Routine "show_rewards"
show_rewardsClock = core.Clock()
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

# Initialize components for Routine "task"
taskClock = core.Clock()
from psychopy.core import wait as wait
import random 

states = ['A', 'B', 'C', 'D']
field_nos = ['','','','']

curr_x = 0
curr_y = 0

rew_visible = 1

printing = 'nothing happened yet'
print(printing)
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
polygon = visual.Rect(
    win=win, name='polygon',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=(-0.5, -0.21), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[0.7098, 0.2941, -0.7490], fillColor=[0.7098, 0.2941, -0.7490],
    opacity=None, depth=-5.0, interpolate=True)
reward_progress = visual.ImageStim(
    win=win,
    name='reward_progress', 
    image='images/coin.png', mask=None, anchor='center',
    ori=0.0, pos=(-0.5, -0.41), size=(0.1, 0.13),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-6.0)
plus_coin_txt = visual.TextStim(win=win, name='plus_coin_txt',
    text='+1 coin',
    font='Open Sans',
    pos=(-0.5, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='black', colorSpace='rgb', opacity=1.0, 
    languageStyle='LTR',
    depth=-7.0);
break_key = keyboard.Keyboard()

# Initialize components for Routine "feedback_screen"
feedback_screenClock = core.Clock()
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
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
skip = data.TrialHandler(nReps=0.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='skip')
thisExp.addLoop(skip)  # add the loop to the experiment
thisSkip = skip.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisSkip.rgb)
if thisSkip != None:
    for paramName in thisSkip:
        exec('{} = thisSkip[paramName]'.format(paramName))

for thisSkip in skip:
    currentLoop = skip
    # abbreviate parameter names if possible (e.g. rgb = thisSkip.rgb)
    if thisSkip != None:
        for paramName in thisSkip:
            exec('{} = thisSkip[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "intro_0"-------
    continueRoutine = True
    # update component parameters for each repeat
    next_intro_0.keys = []
    next_intro_0.rt = []
    _next_intro_0_allKeys = []
    # keep track of which components have finished
    intro_0Components = [ship_intro_0, text, ship_box_intro_0, text_intro_0, next_intro_0]
    for thisComponent in intro_0Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    intro_0Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "intro_0"-------
    while continueRoutine:
        # get current time
        t = intro_0Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=intro_0Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ship_intro_0* updates
        if ship_intro_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ship_intro_0.frameNStart = frameN  # exact frame index
            ship_intro_0.tStart = t  # local t and not account for scr refresh
            ship_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ship_intro_0, 'tStartRefresh')  # time at next scr refresh
            ship_intro_0.setAutoDraw(True)
        if ship_intro_0.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ship_intro_0.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                ship_intro_0.tStop = t  # not accounting for scr refresh
                ship_intro_0.frameNStop = frameN  # exact frame index
                win.timeOnFlip(ship_intro_0, 'tStopRefresh')  # time at next scr refresh
                ship_intro_0.setAutoDraw(False)
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)
        
        # *ship_box_intro_0* updates
        if ship_box_intro_0.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
            # keep track of start time/frame for later
            ship_box_intro_0.frameNStart = frameN  # exact frame index
            ship_box_intro_0.tStart = t  # local t and not account for scr refresh
            ship_box_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ship_box_intro_0, 'tStartRefresh')  # time at next scr refresh
            ship_box_intro_0.setAutoDraw(True)
        
        # *text_intro_0* updates
        if text_intro_0.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
            # keep track of start time/frame for later
            text_intro_0.frameNStart = frameN  # exact frame index
            text_intro_0.tStart = t  # local t and not account for scr refresh
            text_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_0, 'tStartRefresh')  # time at next scr refresh
            text_intro_0.setAutoDraw(True)
        
        # *next_intro_0* updates
        waitOnFlip = False
        if next_intro_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            next_intro_0.frameNStart = frameN  # exact frame index
            next_intro_0.tStart = t  # local t and not account for scr refresh
            next_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_intro_0, 'tStartRefresh')  # time at next scr refresh
            next_intro_0.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(next_intro_0.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(next_intro_0.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if next_intro_0.status == STARTED and not waitOnFlip:
            theseKeys = next_intro_0.getKeys(keyList=['1'], waitRelease=False)
            _next_intro_0_allKeys.extend(theseKeys)
            if len(_next_intro_0_allKeys):
                next_intro_0.keys = _next_intro_0_allKeys[-1].name  # just the last key pressed
                next_intro_0.rt = _next_intro_0_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_0Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "intro_0"-------
    for thisComponent in intro_0Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    skip.addData('ship_intro_0.started', ship_intro_0.tStartRefresh)
    skip.addData('ship_intro_0.stopped', ship_intro_0.tStopRefresh)
    skip.addData('text.started', text.tStartRefresh)
    skip.addData('text.stopped', text.tStopRefresh)
    skip.addData('ship_box_intro_0.started', ship_box_intro_0.tStartRefresh)
    skip.addData('ship_box_intro_0.stopped', ship_box_intro_0.tStopRefresh)
    skip.addData('text_intro_0.started', text_intro_0.tStartRefresh)
    skip.addData('text_intro_0.stopped', text_intro_0.tStopRefresh)
    # check responses
    if next_intro_0.keys in ['', [], None]:  # No response was made
        next_intro_0.keys = None
    skip.addData('next_intro_0.keys',next_intro_0.keys)
    if next_intro_0.keys != None:  # we had a response
        skip.addData('next_intro_0.rt', next_intro_0.rt)
    skip.addData('next_intro_0.started', next_intro_0.tStartRefresh)
    skip.addData('next_intro_0.stopped', next_intro_0.tStopRefresh)
    # the Routine "intro_0" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "intro_1"-------
    continueRoutine = True
    # update component parameters for each repeat
    next_intro_1.keys = []
    next_intro_1.rt = []
    _next_intro_1_allKeys = []
    # keep track of which components have finished
    intro_1Components = [ship_box_intro_1, text_intro_1, next_intro_1]
    for thisComponent in intro_1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    intro_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "intro_1"-------
    while continueRoutine:
        # get current time
        t = intro_1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=intro_1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *ship_box_intro_1* updates
        if ship_box_intro_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ship_box_intro_1.frameNStart = frameN  # exact frame index
            ship_box_intro_1.tStart = t  # local t and not account for scr refresh
            ship_box_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ship_box_intro_1, 'tStartRefresh')  # time at next scr refresh
            ship_box_intro_1.setAutoDraw(True)
        
        # *text_intro_1* updates
        if text_intro_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_1.frameNStart = frameN  # exact frame index
            text_intro_1.tStart = t  # local t and not account for scr refresh
            text_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_1, 'tStartRefresh')  # time at next scr refresh
            text_intro_1.setAutoDraw(True)
        
        # *next_intro_1* updates
        waitOnFlip = False
        if next_intro_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            next_intro_1.frameNStart = frameN  # exact frame index
            next_intro_1.tStart = t  # local t and not account for scr refresh
            next_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_intro_1, 'tStartRefresh')  # time at next scr refresh
            next_intro_1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(next_intro_1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(next_intro_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if next_intro_1.status == STARTED and not waitOnFlip:
            theseKeys = next_intro_1.getKeys(keyList=['1'], waitRelease=False)
            _next_intro_1_allKeys.extend(theseKeys)
            if len(_next_intro_1_allKeys):
                next_intro_1.keys = _next_intro_1_allKeys[-1].name  # just the last key pressed
                next_intro_1.rt = _next_intro_1_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "intro_1"-------
    for thisComponent in intro_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    skip.addData('ship_box_intro_1.started', ship_box_intro_1.tStartRefresh)
    skip.addData('ship_box_intro_1.stopped', ship_box_intro_1.tStopRefresh)
    skip.addData('text_intro_1.started', text_intro_1.tStartRefresh)
    skip.addData('text_intro_1.stopped', text_intro_1.tStopRefresh)
    # check responses
    if next_intro_1.keys in ['', [], None]:  # No response was made
        next_intro_1.keys = None
    skip.addData('next_intro_1.keys',next_intro_1.keys)
    if next_intro_1.keys != None:  # we had a response
        skip.addData('next_intro_1.rt', next_intro_1.rt)
    skip.addData('next_intro_1.started', next_intro_1.tStartRefresh)
    skip.addData('next_intro_1.stopped', next_intro_1.tStopRefresh)
    # the Routine "intro_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "intro_2"-------
    continueRoutine = True
    # update component parameters for each repeat
    next_intro_2.keys = []
    next_intro_2.rt = []
    _next_intro_2_allKeys = []
    # keep track of which components have finished
    intro_2Components = [treasure_box_intro_2, text_intro_2, next_intro_2, navigation_img]
    for thisComponent in intro_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    intro_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "intro_2"-------
    while continueRoutine:
        # get current time
        t = intro_2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=intro_2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *treasure_box_intro_2* updates
        if treasure_box_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            treasure_box_intro_2.frameNStart = frameN  # exact frame index
            treasure_box_intro_2.tStart = t  # local t and not account for scr refresh
            treasure_box_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(treasure_box_intro_2, 'tStartRefresh')  # time at next scr refresh
            treasure_box_intro_2.setAutoDraw(True)
        
        # *text_intro_2* updates
        if text_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_2.frameNStart = frameN  # exact frame index
            text_intro_2.tStart = t  # local t and not account for scr refresh
            text_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_2, 'tStartRefresh')  # time at next scr refresh
            text_intro_2.setAutoDraw(True)
        
        # *next_intro_2* updates
        waitOnFlip = False
        if next_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            next_intro_2.frameNStart = frameN  # exact frame index
            next_intro_2.tStart = t  # local t and not account for scr refresh
            next_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_intro_2, 'tStartRefresh')  # time at next scr refresh
            next_intro_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(next_intro_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(next_intro_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if next_intro_2.status == STARTED and not waitOnFlip:
            theseKeys = next_intro_2.getKeys(keyList=['1'], waitRelease=False)
            _next_intro_2_allKeys.extend(theseKeys)
            if len(_next_intro_2_allKeys):
                next_intro_2.keys = _next_intro_2_allKeys[-1].name  # just the last key pressed
                next_intro_2.rt = _next_intro_2_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *navigation_img* updates
        if navigation_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            navigation_img.frameNStart = frameN  # exact frame index
            navigation_img.tStart = t  # local t and not account for scr refresh
            navigation_img.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(navigation_img, 'tStartRefresh')  # time at next scr refresh
            navigation_img.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "intro_2"-------
    for thisComponent in intro_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    skip.addData('treasure_box_intro_2.started', treasure_box_intro_2.tStartRefresh)
    skip.addData('treasure_box_intro_2.stopped', treasure_box_intro_2.tStopRefresh)
    skip.addData('text_intro_2.started', text_intro_2.tStartRefresh)
    skip.addData('text_intro_2.stopped', text_intro_2.tStopRefresh)
    # check responses
    if next_intro_2.keys in ['', [], None]:  # No response was made
        next_intro_2.keys = None
    skip.addData('next_intro_2.keys',next_intro_2.keys)
    if next_intro_2.keys != None:  # we had a response
        skip.addData('next_intro_2.rt', next_intro_2.rt)
    skip.addData('next_intro_2.started', next_intro_2.tStartRefresh)
    skip.addData('next_intro_2.stopped', next_intro_2.tStopRefresh)
    skip.addData('navigation_img.started', navigation_img.tStartRefresh)
    skip.addData('navigation_img.stopped', navigation_img.tStopRefresh)
    # the Routine "intro_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "up_down_practice"-------
    continueRoutine = True
    # update component parameters for each repeat
    curr_x = 0
    curr_y = 0
    
    all_locs_been_to = False
    locs_walked =[]
    
    no_keys_pressed = [0]
    nav_vert_key.clearEvents()
    nav_vert_key.keys = []
    nav_vert_key.rt = []
    
    msg = "Use your middle finger to go up and your ring finger to go down."
    isi = 2.5
    
    
    interval = 0.05
    move_timer = []
    move_timer.append(globalClock.getTime())
    move_counter = 0
    time_at_press = globalClock.getTime()
    key1_press_trigger = 0
    key2_press_trigger = 0
    key3_press_trigger = 0
    key4_press_trigger = 0
    loopno = 0
    nav_vert_key.keys = []
    nav_vert_key.rt = []
    _nav_vert_key_allKeys = []
    # keep track of which components have finished
    up_down_practiceComponents = [parrot_updown_empty, help_txt_updown, updown_pract, feedback_updown_txt, nav_vert_key, foot_vert]
    for thisComponent in up_down_practiceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    up_down_practiceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "up_down_practice"-------
    while continueRoutine:
        # get current time
        t = up_down_practiceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=up_down_practiceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        loopno += 1
        
        # first check if you practiced enough.
        if all_locs_been_to == True:
            #wait(2)
            break
        
        isi = 2.5
        
        if (key2_press_trigger == 0) and (key3_press_trigger == 0):
            keys = nav_vert_key.getKeys(keyList = ['2', '3'], clear = False)
            no_keys_pressed.append(len(keys))
        
        print(f'whats up with the keys: key length is {len(keys)}, key 1 pressed {key2_press_trigger} {key3_press_trigger}')
        
        if len(keys) > 0:
            # check if a new key has been pressed or if we are still updating the position.
            if (no_keys_pressed[-1] > no_keys_pressed[-2]) or (key2_press_trigger == 1) or (key3_press_trigger == 1):
                # check which key has been pressed
                if (keys[-1].name == '2') or (key2_press_trigger == 1):
                    if curr_y < 29/100:
                        key2_press_trigger = 1
                        if move_counter == 0:
                            print('start moving up')
                            last_y = curr_y
                            time_at_press = globalClock.getTime()
                            update_timer = time_at_press + interval
                            move_counter = 1
                        if move_timer[-1] < time_at_press+isi:
                            move_timer.append(globalClock.getTime())
                            if move_timer[-1] >= update_timer:
                                curr_y += ((28/100)/(isi/interval))
                                update_timer += interval #increment by 100ms
                        elif move_timer[-1] >= time_at_press+isi:
                            curr_y = last_y + 29/100
                            move_counter = 0
                            key2_press_trigger = 0
                            print('done moving 2')
                            direc = 'up'
                            # add the location to the location you walked.
                            locs_walked.append(curr_y)
                            msg = f"Well done! You walked {direc}. Take some more steps!"
                            # and save that you made a step.
                            thisExp.nextEntry()
                            thisExp.addData('curr_loc_x', curr_x)
                            thisExp.addData('curr_loc_y', curr_y)
                            thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                            thisExp.addData('t_step_tglobal', globalClock.getTime())
                            thisExp.addData('length_step', isi)
                # check which key had been pressed
                if (keys[-1].name == '3') or (key3_press_trigger == 1):
                    print(f'current position is {curr_y}. It needs to be bigger than -29/100')
                    if curr_y > -29/100:
                        key3_press_trigger = 1
                        if move_counter == 0:
                            print('start moving down')
                            last_y = curr_y
                            time_at_press = globalClock.getTime()
                            update_timer = time_at_press + interval
                            print(f'update from now ({time_at_press}) until {time_at_press+isi}.')
                            move_counter = 1
                        if move_timer[-1] < time_at_press+isi:
                            move_timer.append(globalClock.getTime())
                            print(f'check if move timer works. should get to bigger then {time_at_press+isi} and then end step. Its now: {move_timer[-1]}')
                            if move_timer[-1] >= update_timer:
                                curr_y -= ((28/100)/(isi/interval))
                                update_timer += interval #increment by 100ms
                        elif move_timer[-1] >= time_at_press+isi:
                            curr_y = last_y - 29/100
                            move_counter = 0
                            key3_press_trigger = 0
                            print('done moving 3')
                            direc = 'down'
                            # add the location to the location you walked.
                            locs_walked.append(curr_y)
                            msg = f"Well done! You walked {direc}. Take some more steps!"
                            # and save that you made a step.
                            thisExp.nextEntry()
                            thisExp.addData('curr_loc_x', curr_x)
                            thisExp.addData('curr_loc_y', curr_y)
                            thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                            thisExp.addData('t_step_tglobal', globalClock.getTime())
                            thisExp.addData('length_step', isi)
        
        # then check if you already walked along the whole line
        if no_keys_pressed[-1] > 10:
            print(f'you pressed {no_keys_pressed[-1]} keys.')
            all_locs_been_to = set([-29/100, 0, 29/100]).issubset(set(locs_walked))
            msg = f"Nice! You seem to be comfortable walking along a vertical line."
        
        
        # *parrot_updown_empty* updates
        if parrot_updown_empty.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            parrot_updown_empty.frameNStart = frameN  # exact frame index
            parrot_updown_empty.tStart = t  # local t and not account for scr refresh
            parrot_updown_empty.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(parrot_updown_empty, 'tStartRefresh')  # time at next scr refresh
            parrot_updown_empty.setAutoDraw(True)
        if parrot_updown_empty.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > parrot_updown_empty.tStartRefresh + 3.5-frameTolerance:
                # keep track of stop time/frame for later
                parrot_updown_empty.tStop = t  # not accounting for scr refresh
                parrot_updown_empty.frameNStop = frameN  # exact frame index
                win.timeOnFlip(parrot_updown_empty, 'tStopRefresh')  # time at next scr refresh
                parrot_updown_empty.setAutoDraw(False)
        
        # *help_txt_updown* updates
        if help_txt_updown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            help_txt_updown.frameNStart = frameN  # exact frame index
            help_txt_updown.tStart = t  # local t and not account for scr refresh
            help_txt_updown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(help_txt_updown, 'tStartRefresh')  # time at next scr refresh
            help_txt_updown.setAutoDraw(True)
        if help_txt_updown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > help_txt_updown.tStartRefresh + 3.5-frameTolerance:
                # keep track of stop time/frame for later
                help_txt_updown.tStop = t  # not accounting for scr refresh
                help_txt_updown.frameNStop = frameN  # exact frame index
                win.timeOnFlip(help_txt_updown, 'tStopRefresh')  # time at next scr refresh
                help_txt_updown.setAutoDraw(False)
        
        # *updown_pract* updates
        if updown_pract.status == NOT_STARTED and tThisFlip >= 3.5-frameTolerance:
            # keep track of start time/frame for later
            updown_pract.frameNStart = frameN  # exact frame index
            updown_pract.tStart = t  # local t and not account for scr refresh
            updown_pract.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(updown_pract, 'tStartRefresh')  # time at next scr refresh
            updown_pract.setAutoDraw(True)
        
        # *feedback_updown_txt* updates
        if feedback_updown_txt.status == NOT_STARTED and tThisFlip >= 3.5-frameTolerance:
            # keep track of start time/frame for later
            feedback_updown_txt.frameNStart = frameN  # exact frame index
            feedback_updown_txt.tStart = t  # local t and not account for scr refresh
            feedback_updown_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_updown_txt, 'tStartRefresh')  # time at next scr refresh
            feedback_updown_txt.setAutoDraw(True)
        if feedback_updown_txt.status == STARTED:  # only update if drawing
            feedback_updown_txt.setText(msg, log=False)
        
        # *nav_vert_key* updates
        waitOnFlip = False
        if nav_vert_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            nav_vert_key.frameNStart = frameN  # exact frame index
            nav_vert_key.tStart = t  # local t and not account for scr refresh
            nav_vert_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(nav_vert_key, 'tStartRefresh')  # time at next scr refresh
            nav_vert_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(nav_vert_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(nav_vert_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if nav_vert_key.status == STARTED and not waitOnFlip:
            theseKeys = nav_vert_key.getKeys(keyList=['2','3'], waitRelease=False)
            _nav_vert_key_allKeys.extend(theseKeys)
            if len(_nav_vert_key_allKeys):
                nav_vert_key.keys = _nav_vert_key_allKeys[-1].name  # just the last key pressed
                nav_vert_key.rt = _nav_vert_key_allKeys[-1].rt
        
        # *foot_vert* updates
        if foot_vert.status == NOT_STARTED and tThisFlip >= 3.5-frameTolerance:
            # keep track of start time/frame for later
            foot_vert.frameNStart = frameN  # exact frame index
            foot_vert.tStart = t  # local t and not account for scr refresh
            foot_vert.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(foot_vert, 'tStartRefresh')  # time at next scr refresh
            foot_vert.setAutoDraw(True)
        if foot_vert.status == STARTED:  # only update if drawing
            foot_vert.setPos((curr_x, curr_y), log=False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in up_down_practiceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "up_down_practice"-------
    for thisComponent in up_down_practiceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    skip.addData('parrot_updown_empty.started', parrot_updown_empty.tStartRefresh)
    skip.addData('parrot_updown_empty.stopped', parrot_updown_empty.tStopRefresh)
    skip.addData('help_txt_updown.started', help_txt_updown.tStartRefresh)
    skip.addData('help_txt_updown.stopped', help_txt_updown.tStopRefresh)
    skip.addData('updown_pract.started', updown_pract.tStartRefresh)
    skip.addData('updown_pract.stopped', updown_pract.tStopRefresh)
    skip.addData('feedback_updown_txt.started', feedback_updown_txt.tStartRefresh)
    skip.addData('feedback_updown_txt.stopped', feedback_updown_txt.tStopRefresh)
    # check responses
    if nav_vert_key.keys in ['', [], None]:  # No response was made
        nav_vert_key.keys = None
    skip.addData('nav_vert_key.keys',nav_vert_key.keys)
    if nav_vert_key.keys != None:  # we had a response
        skip.addData('nav_vert_key.rt', nav_vert_key.rt)
    skip.addData('nav_vert_key.started', nav_vert_key.tStartRefresh)
    skip.addData('nav_vert_key.stopped', nav_vert_key.tStopRefresh)
    skip.addData('foot_vert.started', foot_vert.tStartRefresh)
    skip.addData('foot_vert.stopped', foot_vert.tStopRefresh)
    # the Routine "up_down_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "left_right_practice"-------
    continueRoutine = True
    # update component parameters for each repeat
    curr_x = 0
    curr_y = 0
    
    all_locs_been_to = False
    locs_walked =[]
    
    no_keys_pressed = [0]
    nav_horz_key.clearEvents()
    nav_horz_key.keys = []
    nav_horz_key.rt = []
    
    msg = "Use your little finger to go left and your index finger to go right."
    isi = 2.5
    
    
    interval = 0.05
    move_timer = []
    move_timer.append(globalClock.getTime())
    move_counter = 0
    time_at_press = globalClock.getTime()
    key1_press_trigger = 0
    key2_press_trigger = 0
    key3_press_trigger = 0
    key4_press_trigger = 0
    loopno = 0
    nav_horz_key.keys = []
    nav_horz_key.rt = []
    _nav_horz_key_allKeys = []
    # keep track of which components have finished
    left_right_practiceComponents = [left_right_pract, foot_leftright, intro_leftright, feedback_leftright, nav_horz_key]
    for thisComponent in left_right_practiceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    left_right_practiceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "left_right_practice"-------
    while continueRoutine:
        # get current time
        t = left_right_practiceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=left_right_practiceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        loopno += 1
        
        # first check if you practiced enough.
        if all_locs_been_to == True:
            #wait(2)
            break
        
        isi = 2.5
        
        if (key1_press_trigger == 0) and (key4_press_trigger == 0):
            keys = nav_horz_key.getKeys(keyList = ['1', '4'], clear = False)
            no_keys_pressed.append(len(keys))
            
        if len(keys) > 0:
            # check if a new key has been pressed or if we are still updating the position.
            if (no_keys_pressed[-1] > no_keys_pressed[-2]) or (key1_press_trigger == 1) or (key4_press_trigger == 1):
                # check which key has been pressed
                if (keys[-1].name == '1') or (key1_press_trigger == 1):
                    if curr_x > -21/100:
                        key1_press_trigger = 1
                        if move_counter == 0:
                            last_x = curr_x
                            time_at_press = globalClock.getTime()
                            update_timer = time_at_press + interval
                            move_counter = 1
                        if move_timer[-1] < time_at_press+isi:
                            move_timer.append(globalClock.getTime())
                            if move_timer[-1] >= update_timer:
                                curr_x -= ((20/100)/(isi/interval))
                                update_timer += interval #increment by 100ms
                        elif move_timer[-1] >= time_at_press+isi:
                            curr_x = last_x - 21/100
                            move_counter = 0
                            key1_press_trigger = 0
                            direc = 'to the left'
                            # add the location to the location you walked.
                            locs_walked.append(curr_x)
                            msg = f"Well done! You walked {direc}. Take some more steps!"
                            # and save that you made a step.
                            thisExp.nextEntry()
                            thisExp.addData('curr_loc_x', curr_x)
                            thisExp.addData('curr_loc_y', curr_y)
                            thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                            thisExp.addData('t_step_tglobal', globalClock.getTime())
                            thisExp.addData('length_step', isi)
                # check which key had been pressed
                if (keys[-1].name == '4') or (key4_press_trigger == 1):
                    if curr_x < 21/100:
                        key4_press_trigger = 1
                        if move_counter == 0:
                            last_x = curr_x
                            time_at_press = globalClock.getTime()
                            update_timer = time_at_press + interval
                            move_counter = 1
                        if move_timer[-1] < time_at_press+isi:
                            move_timer.append(globalClock.getTime())
                            if move_timer[-1] >= update_timer:
                                curr_x += ((20/100)/(isi/interval))
                                update_timer += interval
                        elif move_timer[-1] >= time_at_press+isi:
                            curr_x = last_x + 21/100
                            move_counter = 0
                            key4_press_trigger = 0
                            direc = 'to the right'
                            # add the location to the location you walked.
                            locs_walked.append(curr_x)
                            msg = f"Well done! You walked {direc}. Take some more steps!"
                            # and save that you made a step.
                            thisExp.nextEntry()
                            thisExp.addData('curr_loc_x', curr_x)
                            thisExp.addData('curr_loc_y', curr_y)
                            thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                            thisExp.addData('t_step_tglobal', globalClock.getTime())
                            thisExp.addData('length_step', isi)
        
        # then check if you already walked along the whole line
        if no_keys_pressed[-1] > 10:
            print(f'you pressed {no_keys_pressed[-1]} keys.')
            all_locs_been_to = set([-21/100, 0, 21/100]).issubset(set(locs_walked))
            msg = f"Nice! You seem to be comfortable walking along a horizontal line."
        
        
        # *left_right_pract* updates
        if left_right_pract.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left_right_pract.frameNStart = frameN  # exact frame index
            left_right_pract.tStart = t  # local t and not account for scr refresh
            left_right_pract.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_right_pract, 'tStartRefresh')  # time at next scr refresh
            left_right_pract.setAutoDraw(True)
        
        # *foot_leftright* updates
        if foot_leftright.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            foot_leftright.frameNStart = frameN  # exact frame index
            foot_leftright.tStart = t  # local t and not account for scr refresh
            foot_leftright.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(foot_leftright, 'tStartRefresh')  # time at next scr refresh
            foot_leftright.setAutoDraw(True)
        if foot_leftright.status == STARTED:  # only update if drawing
            foot_leftright.setPos((curr_x, curr_y), log=False)
        
        # *intro_leftright* updates
        if intro_leftright.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_leftright.frameNStart = frameN  # exact frame index
            intro_leftright.tStart = t  # local t and not account for scr refresh
            intro_leftright.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_leftright, 'tStartRefresh')  # time at next scr refresh
            intro_leftright.setAutoDraw(True)
        if intro_leftright.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > intro_leftright.tStartRefresh + 4.5-frameTolerance:
                # keep track of stop time/frame for later
                intro_leftright.tStop = t  # not accounting for scr refresh
                intro_leftright.frameNStop = frameN  # exact frame index
                win.timeOnFlip(intro_leftright, 'tStopRefresh')  # time at next scr refresh
                intro_leftright.setAutoDraw(False)
        
        # *feedback_leftright* updates
        if feedback_leftright.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
            # keep track of start time/frame for later
            feedback_leftright.frameNStart = frameN  # exact frame index
            feedback_leftright.tStart = t  # local t and not account for scr refresh
            feedback_leftright.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_leftright, 'tStartRefresh')  # time at next scr refresh
            feedback_leftright.setAutoDraw(True)
        if feedback_leftright.status == STARTED:  # only update if drawing
            feedback_leftright.setText(msg, log=False)
        
        # *nav_horz_key* updates
        waitOnFlip = False
        if nav_horz_key.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
            # keep track of start time/frame for later
            nav_horz_key.frameNStart = frameN  # exact frame index
            nav_horz_key.tStart = t  # local t and not account for scr refresh
            nav_horz_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(nav_horz_key, 'tStartRefresh')  # time at next scr refresh
            nav_horz_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(nav_horz_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(nav_horz_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if nav_horz_key.status == STARTED and not waitOnFlip:
            theseKeys = nav_horz_key.getKeys(keyList=['1', '4'], waitRelease=False)
            _nav_horz_key_allKeys.extend(theseKeys)
            if len(_nav_horz_key_allKeys):
                nav_horz_key.keys = [key.name for key in _nav_horz_key_allKeys]  # storing all keys
                nav_horz_key.rt = [key.rt for key in _nav_horz_key_allKeys]
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in left_right_practiceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "left_right_practice"-------
    for thisComponent in left_right_practiceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    skip.addData('left_right_pract.started', left_right_pract.tStartRefresh)
    skip.addData('left_right_pract.stopped', left_right_pract.tStopRefresh)
    skip.addData('foot_leftright.started', foot_leftright.tStartRefresh)
    skip.addData('foot_leftright.stopped', foot_leftright.tStopRefresh)
    skip.addData('intro_leftright.started', intro_leftright.tStartRefresh)
    skip.addData('intro_leftright.stopped', intro_leftright.tStopRefresh)
    skip.addData('feedback_leftright.started', feedback_leftright.tStartRefresh)
    skip.addData('feedback_leftright.stopped', feedback_leftright.tStopRefresh)
    # check responses
    if nav_horz_key.keys in ['', [], None]:  # No response was made
        nav_horz_key.keys = None
    skip.addData('nav_horz_key.keys',nav_horz_key.keys)
    if nav_horz_key.keys != None:  # we had a response
        skip.addData('nav_horz_key.rt', nav_horz_key.rt)
    skip.addData('nav_horz_key.started', nav_horz_key.tStartRefresh)
    skip.addData('nav_horz_key.stopped', nav_horz_key.tStopRefresh)
    # the Routine "left_right_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "summary_intro"-------
    continueRoutine = True
    # update component parameters for each repeat
    next_summary_key.keys = []
    next_summary_key.rt = []
    _next_summary_key_allKeys = []
    # keep track of which components have finished
    summary_introComponents = [summary_parrot, next_summary_key]
    for thisComponent in summary_introComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    summary_introClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "summary_intro"-------
    while continueRoutine:
        # get current time
        t = summary_introClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=summary_introClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *summary_parrot* updates
        if summary_parrot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            summary_parrot.frameNStart = frameN  # exact frame index
            summary_parrot.tStart = t  # local t and not account for scr refresh
            summary_parrot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(summary_parrot, 'tStartRefresh')  # time at next scr refresh
            summary_parrot.setAutoDraw(True)
        
        # *next_summary_key* updates
        waitOnFlip = False
        if next_summary_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            next_summary_key.frameNStart = frameN  # exact frame index
            next_summary_key.tStart = t  # local t and not account for scr refresh
            next_summary_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_summary_key, 'tStartRefresh')  # time at next scr refresh
            next_summary_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(next_summary_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(next_summary_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if next_summary_key.status == STARTED and not waitOnFlip:
            theseKeys = next_summary_key.getKeys(keyList=['1'], waitRelease=False)
            _next_summary_key_allKeys.extend(theseKeys)
            if len(_next_summary_key_allKeys):
                next_summary_key.keys = _next_summary_key_allKeys[-1].name  # just the last key pressed
                next_summary_key.rt = _next_summary_key_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in summary_introComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "summary_intro"-------
    for thisComponent in summary_introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    skip.addData('summary_parrot.started', summary_parrot.tStartRefresh)
    skip.addData('summary_parrot.stopped', summary_parrot.tStopRefresh)
    # check responses
    if next_summary_key.keys in ['', [], None]:  # No response was made
        next_summary_key.keys = None
    skip.addData('next_summary_key.keys',next_summary_key.keys)
    if next_summary_key.keys != None:  # we had a response
        skip.addData('next_summary_key.rt', next_summary_key.rt)
    skip.addData('next_summary_key.started', next_summary_key.tStartRefresh)
    skip.addData('next_summary_key.stopped', next_summary_key.tStopRefresh)
    # the Routine "summary_intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 0.0 repeats of 'skip'


# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('pract_cond_3x3_debug.xlsx'),
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
    
    # ------Prepare to start Routine "new_rewards_prep"-------
    continueRoutine = True
    routineTimer.add(4.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    new_rewards_prepComponents = [new_rewards]
    for thisComponent in new_rewards_prepComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    new_rewards_prepClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "new_rewards_prep"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = new_rewards_prepClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=new_rewards_prepClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *new_rewards* updates
        if new_rewards.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            new_rewards.frameNStart = frameN  # exact frame index
            new_rewards.tStart = t  # local t and not account for scr refresh
            new_rewards.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(new_rewards, 'tStartRefresh')  # time at next scr refresh
            new_rewards.setAutoDraw(True)
        if new_rewards.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > new_rewards.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                new_rewards.tStop = t  # not accounting for scr refresh
                new_rewards.frameNStop = frameN  # exact frame index
                win.timeOnFlip(new_rewards, 'tStopRefresh')  # time at next scr refresh
                new_rewards.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in new_rewards_prepComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "new_rewards_prep"-------
    for thisComponent in new_rewards_prepComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('new_rewards.started', new_rewards.tStartRefresh)
    trials.addData('new_rewards.stopped', new_rewards.tStopRefresh)
    
    # ------Prepare to start Routine "parrot_tip"-------
    continueRoutine = True
    routineTimer.add(4.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    parrot_tipComponents = [parrot]
    for thisComponent in parrot_tipComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    parrot_tipClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "parrot_tip"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = parrot_tipClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=parrot_tipClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *parrot* updates
        if parrot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            parrot.frameNStart = frameN  # exact frame index
            parrot.tStart = t  # local t and not account for scr refresh
            parrot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(parrot, 'tStartRefresh')  # time at next scr refresh
            parrot.setAutoDraw(True)
        if parrot.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > parrot.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                parrot.tStop = t  # not accounting for scr refresh
                parrot.frameNStop = frameN  # exact frame index
                win.timeOnFlip(parrot, 'tStopRefresh')  # time at next scr refresh
                parrot.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in parrot_tipComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "parrot_tip"-------
    for thisComponent in parrot_tipComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('parrot.started', parrot.tStartRefresh)
    trials.addData('parrot.stopped', parrot.tStopRefresh)
    
    # ------Prepare to start Routine "show_rewards"-------
    continueRoutine = True
    routineTimer.add(10.000000)
    # update component parameters for each repeat
    time_start = globalClock.getTime()
    
    op_rewB = 0
    op_rewA = 0
    op_rewC = 0
    op_rewD = 0
    reward_A.setPos((rew_x_A, rew_y_A))
    reward_B.setPos((rew_x_B, rew_y_B))
    reward_C.setPos((rew_x_C, rew_y_C))
    reward_D.setPos((rew_x_D, rew_y_D))
    # keep track of which components have finished
    show_rewardsComponents = [sand_pirate, reward_A, reward_B, reward_C, reward_D]
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
    show_rewardsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "show_rewards"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = show_rewardsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=show_rewardsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
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
        if sand_pirate.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sand_pirate.frameNStart = frameN  # exact frame index
            sand_pirate.tStart = t  # local t and not account for scr refresh
            sand_pirate.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sand_pirate, 'tStartRefresh')  # time at next scr refresh
            sand_pirate.setAutoDraw(True)
        if sand_pirate.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sand_pirate.tStartRefresh + 10.-frameTolerance:
                # keep track of stop time/frame for later
                sand_pirate.tStop = t  # not accounting for scr refresh
                sand_pirate.frameNStop = frameN  # exact frame index
                win.timeOnFlip(sand_pirate, 'tStopRefresh')  # time at next scr refresh
                sand_pirate.setAutoDraw(False)
        
        # *reward_A* updates
        if reward_A.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_A.frameNStart = frameN  # exact frame index
            reward_A.tStart = t  # local t and not account for scr refresh
            reward_A.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_A, 'tStartRefresh')  # time at next scr refresh
            reward_A.setAutoDraw(True)
        if reward_A.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_A.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_A.tStop = t  # not accounting for scr refresh
                reward_A.frameNStop = frameN  # exact frame index
                win.timeOnFlip(reward_A, 'tStopRefresh')  # time at next scr refresh
                reward_A.setAutoDraw(False)
        if reward_A.status == STARTED:  # only update if drawing
            reward_A.setOpacity(op_rewA, log=False)
        
        # *reward_B* updates
        if reward_B.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_B.frameNStart = frameN  # exact frame index
            reward_B.tStart = t  # local t and not account for scr refresh
            reward_B.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_B, 'tStartRefresh')  # time at next scr refresh
            reward_B.setAutoDraw(True)
        if reward_B.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_B.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_B.tStop = t  # not accounting for scr refresh
                reward_B.frameNStop = frameN  # exact frame index
                win.timeOnFlip(reward_B, 'tStopRefresh')  # time at next scr refresh
                reward_B.setAutoDraw(False)
        if reward_B.status == STARTED:  # only update if drawing
            reward_B.setOpacity(op_rewB, log=False)
        
        # *reward_C* updates
        if reward_C.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_C.frameNStart = frameN  # exact frame index
            reward_C.tStart = t  # local t and not account for scr refresh
            reward_C.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_C, 'tStartRefresh')  # time at next scr refresh
            reward_C.setAutoDraw(True)
        if reward_C.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_C.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_C.tStop = t  # not accounting for scr refresh
                reward_C.frameNStop = frameN  # exact frame index
                win.timeOnFlip(reward_C, 'tStopRefresh')  # time at next scr refresh
                reward_C.setAutoDraw(False)
        if reward_C.status == STARTED:  # only update if drawing
            reward_C.setOpacity(op_rewC, log=False)
        
        # *reward_D* updates
        if reward_D.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            reward_D.frameNStart = frameN  # exact frame index
            reward_D.tStart = t  # local t and not account for scr refresh
            reward_D.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(reward_D, 'tStartRefresh')  # time at next scr refresh
            reward_D.setAutoDraw(True)
        if reward_D.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > reward_D.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                reward_D.tStop = t  # not accounting for scr refresh
                reward_D.frameNStop = frameN  # exact frame index
                win.timeOnFlip(reward_D, 'tStopRefresh')  # time at next scr refresh
                reward_D.setAutoDraw(False)
        if reward_D.status == STARTED:  # only update if drawing
            reward_D.setOpacity(op_rewD, log=False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in show_rewardsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "show_rewards"-------
    for thisComponent in show_rewardsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('sand_pirate.started', sand_pirate.tStartRefresh)
    trials.addData('sand_pirate.stopped', sand_pirate.tStopRefresh)
    trials.addData('reward_A.started', reward_A.tStartRefresh)
    trials.addData('reward_A.stopped', reward_A.tStopRefresh)
    trials.addData('reward_B.started', reward_B.tStartRefresh)
    trials.addData('reward_B.stopped', reward_B.tStopRefresh)
    trials.addData('reward_C.started', reward_C.tStartRefresh)
    trials.addData('reward_C.stopped', reward_C.tStopRefresh)
    trials.addData('reward_D.started', reward_D.tStartRefresh)
    trials.addData('reward_D.stopped', reward_D.tStopRefresh)
    
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
        
        # ------Prepare to start Routine "task"-------
        continueRoutine = True
        # update component parameters for each repeat
        rew_visible = 0
        index_rew = [0,0]
        no_keys_pressed = [0]
        
        # flush out all potentially recorded button press events
        nav_key_task.clearEvents()
        loop_no = 0
        
        rew_x = [rew_x_A, rew_x_B, rew_x_C, rew_x_D]
        rew_y = [rew_y_A, rew_y_B, rew_y_C, rew_y_D]
        
        #rew_x = [75/100, 75/100, -75/100, 75/100]
        #rew_y = [75/100, -75/100, -25/100, -25/100]
        states = ['A', 'B', 'C', 'D']
        field_nos = ['','','','']
        msg = ''
        
        thisExp.nextEntry()
        thisExp.addData('start_ABCD_game', globalClock.getTime())
        states = ['A', 'B', 'C', 'D']
        field_nos = ['','','','']
        msg = ''
        
        
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
        reward_waiting = random.uniform(2.2, 3.5)
        
        nav_key_task.keys = []
        nav_key_task.rt = []
        _nav_key_task_allKeys = []
        break_key.keys = []
        break_key.rt = []
        _break_key_allKeys = []
        # keep track of which components have finished
        taskComponents = [sand_box, foot, reward, nav_key_task, polygon, reward_progress, plus_coin_txt, break_key]
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
        taskClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "task"-------
        while continueRoutine:
            # get current time
            t = taskClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=taskClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            loop_no += 1
            #TRs = TR_count.getKeys(keyList = ['5'], clear = False)
            #if len(TRs) > 0:
            #    print(TRs[-1].name)
            
            # first check if you have found the final reward.
            if index_rew[-1] == len(states):
                reward_waiting = random.uniform(2.2, 3.5)
                print('now it should continue')
                #wait(reward_waiting)
                break 
            
            # activate these only if just starting or if a step has been made
            if loop_no == 1: 
                thisExp.nextEntry()
                thisExp.addData('start_finding_rewards', globalClock.getTime())
                #if len(TRs) > 0:
                    #thisExp.nextEntry()
                    #thisExp.addData('first_task_TR', TRs[0].rt)
                    #print(TRs.rt[0])
                curr_state = states[index_rew[-1]]
                curr_rew_x = rew_x[index_rew[-1]] 
                curr_rew_y = rew_y[index_rew[-1]]
            
            # change this back later!!
            isi = random.uniform(1.5, 2.5)
            # iti = random.uniform(0.1, 0.5)
            
            
            if (key1_press_trigger == 0) and (key2_press_trigger == 0) and (key3_press_trigger == 0) and (key4_press_trigger == 0):
                keys = nav_key_task.getKeys(keyList = ['1', '2', '3', '4'], clear = False)
                no_keys_pressed.append(len(keys))
            
            
            if len(keys) > 0:
                # first check if you last found a reward. this means that you can't 
                # react to key presses for a bit.
                if index_rew[-1] > index_rew[-2]:
                    # enter loop where you framewise increase the progress_height 
                    # until the reward_wait time is elapsed.
                    if progress_bar_on == 0:
                        reward_waiting = random.uniform(2.2, 3.5)
                        last_height = height_progress
                        time_rew_found = globalClock.getTime()
                        progress_update = time_rew_found + interval
                        progress_bar_on = 1
                    if progress_timer[-1] < time_at_press+reward_waiting:
                        progress_timer.append(globalClock.getTime())
                        if progress_timer[-1] >= progress_update:
                            #it can be max 0.43 long. I have 3 repeats and 4 rews. > increase by 0.035 each rew
                            # I now have variable repeats. reps_per_run 
                            height_progress += ((0.43/(4*reps_per_run))/(reward_waiting/interval))
                            progress_update += interval #increment by 100ms
                    elif progress_timer[-1] >= time_at_press+reward_waiting:
                        progress_bar_on = 0
                        print('done progress bar update')
                        # and save that you made a step.
                        thisExp.nextEntry()
                        thisExp.addData('t_reward_afterwait', globalClock.getTime())
                        # and make it invisible again
                        rew_visible = 0
                        # and add the same last value, so it wouldn't enter this loop again.
                        index_rew.append(index_rew[-1])
                        # and update the reward!! now off to find the next one.
                        curr_state = states[index_rew[-1]]
                        curr_rew_x = rew_x[index_rew[-1]] 
                        curr_rew_y = rew_y[index_rew[-1]]
                        print(f'Reward found and waited! These should all be 0: key 1:{key1_press_trigger} key 2:{key2_press_trigger} key3:{key3_press_trigger} key 4:{key4_press_trigger}')
                        print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                # if you didnt last found a reward, you can check for key presses.   
                # then check if a new key has been pressed or if we are still updating the position.
                elif (no_keys_pressed[-1] > no_keys_pressed[-2]) or (key1_press_trigger == 1) or (key2_press_trigger == 1) or (key3_press_trigger == 1) or (key4_press_trigger == 1):
                    # check which key has been pressed
                    if (keys[-1].name == '1') or (key1_press_trigger == 1):
                        if (curr_x> -21/100):
                            key1_press_trigger = 1
                            if move_counter == 0:
                                last_x = curr_x
                                time_at_press = globalClock.getTime()
                                update_timer = time_at_press + interval
                                move_counter = 1
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_x -= ((20/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                            elif move_timer[-1] >= time_at_press+isi:
                                curr_x = last_x - 21/100
                                move_counter = 0
                                key1_press_trigger = 0
                                print('done moving 1')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.nextEntry()
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                                thisExp.addData('t_step_tglobal', globalClock.getTime())
                                thisExp.addData('length_step', isi)
                    # check which key has been pressed
                    if (keys[-1].name == '2') or (key2_press_trigger == 1):
                        if curr_y < 29/100:
                            key2_press_trigger = 1
                            if move_counter == 0:
                                last_y = curr_y
                                time_at_press = globalClock.getTime()
                                update_timer = time_at_press + interval
                                move_counter = 1
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_y += ((28/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                            elif move_timer[-1] >= time_at_press+isi:
                                curr_y = last_y + 29/100
                                move_counter = 0
                                key2_press_trigger = 0
                                print('done moving 2')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.nextEntry()
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                                thisExp.addData('t_step_tglobal', globalClock.getTime())
                                thisExp.addData('length_step', isi)
                    # check which key had been pressed
                    if (keys[-1].name == '3') or (key3_press_trigger == 1):
                        if curr_y > -29/100:
                            key3_press_trigger = 1
                            if move_counter == 0:
                                last_y = curr_y
                                time_at_press = globalClock.getTime()
                                update_timer = time_at_press + interval
                                move_counter = 1
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_y -= ((28/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                            elif move_timer[-1] >= time_at_press+isi:
                                curr_y = last_y - 29/100
                                move_counter = 0
                                key3_press_trigger = 0
                                print('done moving 3')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.nextEntry()
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                                thisExp.addData('t_step_tglobal', globalClock.getTime())
                                thisExp.addData('length_step', isi)
                    # check which keys have been pressed
                    if (keys[-1].name == '4') or (key4_press_trigger == 1):
                        if curr_x < 21/100:
                            key4_press_trigger = 1
                            if move_counter == 0:
                                last_x = curr_x
                                time_at_press = globalClock.getTime()
                                update_timer = time_at_press + interval
                                move_counter = 1
                            if move_timer[-1] < time_at_press+isi:
                                move_timer.append(globalClock.getTime())
                                if move_timer[-1] >= update_timer:
                                    curr_x += ((20/100)/(isi/interval))
                                    update_timer += interval #increment by 100ms
                            elif move_timer[-1] >= time_at_press+isi:
                                curr_x = last_x + 21/100
                                move_counter = 0
                                key4_press_trigger = 0
                                print('done moving 4')
                                print(f'this is current x: {curr_x}, reward x: {curr_rew_x}, and current y: {curr_y}, reward y: {curr_rew_y}.')
                                # and save that you made a step.
                                thisExp.nextEntry()
                                thisExp.addData('curr_loc_x', curr_x)
                                thisExp.addData('curr_loc_y', curr_y)
                                thisExp.addData('t_step_from_start_currrun', keys[-1].rt)
                                thisExp.addData('t_step_tglobal', globalClock.getTime())
                                thisExp.addData('length_step', isi)
            
            
                    
                    # then check if reward location and curr loc are the same
                    if (curr_x == curr_rew_x) and (curr_y == curr_rew_y):
                        # go to next reward
                        index_rew.append(index_rew[-1]+1)
                        print('found reward')
                        # show the reward!
                        rew_visible = 1
                        thisExp.addData('rew_loc_x', curr_rew_x)
                        thisExp.addData('rew_loc_y', curr_rew_y)
                        thisExp.addData('t_reward_start', globalClock.getTime())
                        thisExp.addData('isi_reward', reward_waiting)
                        # write a function which increases the opacity of the reward 1 for
                        # [there will be 5 repeats * 4 rewards > 20]
                        # for the reward_waiting time, increase the reward by overall 1/20.
                        # make sure it doesnt end up always at 1
                        # this can probably be done with some sort of loop similar to the moving timer!
                        progress1 = 0
                 
                    # then let everything update all the shapes.
            
            # *sand_box* updates
            if sand_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sand_box.frameNStart = frameN  # exact frame index
                sand_box.tStart = t  # local t and not account for scr refresh
                sand_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sand_box, 'tStartRefresh')  # time at next scr refresh
                sand_box.setAutoDraw(True)
            
            # *foot* updates
            if foot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                foot.frameNStart = frameN  # exact frame index
                foot.tStart = t  # local t and not account for scr refresh
                foot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(foot, 'tStartRefresh')  # time at next scr refresh
                foot.setAutoDraw(True)
            if foot.status == STARTED:  # only update if drawing
                foot.setPos((curr_x, curr_y), log=False)
            
            # *reward* updates
            if reward.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reward.frameNStart = frameN  # exact frame index
                reward.tStart = t  # local t and not account for scr refresh
                reward.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reward, 'tStartRefresh')  # time at next scr refresh
                reward.setAutoDraw(True)
            if reward.status == STARTED:  # only update if drawing
                reward.setOpacity(rew_visible, log=False)
                reward.setPos((curr_rew_x, curr_rew_y), log=False)
            
            # *nav_key_task* updates
            waitOnFlip = False
            if nav_key_task.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nav_key_task.frameNStart = frameN  # exact frame index
                nav_key_task.tStart = t  # local t and not account for scr refresh
                nav_key_task.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nav_key_task, 'tStartRefresh')  # time at next scr refresh
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
            
            # *polygon* updates
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                polygon.setAutoDraw(True)
            if polygon.status == STARTED:  # only update if drawing
                polygon.setSize((0.1, height_progress), log=False)
            
            # *reward_progress* updates
            if reward_progress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reward_progress.frameNStart = frameN  # exact frame index
                reward_progress.tStart = t  # local t and not account for scr refresh
                reward_progress.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reward_progress, 'tStartRefresh')  # time at next scr refresh
                reward_progress.setAutoDraw(True)
            if reward_progress.status == STARTED:  # only update if drawing
                reward_progress.setOpacity(rew_visible, log=False)
            
            # *plus_coin_txt* updates
            if plus_coin_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                plus_coin_txt.frameNStart = frameN  # exact frame index
                plus_coin_txt.tStart = t  # local t and not account for scr refresh
                plus_coin_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(plus_coin_txt, 'tStartRefresh')  # time at next scr refresh
                plus_coin_txt.setAutoDraw(True)
            if plus_coin_txt.status == STARTED:  # only update if drawing
                plus_coin_txt.setOpacity(rew_visible, log=False)
            
            # *break_key* updates
            waitOnFlip = False
            if break_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_key.frameNStart = frameN  # exact frame index
                break_key.tStart = t  # local t and not account for scr refresh
                break_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_key, 'tStartRefresh')  # time at next scr refresh
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
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in taskComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "task"-------
        for thisComponent in taskComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        rep_runs.addData('sand_box.started', sand_box.tStartRefresh)
        rep_runs.addData('sand_box.stopped', sand_box.tStopRefresh)
        rep_runs.addData('foot.started', foot.tStartRefresh)
        rep_runs.addData('foot.stopped', foot.tStopRefresh)
        rep_runs.addData('reward.started', reward.tStartRefresh)
        rep_runs.addData('reward.stopped', reward.tStopRefresh)
        # check responses
        if nav_key_task.keys in ['', [], None]:  # No response was made
            nav_key_task.keys = None
        rep_runs.addData('nav_key_task.keys',nav_key_task.keys)
        if nav_key_task.keys != None:  # we had a response
            rep_runs.addData('nav_key_task.rt', nav_key_task.rt)
        rep_runs.addData('nav_key_task.started', nav_key_task.tStartRefresh)
        rep_runs.addData('nav_key_task.stopped', nav_key_task.tStopRefresh)
        rep_runs.addData('polygon.started', polygon.tStartRefresh)
        rep_runs.addData('polygon.stopped', polygon.tStopRefresh)
        rep_runs.addData('reward_progress.started', reward_progress.tStartRefresh)
        rep_runs.addData('reward_progress.stopped', reward_progress.tStopRefresh)
        rep_runs.addData('plus_coin_txt.started', plus_coin_txt.tStartRefresh)
        rep_runs.addData('plus_coin_txt.stopped', plus_coin_txt.tStopRefresh)
        # check responses
        if break_key.keys in ['', [], None]:  # No response was made
            break_key.keys = None
        rep_runs.addData('break_key.keys',break_key.keys)
        if break_key.keys != None:  # we had a response
            rep_runs.addData('break_key.rt', break_key.rt)
        rep_runs.addData('break_key.started', break_key.tStartRefresh)
        rep_runs.addData('break_key.stopped', break_key.tStopRefresh)
        # the Routine "task" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed reps_per_run repeats of 'rep_runs'
    
    
    # ------Prepare to start Routine "feedback_screen"-------
    continueRoutine = True
    routineTimer.add(5.000000)
    # update component parameters for each repeat
    feedback_msg= f'You were better than {feedback}% of the candidates. Try to be even more precise and walk only the same routes to impress me!'
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
    feedback_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "feedback_screen"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = feedback_screenClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=feedback_screenClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background_feedback* updates
        if background_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background_feedback.frameNStart = frameN  # exact frame index
            background_feedback.tStart = t  # local t and not account for scr refresh
            background_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background_feedback, 'tStartRefresh')  # time at next scr refresh
            background_feedback.setAutoDraw(True)
        if background_feedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > background_feedback.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                background_feedback.tStop = t  # not accounting for scr refresh
                background_feedback.frameNStop = frameN  # exact frame index
                win.timeOnFlip(background_feedback, 'tStopRefresh')  # time at next scr refresh
                background_feedback.setAutoDraw(False)
        
        # *feedback_text* updates
        if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            feedback_text.frameNStart = frameN  # exact frame index
            feedback_text.tStart = t  # local t and not account for scr refresh
            feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
            feedback_text.setAutoDraw(True)
        if feedback_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > feedback_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                feedback_text.tStop = t  # not accounting for scr refresh
                feedback_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(feedback_text, 'tStopRefresh')  # time at next scr refresh
                feedback_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedback_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "feedback_screen"-------
    for thisComponent in feedback_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('background_feedback.started', background_feedback.tStartRefresh)
    trials.addData('background_feedback.stopped', background_feedback.tStopRefresh)
    trials.addData('feedback_text.started', feedback_text.tStartRefresh)
    trials.addData('feedback_text.stopped', feedback_text.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'trials'


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
