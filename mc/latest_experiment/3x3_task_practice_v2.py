#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.1.3),
    on Fri Nov 17 16:02:15 2023
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
expName = '3x3_task_practice_v2'  # from the Builder filename that created this script
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
    originPath='/Volumes/NO NAME/latest_experiment/3x3_task_practice_v2.py',
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
    size=[1920, 1440], fullscr=False, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='dell_compneuro_laptop', color=[0,0,0], colorSpace='rgb',
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

# --- Initialize components for Routine "intro_0" ---
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

# --- Initialize components for Routine "intro_1" ---
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

# --- Initialize components for Routine "intro_1_1" ---
intro1_1_ship = visual.ImageStim(
    win=win,
    name='intro1_1_ship', 
    image='images/ship_box.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
intro1_1_txt = visual.TextStim(win=win, name='intro1_1_txt',
    text='Most of the times, you will have to find the gold coins in the order they are presented.\nBut careful! To properly test you, the pirate captain will sometimes tell you to find the rewards in revsered order.\nThe most important part is to clearly visualise the paths you want to walk, and to always walk the same paths as long as the locations stay the same.\n\nPress 1 to continue!',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
next_intro_1_1 = keyboard.Keyboard()

# --- Initialize components for Routine "intro_2" ---
treasure_box_intro_2 = visual.ImageStim(
    win=win,
    name='treasure_box_intro_2', 
    image='images/map_w_txt.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
text_intro_2 = visual.TextStim(win=win, name='text_intro_2',
    text='To navigate through the sand, you can use the keys 1, 2, 3, and 4.\nPlease use your right hand so that your index finger (1) will take you left, your middle finger (2) moves you north, your ring finger (3) leads you south, and your little finger (4) guides you right.\nUnfortunately, walking through sand can take quite long and is effortful...\n\nPress 1 to start the practice!',
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

# --- Initialize components for Routine "up_down_practice" ---
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

# --- Initialize components for Routine "left_right_practice" ---
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

# --- Initialize components for Routine "summary_intro" ---
summary_parrot = visual.ImageStim(
    win=win,
    name='summary_parrot', 
    image='images/summary_parrot.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
next_summary_key = keyboard.Keyboard()

# --- Initialize components for Routine "new_rewards_prep" ---
new_rewards = visual.ImageStim(
    win=win,
    name='new_rewards', 
    image='images/sand_3x3grid_pirate_new.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)

# --- Initialize components for Routine "parrot_tip" ---
parrot = visual.ImageStim(
    win=win,
    name='parrot', 
    image='images/parrot_tip.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
backw_warning_parrot = visual.TextStim(win=win, name='backw_warning_parrot',
    text='Careful! This is a backwards trial!',
    font='Open Sans',
    pos=(0.4, 0), height=0.1, wrapWidth=0.6, ori=0.0, 
    color='black', colorSpace='rgb', opacity=1.0, 
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
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
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
    opacity=1.0, depth=-6.0, interpolate=True)
progress_bar = visual.Rect(
    win=win, name='progress_bar',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=(-0.5, -0.21), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=None, fillColor=[0.7098, 0.2941, -0.7490],
    opacity=1.0, depth=-7.0, interpolate=True)
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

# --- Initialize components for Routine "the_end" ---
full_ship_end = visual.ImageStim(
    win=win,
    name='full_ship_end', 
    image='images/pirates.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
ship_box = visual.ImageStim(
    win=win,
    name='ship_box', 
    image='images/ship_box.png', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(2,2),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
end_practice_key = keyboard.Keyboard()
outro = visual.TextStim(win=win, name='outro',
    text="Arr, ye be a fine pirate, ye scallywag! \nIn the MRI, it be crucial to be swift as a cutlass and accurate as a marksman. Always visualise the paths you want to take.\nThe better you perform, the higher your payment in the end!\nJust one more hour, and nothin' shall stand in yer way of joinin' our crew!\n\n\nEnd with 1.",
    font='Open Sans',
    pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "intro_0" ---
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
frameN = -1

# --- Run Routine "intro_0" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *ship_intro_0* updates
    
    # if ship_intro_0 is starting this frame...
    if ship_intro_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        ship_intro_0.frameNStart = frameN  # exact frame index
        ship_intro_0.tStart = t  # local t and not account for scr refresh
        ship_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(ship_intro_0, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'ship_intro_0.started')
        # update status
        ship_intro_0.status = STARTED
        ship_intro_0.setAutoDraw(True)
    
    # if ship_intro_0 is active this frame...
    if ship_intro_0.status == STARTED:
        # update params
        pass
    
    # if ship_intro_0 is stopping this frame...
    if ship_intro_0.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > ship_intro_0.tStartRefresh + 2.5-frameTolerance:
            # keep track of stop time/frame for later
            ship_intro_0.tStop = t  # not accounting for scr refresh
            ship_intro_0.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ship_intro_0.stopped')
            # update status
            ship_intro_0.status = FINISHED
            ship_intro_0.setAutoDraw(False)
    
    # *text* updates
    
    # if text is starting this frame...
    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text.started')
        # update status
        text.status = STARTED
        text.setAutoDraw(True)
    
    # if text is active this frame...
    if text.status == STARTED:
        # update params
        pass
    
    # if text is stopping this frame...
    if text.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text.tStartRefresh + 2.5-frameTolerance:
            # keep track of stop time/frame for later
            text.tStop = t  # not accounting for scr refresh
            text.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.stopped')
            # update status
            text.status = FINISHED
            text.setAutoDraw(False)
    
    # *ship_box_intro_0* updates
    
    # if ship_box_intro_0 is starting this frame...
    if ship_box_intro_0.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
        # keep track of start time/frame for later
        ship_box_intro_0.frameNStart = frameN  # exact frame index
        ship_box_intro_0.tStart = t  # local t and not account for scr refresh
        ship_box_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(ship_box_intro_0, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'ship_box_intro_0.started')
        # update status
        ship_box_intro_0.status = STARTED
        ship_box_intro_0.setAutoDraw(True)
    
    # if ship_box_intro_0 is active this frame...
    if ship_box_intro_0.status == STARTED:
        # update params
        pass
    
    # *text_intro_0* updates
    
    # if text_intro_0 is starting this frame...
    if text_intro_0.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
        # keep track of start time/frame for later
        text_intro_0.frameNStart = frameN  # exact frame index
        text_intro_0.tStart = t  # local t and not account for scr refresh
        text_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_intro_0, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_intro_0.started')
        # update status
        text_intro_0.status = STARTED
        text_intro_0.setAutoDraw(True)
    
    # if text_intro_0 is active this frame...
    if text_intro_0.status == STARTED:
        # update params
        pass
    
    # *next_intro_0* updates
    waitOnFlip = False
    
    # if next_intro_0 is starting this frame...
    if next_intro_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        next_intro_0.frameNStart = frameN  # exact frame index
        next_intro_0.tStart = t  # local t and not account for scr refresh
        next_intro_0.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(next_intro_0, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'next_intro_0.started')
        # update status
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
            next_intro_0.duration = _next_intro_0_allKeys[-1].duration
            # a response ends the routine
            continueRoutine = False
    
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
    for thisComponent in intro_0Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "intro_0" ---
for thisComponent in intro_0Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if next_intro_0.keys in ['', [], None]:  # No response was made
    next_intro_0.keys = None
thisExp.addData('next_intro_0.keys',next_intro_0.keys)
if next_intro_0.keys != None:  # we had a response
    thisExp.addData('next_intro_0.rt', next_intro_0.rt)
    thisExp.addData('next_intro_0.duration', next_intro_0.duration)
thisExp.nextEntry()
# the Routine "intro_0" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "intro_1" ---
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
frameN = -1

# --- Run Routine "intro_1" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *ship_box_intro_1* updates
    
    # if ship_box_intro_1 is starting this frame...
    if ship_box_intro_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        ship_box_intro_1.frameNStart = frameN  # exact frame index
        ship_box_intro_1.tStart = t  # local t and not account for scr refresh
        ship_box_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(ship_box_intro_1, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'ship_box_intro_1.started')
        # update status
        ship_box_intro_1.status = STARTED
        ship_box_intro_1.setAutoDraw(True)
    
    # if ship_box_intro_1 is active this frame...
    if ship_box_intro_1.status == STARTED:
        # update params
        pass
    
    # *text_intro_1* updates
    
    # if text_intro_1 is starting this frame...
    if text_intro_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_intro_1.frameNStart = frameN  # exact frame index
        text_intro_1.tStart = t  # local t and not account for scr refresh
        text_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_intro_1, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_intro_1.started')
        # update status
        text_intro_1.status = STARTED
        text_intro_1.setAutoDraw(True)
    
    # if text_intro_1 is active this frame...
    if text_intro_1.status == STARTED:
        # update params
        pass
    
    # *next_intro_1* updates
    waitOnFlip = False
    
    # if next_intro_1 is starting this frame...
    if next_intro_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        next_intro_1.frameNStart = frameN  # exact frame index
        next_intro_1.tStart = t  # local t and not account for scr refresh
        next_intro_1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(next_intro_1, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'next_intro_1.started')
        # update status
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
            next_intro_1.duration = _next_intro_1_allKeys[-1].duration
            # a response ends the routine
            continueRoutine = False
    
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
    for thisComponent in intro_1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "intro_1" ---
for thisComponent in intro_1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if next_intro_1.keys in ['', [], None]:  # No response was made
    next_intro_1.keys = None
thisExp.addData('next_intro_1.keys',next_intro_1.keys)
if next_intro_1.keys != None:  # we had a response
    thisExp.addData('next_intro_1.rt', next_intro_1.rt)
    thisExp.addData('next_intro_1.duration', next_intro_1.duration)
thisExp.nextEntry()
# the Routine "intro_1" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "intro_1_1" ---
continueRoutine = True
# update component parameters for each repeat
next_intro_1_1.keys = []
next_intro_1_1.rt = []
_next_intro_1_1_allKeys = []
# keep track of which components have finished
intro_1_1Components = [intro1_1_ship, intro1_1_txt, next_intro_1_1]
for thisComponent in intro_1_1Components:
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

# --- Run Routine "intro_1_1" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *intro1_1_ship* updates
    
    # if intro1_1_ship is starting this frame...
    if intro1_1_ship.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        intro1_1_ship.frameNStart = frameN  # exact frame index
        intro1_1_ship.tStart = t  # local t and not account for scr refresh
        intro1_1_ship.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(intro1_1_ship, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'intro1_1_ship.started')
        # update status
        intro1_1_ship.status = STARTED
        intro1_1_ship.setAutoDraw(True)
    
    # if intro1_1_ship is active this frame...
    if intro1_1_ship.status == STARTED:
        # update params
        pass
    
    # *intro1_1_txt* updates
    
    # if intro1_1_txt is starting this frame...
    if intro1_1_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        intro1_1_txt.frameNStart = frameN  # exact frame index
        intro1_1_txt.tStart = t  # local t and not account for scr refresh
        intro1_1_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(intro1_1_txt, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'intro1_1_txt.started')
        # update status
        intro1_1_txt.status = STARTED
        intro1_1_txt.setAutoDraw(True)
    
    # if intro1_1_txt is active this frame...
    if intro1_1_txt.status == STARTED:
        # update params
        pass
    
    # *next_intro_1_1* updates
    waitOnFlip = False
    
    # if next_intro_1_1 is starting this frame...
    if next_intro_1_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        next_intro_1_1.frameNStart = frameN  # exact frame index
        next_intro_1_1.tStart = t  # local t and not account for scr refresh
        next_intro_1_1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(next_intro_1_1, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'next_intro_1_1.started')
        # update status
        next_intro_1_1.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(next_intro_1_1.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(next_intro_1_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if next_intro_1_1.status == STARTED and not waitOnFlip:
        theseKeys = next_intro_1_1.getKeys(keyList=['1'], waitRelease=False)
        _next_intro_1_1_allKeys.extend(theseKeys)
        if len(_next_intro_1_1_allKeys):
            next_intro_1_1.keys = _next_intro_1_1_allKeys[-1].name  # just the last key pressed
            next_intro_1_1.rt = _next_intro_1_1_allKeys[-1].rt
            next_intro_1_1.duration = _next_intro_1_1_allKeys[-1].duration
            # a response ends the routine
            continueRoutine = False
    
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
    for thisComponent in intro_1_1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "intro_1_1" ---
for thisComponent in intro_1_1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if next_intro_1_1.keys in ['', [], None]:  # No response was made
    next_intro_1_1.keys = None
thisExp.addData('next_intro_1_1.keys',next_intro_1_1.keys)
if next_intro_1_1.keys != None:  # we had a response
    thisExp.addData('next_intro_1_1.rt', next_intro_1_1.rt)
    thisExp.addData('next_intro_1_1.duration', next_intro_1_1.duration)
thisExp.nextEntry()
# the Routine "intro_1_1" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "intro_2" ---
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
frameN = -1

# --- Run Routine "intro_2" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *treasure_box_intro_2* updates
    
    # if treasure_box_intro_2 is starting this frame...
    if treasure_box_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        treasure_box_intro_2.frameNStart = frameN  # exact frame index
        treasure_box_intro_2.tStart = t  # local t and not account for scr refresh
        treasure_box_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(treasure_box_intro_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'treasure_box_intro_2.started')
        # update status
        treasure_box_intro_2.status = STARTED
        treasure_box_intro_2.setAutoDraw(True)
    
    # if treasure_box_intro_2 is active this frame...
    if treasure_box_intro_2.status == STARTED:
        # update params
        pass
    
    # *text_intro_2* updates
    
    # if text_intro_2 is starting this frame...
    if text_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_intro_2.frameNStart = frameN  # exact frame index
        text_intro_2.tStart = t  # local t and not account for scr refresh
        text_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_intro_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text_intro_2.started')
        # update status
        text_intro_2.status = STARTED
        text_intro_2.setAutoDraw(True)
    
    # if text_intro_2 is active this frame...
    if text_intro_2.status == STARTED:
        # update params
        pass
    
    # *next_intro_2* updates
    waitOnFlip = False
    
    # if next_intro_2 is starting this frame...
    if next_intro_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        next_intro_2.frameNStart = frameN  # exact frame index
        next_intro_2.tStart = t  # local t and not account for scr refresh
        next_intro_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(next_intro_2, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'next_intro_2.started')
        # update status
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
            next_intro_2.duration = _next_intro_2_allKeys[-1].duration
            # a response ends the routine
            continueRoutine = False
    
    # *navigation_img* updates
    
    # if navigation_img is starting this frame...
    if navigation_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        navigation_img.frameNStart = frameN  # exact frame index
        navigation_img.tStart = t  # local t and not account for scr refresh
        navigation_img.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(navigation_img, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'navigation_img.started')
        # update status
        navigation_img.status = STARTED
        navigation_img.setAutoDraw(True)
    
    # if navigation_img is active this frame...
    if navigation_img.status == STARTED:
        # update params
        pass
    
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
    for thisComponent in intro_2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "intro_2" ---
for thisComponent in intro_2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if next_intro_2.keys in ['', [], None]:  # No response was made
    next_intro_2.keys = None
thisExp.addData('next_intro_2.keys',next_intro_2.keys)
if next_intro_2.keys != None:  # we had a response
    thisExp.addData('next_intro_2.rt', next_intro_2.rt)
    thisExp.addData('next_intro_2.duration', next_intro_2.duration)
thisExp.nextEntry()
# the Routine "intro_2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "up_down_practice" ---
continueRoutine = True
# update component parameters for each repeat
# Run 'Begin Routine' code from code_updown
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
frameN = -1

# --- Run Routine "up_down_practice" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # Run 'Each Frame' code from code_updown
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
    
    # if parrot_updown_empty is starting this frame...
    if parrot_updown_empty.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        parrot_updown_empty.frameNStart = frameN  # exact frame index
        parrot_updown_empty.tStart = t  # local t and not account for scr refresh
        parrot_updown_empty.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(parrot_updown_empty, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'parrot_updown_empty.started')
        # update status
        parrot_updown_empty.status = STARTED
        parrot_updown_empty.setAutoDraw(True)
    
    # if parrot_updown_empty is active this frame...
    if parrot_updown_empty.status == STARTED:
        # update params
        pass
    
    # if parrot_updown_empty is stopping this frame...
    if parrot_updown_empty.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > parrot_updown_empty.tStartRefresh + 3.5-frameTolerance:
            # keep track of stop time/frame for later
            parrot_updown_empty.tStop = t  # not accounting for scr refresh
            parrot_updown_empty.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'parrot_updown_empty.stopped')
            # update status
            parrot_updown_empty.status = FINISHED
            parrot_updown_empty.setAutoDraw(False)
    
    # *help_txt_updown* updates
    
    # if help_txt_updown is starting this frame...
    if help_txt_updown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        help_txt_updown.frameNStart = frameN  # exact frame index
        help_txt_updown.tStart = t  # local t and not account for scr refresh
        help_txt_updown.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(help_txt_updown, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'help_txt_updown.started')
        # update status
        help_txt_updown.status = STARTED
        help_txt_updown.setAutoDraw(True)
    
    # if help_txt_updown is active this frame...
    if help_txt_updown.status == STARTED:
        # update params
        pass
    
    # if help_txt_updown is stopping this frame...
    if help_txt_updown.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > help_txt_updown.tStartRefresh + 3.5-frameTolerance:
            # keep track of stop time/frame for later
            help_txt_updown.tStop = t  # not accounting for scr refresh
            help_txt_updown.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'help_txt_updown.stopped')
            # update status
            help_txt_updown.status = FINISHED
            help_txt_updown.setAutoDraw(False)
    
    # *updown_pract* updates
    
    # if updown_pract is starting this frame...
    if updown_pract.status == NOT_STARTED and tThisFlip >= 3.5-frameTolerance:
        # keep track of start time/frame for later
        updown_pract.frameNStart = frameN  # exact frame index
        updown_pract.tStart = t  # local t and not account for scr refresh
        updown_pract.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(updown_pract, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'updown_pract.started')
        # update status
        updown_pract.status = STARTED
        updown_pract.setAutoDraw(True)
    
    # if updown_pract is active this frame...
    if updown_pract.status == STARTED:
        # update params
        pass
    
    # *feedback_updown_txt* updates
    
    # if feedback_updown_txt is starting this frame...
    if feedback_updown_txt.status == NOT_STARTED and tThisFlip >= 3.5-frameTolerance:
        # keep track of start time/frame for later
        feedback_updown_txt.frameNStart = frameN  # exact frame index
        feedback_updown_txt.tStart = t  # local t and not account for scr refresh
        feedback_updown_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(feedback_updown_txt, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'feedback_updown_txt.started')
        # update status
        feedback_updown_txt.status = STARTED
        feedback_updown_txt.setAutoDraw(True)
    
    # if feedback_updown_txt is active this frame...
    if feedback_updown_txt.status == STARTED:
        # update params
        feedback_updown_txt.setText(msg, log=False)
    
    # *nav_vert_key* updates
    waitOnFlip = False
    
    # if nav_vert_key is starting this frame...
    if nav_vert_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        nav_vert_key.frameNStart = frameN  # exact frame index
        nav_vert_key.tStart = t  # local t and not account for scr refresh
        nav_vert_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(nav_vert_key, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'nav_vert_key.started')
        # update status
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
            nav_vert_key.duration = _nav_vert_key_allKeys[-1].duration
    
    # *foot_vert* updates
    
    # if foot_vert is starting this frame...
    if foot_vert.status == NOT_STARTED and tThisFlip >= 3.5-frameTolerance:
        # keep track of start time/frame for later
        foot_vert.frameNStart = frameN  # exact frame index
        foot_vert.tStart = t  # local t and not account for scr refresh
        foot_vert.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(foot_vert, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'foot_vert.started')
        # update status
        foot_vert.status = STARTED
        foot_vert.setAutoDraw(True)
    
    # if foot_vert is active this frame...
    if foot_vert.status == STARTED:
        # update params
        foot_vert.setPos((curr_x, curr_y), log=False)
    
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
    for thisComponent in up_down_practiceComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "up_down_practice" ---
for thisComponent in up_down_practiceComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if nav_vert_key.keys in ['', [], None]:  # No response was made
    nav_vert_key.keys = None
thisExp.addData('nav_vert_key.keys',nav_vert_key.keys)
if nav_vert_key.keys != None:  # we had a response
    thisExp.addData('nav_vert_key.rt', nav_vert_key.rt)
    thisExp.addData('nav_vert_key.duration', nav_vert_key.duration)
thisExp.nextEntry()
# the Routine "up_down_practice" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "left_right_practice" ---
continueRoutine = True
# update component parameters for each repeat
# Run 'Begin Routine' code from code_leftright
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
frameN = -1

# --- Run Routine "left_right_practice" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # Run 'Each Frame' code from code_leftright
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
    
    # if left_right_pract is starting this frame...
    if left_right_pract.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        left_right_pract.frameNStart = frameN  # exact frame index
        left_right_pract.tStart = t  # local t and not account for scr refresh
        left_right_pract.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(left_right_pract, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'left_right_pract.started')
        # update status
        left_right_pract.status = STARTED
        left_right_pract.setAutoDraw(True)
    
    # if left_right_pract is active this frame...
    if left_right_pract.status == STARTED:
        # update params
        pass
    
    # *foot_leftright* updates
    
    # if foot_leftright is starting this frame...
    if foot_leftright.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        foot_leftright.frameNStart = frameN  # exact frame index
        foot_leftright.tStart = t  # local t and not account for scr refresh
        foot_leftright.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(foot_leftright, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'foot_leftright.started')
        # update status
        foot_leftright.status = STARTED
        foot_leftright.setAutoDraw(True)
    
    # if foot_leftright is active this frame...
    if foot_leftright.status == STARTED:
        # update params
        foot_leftright.setPos((curr_x, curr_y), log=False)
    
    # *intro_leftright* updates
    
    # if intro_leftright is starting this frame...
    if intro_leftright.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        intro_leftright.frameNStart = frameN  # exact frame index
        intro_leftright.tStart = t  # local t and not account for scr refresh
        intro_leftright.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(intro_leftright, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'intro_leftright.started')
        # update status
        intro_leftright.status = STARTED
        intro_leftright.setAutoDraw(True)
    
    # if intro_leftright is active this frame...
    if intro_leftright.status == STARTED:
        # update params
        pass
    
    # if intro_leftright is stopping this frame...
    if intro_leftright.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > intro_leftright.tStartRefresh + 4.5-frameTolerance:
            # keep track of stop time/frame for later
            intro_leftright.tStop = t  # not accounting for scr refresh
            intro_leftright.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_leftright.stopped')
            # update status
            intro_leftright.status = FINISHED
            intro_leftright.setAutoDraw(False)
    
    # *feedback_leftright* updates
    
    # if feedback_leftright is starting this frame...
    if feedback_leftright.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
        # keep track of start time/frame for later
        feedback_leftright.frameNStart = frameN  # exact frame index
        feedback_leftright.tStart = t  # local t and not account for scr refresh
        feedback_leftright.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(feedback_leftright, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'feedback_leftright.started')
        # update status
        feedback_leftright.status = STARTED
        feedback_leftright.setAutoDraw(True)
    
    # if feedback_leftright is active this frame...
    if feedback_leftright.status == STARTED:
        # update params
        feedback_leftright.setText(msg, log=False)
    
    # *nav_horz_key* updates
    waitOnFlip = False
    
    # if nav_horz_key is starting this frame...
    if nav_horz_key.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
        # keep track of start time/frame for later
        nav_horz_key.frameNStart = frameN  # exact frame index
        nav_horz_key.tStart = t  # local t and not account for scr refresh
        nav_horz_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(nav_horz_key, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'nav_horz_key.started')
        # update status
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
            nav_horz_key.duration = [key.duration for key in _nav_horz_key_allKeys]
    
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
    for thisComponent in left_right_practiceComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "left_right_practice" ---
for thisComponent in left_right_practiceComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if nav_horz_key.keys in ['', [], None]:  # No response was made
    nav_horz_key.keys = None
thisExp.addData('nav_horz_key.keys',nav_horz_key.keys)
if nav_horz_key.keys != None:  # we had a response
    thisExp.addData('nav_horz_key.rt', nav_horz_key.rt)
    thisExp.addData('nav_horz_key.duration', nav_horz_key.duration)
thisExp.nextEntry()
# the Routine "left_right_practice" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- Prepare to start Routine "summary_intro" ---
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
frameN = -1

# --- Run Routine "summary_intro" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *summary_parrot* updates
    
    # if summary_parrot is starting this frame...
    if summary_parrot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        summary_parrot.frameNStart = frameN  # exact frame index
        summary_parrot.tStart = t  # local t and not account for scr refresh
        summary_parrot.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(summary_parrot, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'summary_parrot.started')
        # update status
        summary_parrot.status = STARTED
        summary_parrot.setAutoDraw(True)
    
    # if summary_parrot is active this frame...
    if summary_parrot.status == STARTED:
        # update params
        pass
    
    # *next_summary_key* updates
    waitOnFlip = False
    
    # if next_summary_key is starting this frame...
    if next_summary_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        next_summary_key.frameNStart = frameN  # exact frame index
        next_summary_key.tStart = t  # local t and not account for scr refresh
        next_summary_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(next_summary_key, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'next_summary_key.started')
        # update status
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
            next_summary_key.duration = _next_summary_key_allKeys[-1].duration
            # a response ends the routine
            continueRoutine = False
    
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
    for thisComponent in summary_introComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "summary_intro" ---
for thisComponent in summary_introComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if next_summary_key.keys in ['', [], None]:  # No response was made
    next_summary_key.keys = None
thisExp.addData('next_summary_key.keys',next_summary_key.keys)
if next_summary_key.keys != None:  # we had a response
    thisExp.addData('next_summary_key.rt', next_summary_key.rt)
    thisExp.addData('next_summary_key.duration', next_summary_key.duration)
thisExp.nextEntry()
# the Routine "summary_intro" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('pract_cond_3x3.xlsx'),
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
    
    # --- Prepare to start Routine "new_rewards_prep" ---
    continueRoutine = True
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
    frameN = -1
    
    # --- Run Routine "new_rewards_prep" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *new_rewards* updates
        
        # if new_rewards is starting this frame...
        if new_rewards.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            new_rewards.frameNStart = frameN  # exact frame index
            new_rewards.tStart = t  # local t and not account for scr refresh
            new_rewards.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(new_rewards, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'new_rewards.started')
            # update status
            new_rewards.status = STARTED
            new_rewards.setAutoDraw(True)
        
        # if new_rewards is active this frame...
        if new_rewards.status == STARTED:
            # update params
            pass
        
        # if new_rewards is stopping this frame...
        if new_rewards.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > new_rewards.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                new_rewards.tStop = t  # not accounting for scr refresh
                new_rewards.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'new_rewards.stopped')
                # update status
                new_rewards.status = FINISHED
                new_rewards.setAutoDraw(False)
        
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
        for thisComponent in new_rewards_prepComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "new_rewards_prep" ---
    for thisComponent in new_rewards_prepComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    
    # --- Prepare to start Routine "parrot_tip" ---
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from parrot_tip_code
    if type == 'backw':
        show_warning_p = 1
    elif type == 'forw':
        show_warning_p = 0
    backw_warning_parrot.setOpacity(show_warning_p)
    # keep track of which components have finished
    parrot_tipComponents = [parrot, backw_warning_parrot]
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
    frameN = -1
    
    # --- Run Routine "parrot_tip" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *parrot* updates
        
        # if parrot is starting this frame...
        if parrot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            parrot.frameNStart = frameN  # exact frame index
            parrot.tStart = t  # local t and not account for scr refresh
            parrot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(parrot, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'parrot.started')
            # update status
            parrot.status = STARTED
            parrot.setAutoDraw(True)
        
        # if parrot is active this frame...
        if parrot.status == STARTED:
            # update params
            pass
        
        # if parrot is stopping this frame...
        if parrot.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > parrot.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                parrot.tStop = t  # not accounting for scr refresh
                parrot.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'parrot.stopped')
                # update status
                parrot.status = FINISHED
                parrot.setAutoDraw(False)
        
        # *backw_warning_parrot* updates
        
        # if backw_warning_parrot is starting this frame...
        if backw_warning_parrot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            backw_warning_parrot.frameNStart = frameN  # exact frame index
            backw_warning_parrot.tStart = t  # local t and not account for scr refresh
            backw_warning_parrot.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(backw_warning_parrot, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'backw_warning_parrot.started')
            # update status
            backw_warning_parrot.status = STARTED
            backw_warning_parrot.setAutoDraw(True)
        
        # if backw_warning_parrot is active this frame...
        if backw_warning_parrot.status == STARTED:
            # update params
            pass
        
        # if backw_warning_parrot is stopping this frame...
        if backw_warning_parrot.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > backw_warning_parrot.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                backw_warning_parrot.tStop = t  # not accounting for scr refresh
                backw_warning_parrot.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'backw_warning_parrot.stopped')
                # update status
                backw_warning_parrot.status = FINISHED
                backw_warning_parrot.setAutoDraw(False)
        
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
        for thisComponent in parrot_tipComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "parrot_tip" ---
    for thisComponent in parrot_tipComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    
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
    while continueRoutine and routineTimer.getTime() < 10.0:
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
            if tThisFlipGlobal > sand_pirate.tStartRefresh + 10.-frameTolerance:
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
        routineTimer.addTime(-10.000000)
    
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
        reward_waiting = random.uniform(2.2, 3.5)
        
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
                # reward_waiting = random.uniform(1, 2.5)
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
                        #reward_waiting = random.uniform(1,2.5)
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
                        steps_taken_this_subpath = 0
                        print(f'this subpath you took {steps_taken_this_subpath} steps. should be 0.')
                        # and add the same last value, so it wouldn't enter this loop again.
                        index_rew.append(index_rew[-1])
                        # and update the reward!! now off to find the next one.
                        curr_state = states[index_rew[-1]]
                        curr_rew_x = rew_x[index_rew[-1]] 
                        curr_rew_y = rew_y[index_rew[-1]]
                
                        # and move on to the next subpath_length! 
                        curr_exp_step_no = all_exp_step_nos[index_rew[-1]]
                        secs_per_step = jitter(curr_exp_step_no, 5.75, var_per_step=1.5)
                        steps_taken_this_subpath= 0
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
            
                                if steps_taken_this_subpath < len(secs_per_step):
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps.')
                                else:
                                    isi = random.uniform(0.6, 2.2)
            
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
                                if steps_taken_this_subpath < len(secs_per_step):
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps.')
                                else:
                                    isi = random.uniform(0.6, 2.2)
            
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
                                if steps_taken_this_subpath < len(secs_per_step):
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps.')
                                else:
                                    isi = random.uniform(0.6, 2.2)
            
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
                                if steps_taken_this_subpath < len(secs_per_step):
                                    isi = secs_per_step[steps_taken_this_subpath]
                                    steps_taken_this_subpath += 1
                                    print(f'this subpath you took {steps_taken_this_subpath} steps.')
                                else:
                                    isi = random.uniform(0.6, 2.2)
            
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
                foot.setOpacity(foot_disappear, log=False)
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
                progressbar_background.setOpacity(progress_disappear, log=False)
            
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
                progress_bar.setOpacity(progress2_disappear, log=False)
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
    while continueRoutine and routineTimer.getTime() < 5.0:
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
            if tThisFlipGlobal > background_feedback.tStartRefresh + 5-frameTolerance:
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
            if tThisFlipGlobal > feedback_text.tStartRefresh + 5-frameTolerance:
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
        routineTimer.addTime(-5.000000)
# completed 1.0 repeats of 'trials'


# --- Prepare to start Routine "the_end" ---
continueRoutine = True
# update component parameters for each repeat
end_practice_key.keys = []
end_practice_key.rt = []
_end_practice_key_allKeys = []
# keep track of which components have finished
the_endComponents = [full_ship_end, ship_box, end_practice_key, outro]
for thisComponent in the_endComponents:
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

# --- Run Routine "the_end" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *full_ship_end* updates
    
    # if full_ship_end is starting this frame...
    if full_ship_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        full_ship_end.frameNStart = frameN  # exact frame index
        full_ship_end.tStart = t  # local t and not account for scr refresh
        full_ship_end.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(full_ship_end, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'full_ship_end.started')
        # update status
        full_ship_end.status = STARTED
        full_ship_end.setAutoDraw(True)
    
    # if full_ship_end is active this frame...
    if full_ship_end.status == STARTED:
        # update params
        pass
    
    # if full_ship_end is stopping this frame...
    if full_ship_end.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > full_ship_end.tStartRefresh + 2-frameTolerance:
            # keep track of stop time/frame for later
            full_ship_end.tStop = t  # not accounting for scr refresh
            full_ship_end.frameNStop = frameN  # exact frame index
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'full_ship_end.stopped')
            # update status
            full_ship_end.status = FINISHED
            full_ship_end.setAutoDraw(False)
    
    # *ship_box* updates
    
    # if ship_box is starting this frame...
    if ship_box.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
        # keep track of start time/frame for later
        ship_box.frameNStart = frameN  # exact frame index
        ship_box.tStart = t  # local t and not account for scr refresh
        ship_box.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(ship_box, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'ship_box.started')
        # update status
        ship_box.status = STARTED
        ship_box.setAutoDraw(True)
    
    # if ship_box is active this frame...
    if ship_box.status == STARTED:
        # update params
        pass
    
    # *end_practice_key* updates
    waitOnFlip = False
    
    # if end_practice_key is starting this frame...
    if end_practice_key.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
        # keep track of start time/frame for later
        end_practice_key.frameNStart = frameN  # exact frame index
        end_practice_key.tStart = t  # local t and not account for scr refresh
        end_practice_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_practice_key, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'end_practice_key.started')
        # update status
        end_practice_key.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(end_practice_key.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(end_practice_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if end_practice_key.status == STARTED and not waitOnFlip:
        theseKeys = end_practice_key.getKeys(keyList=['1'], waitRelease=False)
        _end_practice_key_allKeys.extend(theseKeys)
        if len(_end_practice_key_allKeys):
            end_practice_key.keys = _end_practice_key_allKeys[-1].name  # just the last key pressed
            end_practice_key.rt = _end_practice_key_allKeys[-1].rt
            end_practice_key.duration = _end_practice_key_allKeys[-1].duration
            # a response ends the routine
            continueRoutine = False
    
    # *outro* updates
    
    # if outro is starting this frame...
    if outro.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
        # keep track of start time/frame for later
        outro.frameNStart = frameN  # exact frame index
        outro.tStart = t  # local t and not account for scr refresh
        outro.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(outro, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'outro.started')
        # update status
        outro.status = STARTED
        outro.setAutoDraw(True)
    
    # if outro is active this frame...
    if outro.status == STARTED:
        # update params
        pass
    
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
    for thisComponent in the_endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "the_end" ---
for thisComponent in the_endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if end_practice_key.keys in ['', [], None]:  # No response was made
    end_practice_key.keys = None
thisExp.addData('end_practice_key.keys',end_practice_key.keys)
if end_practice_key.keys != None:  # we had a response
    thisExp.addData('end_practice_key.rt', end_practice_key.rt)
    thisExp.addData('end_practice_key.duration', end_practice_key.duration)
thisExp.nextEntry()
# the Routine "the_end" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
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
