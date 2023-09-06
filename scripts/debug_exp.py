#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:08:22 2023

@author: xpsy1114
"""

# test my function experiment update stuff

import time
import random
import pdb; pdb.set_trace() 


start_time = time.perf_counter()
move_timer = []
move_timer.append(time.perf_counter()-start_time)
move_counter = 0
interval = 0.001
curr_y = 0

for i in range(0,10000):
    if curr_y < 29/100:
        key2_press_trigger = 1
        if move_counter == 0:
            last_y = curr_y
            time_at_press = time.perf_counter()-start_time
            update_timer = time_at_press + interval
            move_counter = 1
            isi = 2
            print(f'isi is {isi}')
            time_at_press_isi = time_at_press+isi
            print(f'time at press plus isi is {time_at_press_isi}')
        if move_timer[-1] < time_at_press+isi:
            move_timer.append(time.perf_counter()-start_time)
            print(f'move timer is {move_timer[-1]}')
            if move_timer[-1] >= update_timer:
                curr_y += ((29/100)/(isi/interval))
                print(f' y is {curr_y}')
                update_timer += interval
                print(f'update timer is now {update_timer}')
                if curr_y > 29/100:
                    curr_y = 29/100
        if move_timer[-1] >= time_at_press+isi:
            curr_y = last_y + 29/100
            move_counter = 0
            key2_press_trigger = 0
            print('done moving')
            break
                

                
 # if (keys[-1].name == '4') or (key4_press_trigger == 1):
 #     if curr_x < 21/100:
 #         key4_press_trigger = 1
 #         if move_counter == 0:
 #             last_x = curr_x
 #             time_at_press = globalClock.getTime()
 #             update_timer = time_at_press + interval
 #             move_counter = 1
 #         if move_timer[-1] < time_at_press+isi:
 #             move_timer.append(globalClock.getTime())
 #             if move_timer[-1] >= update_timer:
 #                 curr_x += ((20/100)/(isi/interval))
 #                 update_timer += interval
 #         elif move_timer[-1] >= time_at_press+isi:
 #             curr_x = last_x + 21/100
 #             move_counter = 0
 #             key4_press_trigger = 0
 #             direc = 'to the right'
 #             # add the location to the location you walked.
 #             locs_walked.append(curr_x)