# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:14:36 2022

@author: Wayne
"""

import time
import torch 
import numpy as np
import cv2
import mss


model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 1024, "height": 788}



    while True:
        screen = np.array(sct.grab(monitor))
        ret, frame = True,cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        if ret:
            results = model([frame])
            results.render()
            cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            
        
    cap.release()
    cv2.destroyAllWindows()
