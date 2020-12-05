#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##### IMPORTING LIBRARIES ######

import cv2
import matplotlib.pyplot as plt
### HAND DETECTION CLASS ###
class hand_detection:
    
    ### READING IMAGE AND CHANGE IT COLOR MAPPING AND RETURN IT ###
    def reading_image(self): 
        global image
        ### READING IMAGE ###
        image = cv2.imread('hand.jpeg')
        ### FROM BGR 2 RGB ###
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        return image
    
    ### CROPPING FUNCTION BY PASSING IMAGE AND USING ROI TO LIMIT REGION ###
    def cropping_image(self, image):
        x = 79   # INTIIAL POINT OF X AXIS
        y = 55   # INTIIAL POINT OF Y AXIS
        h = 105  # HEIGHT 
        w = 105  # WIDTH
        image_copy = image.copy()
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        crop_image = image_copy[y:y+h, x:x+w]  # REGION OF INTERSET FOR CROPPING 
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        plt.imshow(crop_image)
        
    ### DETECT HAND BY DRAWING RECTANGLE OF ROI ###     
    def detect_hand(self, image):
        image_detect_copy = image.copy()
        cv2.rectangle(image_detect_copy, pt1 = (79,55), pt2 = (184, 160), color =(0,255,0), thickness = 5)
        plt.imshow(image_detect_copy)
        
image1 = hand_detection()
image1.reading_image()
image1.cropping_image(image)
image1.detect_hand(image)
