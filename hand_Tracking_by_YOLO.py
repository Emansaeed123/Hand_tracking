#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:29:32 2020

"""
import argparse
import cv2
import glob
import os
from yolo import YOLO
def reading_video(filename):
        ap = argparse.ArgumentParser()
        ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn / v4-tiny')
        ap.add_argument('-d', '--device', default=0, help='Device to use')
        ap.add_argument('-v', '--videos', default="videos", help='Path to videos or video file')
        ap.add_argument('-s', '--size', default=416, help='Size for yolo')
        ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
        ap.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
        args = ap.parse_args()
        if args.network == "normal":
            print("loading yolo...")
            yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
        elif args.network == "prn":
            print("loading yolo-tiny-prn...")
            yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
        elif args.network == "v4-tiny":
            print("loading yolov4-tiny-prn...")
            yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
        else:
            print("loading yolo-tiny...")
            yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

        yolo.size = int(args.size)
        yolo.confidence = float(args.confidence)
        
        # opening a window called preview 
        cv2.namedWindow("preview")
        # to open and capture frames from video
        vc = cv2.VideoCapture(filename)
        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
            # to get the first frame
        
        else:
            # some error causes the video to not open
            rval = False
        while (vc.isOpened()):
            # Applying YOLO on the frames
            width, height, inference_time, results = yolo.inference(frame)
            for detection in results:
                id, name, confidence, x, y, w, h = detection
                cx = x + (w / 2)
                cy = y + (h / 2)
                # draw a bounding box rectangle and label on the image
                color = (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) 
                text = "%s (%s)" % (name, round(confidence, 2))
                # put a text on the detected hand with the confidence ratio
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                cv2.imshow("preview", frame)
                rval, frame = vc.read()
                # to close the window we need to click on the ESC button
                key = cv2.waitKey(20)
                if key == 27:  # exit on ESC
                    break
        cv2.destroyWindow("preview")
        vc.release()

reading_video('videos/hand_move.mp4')