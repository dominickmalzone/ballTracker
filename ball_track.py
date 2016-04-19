from collections import deque
import numpy as np
import argparse
import imutils
import cv2

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the optional video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

#define the lower and upper boundaries of the green ball and initialize the list of tracked points
#hsv color space
greenLower = (29,86, 6)
greenUpper = (64,255,255)
pts = deque(maxlen=args["buffer"])

#if a video path was not given grab the webcam
if not args.get("video",False);
    camera = cv2.VideoCapture(0)

#otherwise, grab the video
else:
    camera = cv2.VideoCapture(args["video"])

#keep the looooop
while True:
    #grab the current frame
    (grabbed, frame) = camera.read()

    #if we are viewing a video and we did not grab, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    #resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    #blurre = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #contruct a mask for the color green, then perform a series of dilations and erosions to remove
    # any blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask,None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    #pickup later
