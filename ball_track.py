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
if not args.get("video",False):
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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #contruct a mask for the color green, then perform a series of dilations and erosions to remove
    # any blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask,None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    #find countoursr in the mask and initilze the current (x,y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    #only proceed if at least on contour was found
    if len(cnts) > 0:
        #find largest contour in the mask, then use it to compute the min enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        #only proceed if the radius meets a minimum size
        if radius > 10:
            #draw the circle and centriod on the frame, then update list of tracked points
            cv2.circle(frame, (int(x),int(y)), int(radius), (0,255,255),2)
            cv2.circle(frame, center, 5,(0,0,255), -1)
        #update points
        pts.appendleft(center)
        #loop over the set of tracked points
        for i in xrange(1, len(pts)):
            #if either of the tracked points are none ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue

            #otherwise, compute the thickness of the line and draw connecting lines
            thickness = int(np.sqrt(args["buffer"]/ float(i + 1))* 2.5)
            cv2.line(frame,pts[i - 1], pts[i], (0,255,0),thickness)

        cv2.imshow('BallTrackin Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        #stop loop on q key
        if key == ord('q'):
            break

#cleanup, cleanup, everybody cleanup
camera.release()
cv2.destroyAllWindows()








