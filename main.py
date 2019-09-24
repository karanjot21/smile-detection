# importing necessary packages
import numpy as np
import pandas as pd
import argparse
import cv2
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import time
import numpy as np
import os
import numpy as np
import json
import sys
import math

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

protpath = os.path.join(os.getcwd(),'deploy.prototxt.txt')
modpath = os.path.join(os.getcwd(),'res10_300x300_ssd_iter_140000.caffemodel')

net = cv2.dnn.readNetFromCaffe(protpath, modpath)
conf = 0.3 ## adj ## 0.2

useDlib = True
thresh = 0.25
smileThresh = 0.5
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

def face_points_detection(img, bbox):
    PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    # return the list of (x, y)-coordinates
    return coords

def viewOriginal(image):
    cv2.imshow("Original Size", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def view(image):
    cv2.imshow("Resized", imutils.resize(image, width=1000))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveTemp(image,name=''):
    cv2.imwrite('temp/'+name+str(int(time.time()*100))+'.jpg',image)

def select_face(bbox, im, r=10):
    points = np.asarray(face_points_detection(im, bbox))
    
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    
    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]

def smileVal(s1,s2):
    if np.isnan(s1):
        return s2

    if np.isnan(s2):
        return s1

    # if s1>0.3 or s2>0.3:
    #     return max(s1,s2)

    return (s1+s2)/2


def getFaces(img):
    subjects = []
    image = img.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > conf:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faceBoxRectangleS = dlib.rectangle(left=int(startX), top=int(startY), right=int(endX), bottom=int(endY))
            subjects.append(faceBoxRectangleS)
            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    # view(image)
    # saveTemp(image)
    return subjects#,fcords

def angle_between(p1, p2):
    # ang1 = np.arctan2(*p1[::-1])
    # ang2 = np.arctan2(*p2[::-1])
    # if(p2[1]<p1[1]):
    #     return -1 * np.rad2deg(ang1 - ang2)
    # else:
    #     return np.rad2deg(ang1 - ang2)
    # # return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return math.degrees(math.atan2(yDiff, xDiff))
    

def getSubImage(p1, p2, rect, src):
    center, size, theta = rect
    theta = angle_between(p1,p2)
    # print(theta,center)

    if theta < -45:
        theta = (90 + theta)
    # else:
    #     theta = -theta  
    
    center = tuple(map(int, center))

    M = cv2.getRotationMatrix2D(center, theta, 1)
    out = cv2.transform(src, M)
    # print("out", len(out))
    return out


def getSmileVal(img,subject,filename):
    # print("###")
    image = img
    frame = image.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    shape = predict(gray, subject)
    shape = face_utils.shape_to_np(shape)
    mouth = shape[mStart:mEnd]

    mouthHull = cv2.convexHull(mouth)
    rect = cv2.minAreaRect(mouthHull)
    cv2.drawContours(frame,[np.int0(cv2.boxPoints(rect))],0,(150,100,100),2)
    xpoints = [x[0][0] for x in mouthHull]
    ypoints = [x[0][1] for x in mouthHull]
    xminidx = np.argmin(xpoints)
    xminy = ypoints[xminidx]
    xminx = xpoints[xminidx]
    xmaxidx = np.argmax(xpoints)
    xmaxx = xpoints[xmaxidx]
    xmaxy = ypoints[xmaxidx]

    mouthHullRotated = getSubImage((xminx,xminy),(xmaxx,xmaxy), rect,mouthHull)

    xpoints = [x[0][0] for x in mouthHullRotated]
    ypoints = [x[0][1] for x in mouthHullRotated]
    
    xminidx = np.argmin(xpoints)
    y1 = ypoints[xminidx]
    x1 = xpoints[xminidx]
    # print("\nxmin: "+str(x1)+","+str(y1))

    yminidx = np.argmin(ypoints)
    y2 = ypoints[yminidx]
    x2 = xpoints[yminidx]
    # print("ymin: "+str(x2)+","+str(y2))

    xmaxidx = np.argmax(xpoints)
    x3 = xpoints[xmaxidx]
    y3 = ypoints[xmaxidx]
    # print("xmax: "+str(x3)+","+str(y3))
    
    ymaxidx = np.argmax(ypoints)
    x4 = xpoints[ymaxidx]
    y4 = ypoints[ymaxidx]
    # print("ymax: "+str(x4)+","+str(y4))

    s1 = (y4 - y1)/(x2 - x1)
    s2 = (y4 - y3)/(x3 - x4)
    print("s1: "+str(s1)+", s2: "+str(s2))
    cv2.circle(frame, (x2,y2), 5, (0,0,255), -1)
    # print("SmileVal: "+str(smileVal(s1,s2)),end="\n\n")

    # view(frame)
    saveTemp(frame,filename)

    return smileVal(s1,s2),s1,s2


def main():
    df = pd.read_csv('genki4k/labels.txt',delimiter=' ')
    labels = list(df['class'])
    
    c = 1
    zn = 3
    result = pd.DataFrame()

    for i in range(len(labels)):
        if((i+1)==(10*c)):
            zn -= 1
            c *= 10
        z = '0'*zn
        filename = 'file'+z+str(i+1)+'.jpg'

        image = cv2.imread('genki4k/files/'+filename)
        s = getFaces(image)
        # if (len(s)!=1):
        #     print("issue in face detection")
        sub = s[0]
        smileVal,s1,s2 = getSmileVal(image,sub,filename)
        
        result.loc[i, 'file'] = filename

        if smileVal>smileThresh:
            print(filename,smileVal)
            print(1+labels[i])
            if(1+labels[i]==2):
                result.loc[i, 'correct'] = 1
            else:
                result.loc[i, 'correct'] = 0
            result.loc[i, 'smiling'] = 1
        else:
            print(filename,smileVal)
            print(0+labels[i])
            if(0+labels[i]==0):
                result.loc[i, 'correct'] = 1
            else:
                result.loc[i, 'correct'] = 0
            result.loc[i, 'smiling'] = 0
        
        result.loc[i, 'label'] = labels[i]
        result.loc[i, 'smileval'] = smileVal
        result.loc[i, 's1'] = s1
        result.loc[i, 's2'] = s2
    
    result.to_csv(str(int(time.time()*100))+'.csv')


if __name__ == "__main__":
    main()