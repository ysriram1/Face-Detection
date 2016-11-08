# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:08:24 2016

@author: Sriram Yarlagadda

Written in Python 2.7
"""

import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

os.chdir('C:/Users/SYARLAG1/Desktop/Face-Detection') # haar cascade location

# reading in cascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # building the classifier for face detection.
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml') # building the classifier for eye detection.
os.chdir('C:/Users/SYARLAG1/Desktop/') # image location
imMat = cv2.imread('upe.jpg')


grayImMat = cv2.cvtColor(imMat, cv2.COLOR_BGR2GRAY) # image needs to be in grayscale for Haar

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    grayImMat,
    scaleFactor=1.1,
    minNeighbors=20, # higher value results in a lower senstivity
    #minSize=(30, 30),
)

eyes = eyeCascade.detectMultiScale(
    grayImMat
)

realFaces = [] # faces after removing fake ones go here
facesLst = []
facesLstgray = []
count = 0
for x, y, w, h in faces:
    print count
    count += 1
    print x,y,w,h
    faceImgCol = imMat[y:y+h,x:x+w]
    faceImgGray = grayImMat[y:y+h,x:x+w]
    # check to see if there are eyes, if not, they arent faces (removing false positives)
    eyes = eyeCascade.detectMultiScale(faceImgGray)
    if len(eyes) == 2:
        facesLst.append(faceImgCol)
        facesLstgray.append(faceImgGray)
        realFaces.append([x,y,w,h])

os.chdir('./faces')
for iface, face in enumerate(facesLst):
    fileName = 'face'+str(iface+1)+'.jpg'
    cv2.imwrite(fileName, face)
    
for x, y, w, h in realFaces:
    cv2.rectangle(imMat, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite('upeDetected.jpg', imMat)



# Face preprocessing
## Removing the background (with thanks to stackoverflow:jedwards)
def removeBackground(img, canny1, canny2, ):

ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms




# cv2.imshow("Faces detected" ,imMat) # usiong this causes kernel to crash in windows

# Detect facial features
