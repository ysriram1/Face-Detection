# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:08:24 2016

@author: syarlag1
"""

import os
import numpy as np
import pandas as pd
import cv2
os.chdir('C:/Users/SYARLAG1/Desktop/CSC481/Project')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # building the classifies using Haar face XML
imMat = cv2.imread('upe.jpg')
grayImMat = cv2.cvtColor(imMat, cv2.COLOR_BGR2GRAY) # image needs to be in grayscale for Haar

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    grayImMat,
    scaleFactor=1.1,
    minNeighbors=20,
    #minSize=(30, 30),
)

facesLst = []
facesLstgray = []
count = 0
for x, y, w, h in faces:
    print count
    count += 1
    print x,y,w,h
    facesLst.append(imMat[y:y+h,x:x+w])
    facesLstgray.append(grayImMat[y:y+h,x:x+w])
    
for (x, y, w, h) in faces:
    cv2.rectangle(imMat, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite('upeDetected.jpg', imMat)

os.chdir('./faces')
for iface, face in enumerate(facesLst):
    fileName = 'face'+str(iface+1)+'.jpg'
    cv2.imwrite(fileName, face)

#cv2.imshow("Faces detected" ,imMat)

# Detect facial features
