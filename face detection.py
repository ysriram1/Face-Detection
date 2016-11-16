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
import dlib


os.chdir('C:/Users/SYARLAG1/Desktop/Face-Detection') # haar cascade location

# reading in cascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # building the classifier for face detection.
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml') # building the classifier for eye detection.
os.chdir('C:/Users/SYARLAG1/Desktop/') # image location
imMat = cv2.imread('cubs.jpg')


grayImMat = cv2.cvtColor(imMat, cv2.COLOR_BGR2GRAY) # image needs to be in grayscale for Haar


## Removing the background (with thanks to stackoverflow:jedwards)
## rec: should have coords of a rectangle into whicht the image will fit into
def removeBackground(img, rec):
    mask = np.zeros(img.shape[:2],np.uint8) # create mask of same dim as image
    
    bgdModel = np.zeros((1,65),np.float64) # background
    fgdModel = np.zeros((1,65),np.float64)  # foreground
 
    cv2.grabCut(img,mask,rec,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    img = img*mask2[:,:,np.newaxis]
    
    return img

# second background removal algo
def removeBackground2(img):
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0,0,0) # In BGR format
    
    # Read image 

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Edge detection 
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # Find contours in edges, sort by area 
    contour_info = []
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
    
    # Create empty mask, draw filled polygon on it corresponding to largest contour 
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    
    # Smooth mask, then blur it 
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
    
    # Blend masked img into MASK_COLOR background 
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending
    
    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to
    
    return masked

def padImage(imageMat, maxRows, maxCols):
    padImageMat = np.zeros(shape=[maxRows,maxCols])
    imageMatShape = imageMat.shape
    rowDiff = maxRows - imageMatShape[0]
    colDiff = maxCols - imageMatShape[1]

    rowOffSet = rowDiff//2 
    colOffSet = colDiff//2
    padImageMat[rowOffSet:imageMatShape[0]+rowOffSet,colOffSet:imageMatShape[1]+colOffSet] = imageMat
    
    return padImageMat           



####################################### Detect and Extract faces in the image #
faces = faceCascade.detectMultiScale(
    grayImMat,
    scaleFactor=1.1,
    minNeighbors=5, # higher value results in a lower senstivity
    #minSize=(30, 30),
)

realFaces = [] # faces after removing fake ones go here
facesLst = []
facesLstgray = []
count = 0

eyeCheck = False

for x, y, w, h in faces:
    print count
    count += 1 # just a counter to make sure we iterate through all partitions

    faceImgCol = imMat[y:y+h,x:x+w]
    faceImgGray = grayImMat[y:y+h,x:x+w]
    # check to see if there are eyes, if not, they arent faces (removing false positives)
    if eyeCheck:
        eyes = eyeCascade.detectMultiScale(faceImgGray)
        if len(eyes) == 2:
            facesLst.append(faceImgCol)
            facesLstgray.append(faceImgGray)
            realFaces.append([x,y,w,h])
    else:
        facesLst.append(faceImgCol)
        facesLstgray.append(faceImgGray)
        realFaces.append([x,y,w,h])

os.chdir('./faces')
for iface, face in enumerate(facesLst):
    fileName = 'face'+str(iface+1)+'.jpg'
    cv2.imwrite(fileName, face)
   
#run this to draw rectangles around orignal image
for x, y, w, h in realFaces:
    cv2.rectangle(imMat, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite('withoutEyes.jpg', imMat)

# apply foreground extraction to get the foreground of an image
#img =  np.copy(faceImgCol); rec = (0,0,img.shape[1]-1,img.shape[0]-1)
#
#mask = np.zeros(img.shape[:2],np.uint8) # create mask of same dim as image
#
#bgdModel = np.zeros((1,65),np.float64) # background
#fgdModel = np.zeros((1,65),np.float64)  # foreground
#
#cv2.grabCut(img,mask,rec,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#img = img*mask2[:,:,np.newaxis] 
#
#cv2.imwrite('testbackextracted.jpg', img)

# background removal not doing much here
os.chdir('C:/Users/syarlag1/Desktop/faces_backgroundExtracted')

faceBGRemoved = []

for i,img in enumerate(facesLst):
    fileName = 'face'+str(i+1)+'.jpg'
    rec = (0,0,img.shape[1]-1,img.shape[0]-1)
    newImg = removeBackground(img,rec)
    cv2.imwrite(fileName, newImg)
    faceBGRemoved.append(removeBackground(img,rec))
      

os.chdir('C:/Users/syarlag1/Desktop/faces_backgroundExtracted2')

faceBGRemoved2 = []

for i,img in enumerate(facesLst):
    fileName = 'face'+str(i+1)+'.jpg'
    newImg = removeBackground2(img)
    cv2.imwrite(fileName, newImg)
    faceBGRemoved2.append(newImg)

##################################### Align the faces ########################
## Training a predictor, with thanks to http://dlib.net/train_shape_predictor.py.html
options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

os.chdir('C:/Users/SYARLAG1/Desktop/Face-Detection')
# this may take a while ... uses the images in the folder. DONT run if predictor.dat already exists
dlib.train_shape_predictor('testing_with_face_landmarks.xml', "predictor.dat", options)

def faceAlign(im):   
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("predictor.dat")
    imNew = detector(im,1)
    return np.matrix([[p.x, p.y] for p in predictor(im, imNew[0]).parts()])

############################################### Finding the mean image #######  
maxRows = 0
maxCols = 0

facesCount = len(facesLst)

for face in facesLst: # finding max dimensions
    r, c, _ = face.shape
    if r > maxRows: maxRows = r
    if c > maxCols: maxCols = c    
    

def reSize(x): return cv2.resize(x, (r,c), interpolation = cv2.INTER_NEAREST)

facesLstResized = [reSize(face) for face in faceBGRemoved]

## first method
#facesMean = (sum(facesLstResized)).astype('uint8') 
#cv2.imwrite('meanImg.jpg', facesMean)
#
## the average of each channel
#redMat = []
#greenMat = []
#blueMat = []
#
#for face in facesLstResized:
#    redMat.append(face[:,:,0])
#    greenMat.append(face[:,:,1])
#    blueMat.append(face[:,:,2])
#
#
#redMatMean = (sum(redMat)/facesCount).astype('uint8')  
#greenMatMean = (sum(greenMat)/facesCount).astype('uint8')  
#blueMatMean = (sum(blueMat)/facesCount).astype('uint8')  
#
#imMeanMat = np.dstack((redMatMean,greenMatMean,blueMatMean))
#
#cv2.imwrite('meanImg2.jpg', imMeanMat)

# Third method
sumIm = np.zeros(shape=[r,c,3])
for face in facesLstResized[1:]:
    sumIm = sumIm + face

facesMean = (sumIm/facesCount).astype('uint8')
cv2.imwrite('meanImg3.jpg', facesMean)


# with thanks to https://github.com/spmallick/learnopencv/blob/master/FaceAverage/faceAverage.py

# cv2.imshow("Faces detected" ,imMat) # usiong this causes kernel to crash in windows

# Detect facial features
