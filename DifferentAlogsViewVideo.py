'''
Created on Sep 22, 2016

@author: isyed
'''
import json
import time

import cv2
# import imutils

import numpy as np
import base64
import os

def _processORB(sourceFrame,targetFrame):
    e1_orb = cv2.getTickCount()
    orb = cv2.ORB_create()
    kp = orb.detect(sourceFrame, None)
    kp, des = orb.compute(sourceFrame, kp)
    e2_orb = cv2.getTickCount()
    time_orb = (e2_orb-e1_orb)/cv2.getTickFrequency()
    print ('ORB\t' + str(time_orb) + '\t' + str(time_orb/len(kp)))
    return cv2.drawKeypoints(targetFrame, kp, None, color=(0, 255, 0))

def _processAkaze(sourceFrame,targetFrame):
    e1_akaze = cv2.getTickCount()
    akaze = cv2.AKAZE_create()
    kp_akaze = akaze.detect(sourceFrame, None)
    kp_akaze, des_akaze = akaze.compute(sourceFrame, kp_akaze)
    #img_akaze = cv2.drawKeypoints(gray, kp_akaze, None, color=(0, 255, 0))
    e2_akaze = cv2.getTickCount()
    time_akaze = (e2_akaze-e1_akaze)/cv2.getTickFrequency()
    print ('AKAZE\t' + str(time_akaze) + '\t' + str(time_akaze/len(kp_akaze)))
    return cv2.drawKeypoints(targetFrame, kp_akaze, None)

def _processKaze(sourceFrame,targetFrame):
    # KAZE
    e1_kaze = cv2.getTickCount()
    kaze = cv2.KAZE_create()
    kp_kaze = kaze.detect(sourceFrame, None)
    kp_kaze, des_kaze = kaze.compute(sourceFrame, kp_kaze)
    #img_kaze = cv2.drawKeypoints(gray, kp_kaze, None, color=(0, 255, 0))
    e2_kaze = cv2.getTickCount()
    time_kaze = (e2_kaze-e1_kaze)/cv2.getTickFrequency()
    print ('KAZE\t' + str(time_kaze) + '\t' + str(time_kaze/len(kp_kaze)))
    return cv2.drawKeypoints(targetFrame, kp_kaze, None)

def _processFAST(sourceFrame,targetFrame):
    # FAST
    e1_fast = cv2.getTickCount()
    fast = cv2.ORB_create()
    kp_fast = fast.detect(sourceFrame, None)
    kp_fast, des_fast = fast.compute(sourceFrame, kp_fast)
    e2_fast = cv2.getTickCount()
    time_fast = (e2_fast-e1_fast)/cv2.getTickFrequency()
    print ('FAST\t' + str(time_fast) + '\t' + str(time_fast/len(kp_fast)))
    return cv2.drawKeypoints(targetFrame, kp_fast, None)

def _processBRISK(sourceFrame,targetFrame):
    # BRISK
    e1_brisk = cv2.getTickCount()
    brisk = cv2.ORB_create()
    kp_brisk = brisk.detect(sourceFrame, None)
    kp_brisk, des_brisk = brisk.compute(sourceFrame, kp_brisk)
    e2_brisk = cv2.getTickCount()
    time_brisk = (e2_brisk-e1_brisk)/cv2.getTickFrequency()
    print ('BRISK\t' + str(time_brisk) + '\t' + str(time_brisk/len(kp_brisk)))
    return cv2.drawKeypoints(targetFrame, kp_brisk, None)

def _processMSER(sourceFrame,targetFrame):
    # MSER
    mser = cv2.MSER_create()
    img_mser = targetFrame.copy()
    regions = mser.detectRegions(sourceFrame, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img_mser, hulls, 1, (0, 255, 0))
    cv2.imshow('_processBRISK',img_mser)
    

capture = cv2.VideoCapture(0)
while True:
    retvalOne, imageOne = capture.read()
    #imageGrayOne = cv2.cvtColor(imageOne, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('Difference Image',_identifyObjectThroughSWIFT(imageOne.copy(),imageOne))
    
    cv2.imshow('_processORB',_processORB(imageOne.copy(),imageOne.copy()))
    #cv2.imshow('_processAkazeq',_processAkaze(imageOne.copy(),imageOne.copy()))
    #cv2.imshow('_processKaze',_processKaze(imageOne.copy(),imageOne.copy()))
    
    #cv2.imshow('_processFAST',_processFAST(imageOne.copy(),imageOne.copy()))
    #cv2.imshow('_processBRISK',_processBRISK(imageOne.copy(),imageOne.copy()))
    _processMSER(imageOne.copy(),imageOne.copy())
    
     
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break;

capture.release()
cv2.destroyAllWindows() 
