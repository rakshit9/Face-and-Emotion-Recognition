#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:43:43 2019

@author: rakshit
"""

import numpy as np
import cv2



face_cascade = cv2.CascadeClassifier('/home/rakshit/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

rec = cv2.createLBPHFaceRecoginzer();

rec.load("/home/rakshit/Documents/finalyearproject/trainingdata.yml")


id = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==2):
            id="rakshit"
        if id==1:
            id="rakshit"
        if id==3:
            id="vivekand"
        
        
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()

