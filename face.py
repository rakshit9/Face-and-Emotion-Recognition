#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:51:29 2019

@author: rakshit
"""



import numpy as np
import cv2


 
    


def main():
   
   # face_cascade = cv2.CascadeClassifier('/home/rakshit/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    
    
    cap = cv2.VideoCapture(0)
    
    while(True):        
        
        ret,frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            
            
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) ==27:
            break


    cap.release()
    cv2.destroyAllWindows()
        



    
    
    
if __name__ == "__main__":
    main()





