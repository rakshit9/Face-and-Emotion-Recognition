
import os

import numpy as np

import cv2


#from PIL import Image





path = "/home/rakshit/Documents/finalyearproject/dataset"

faces = []
faceID = []
 

for p, subdirnames, filenames in os.walk(path):
    for filename in filenames:
        if filename.startswith('.'):
            continue
        id = os.path.basename(p)
        
        img_path = os.path.join(p,filename)
        print("id:" ,id,"img_path :",img_path)
        test_img = cv2.imread(img_path)
        if test_img is None:
            continue
        
        gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('/home/rakshit/.local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
        faces_rect = face_cascade.detectMultiScale(gray_img,1.32 ,5)
        
        if len(faces_rect) != 1:
            continue
        (x,y,w,h) = faces_rect[0]
        
        (x,y,w,h) = faces_rect[0]
        
        roi_gray = gray_img[y: y+h, x:x+w]
        faces.append(roi_gray)
        faceID.append(id)
        
recognizer = cv2.face.LBPHFaceRecognizer_create();
                
#recognizer.train(faces,np.array(faceID))                
        roi_gray = gray_img[y: y+h, x:x+w]
        faces.append(roi_gray)
        faceID.append(id)
        
recognizer = cv2.face.LBPHFaceRecognizer_create();
                
recognizer.train(faces,np.array(faceID))                
recognizer.save("/home/rakshit/Documents/finalyearproject/trainingdata.yml")




















