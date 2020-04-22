# -*- coding: utf-8 -*-
"""
@author: diyasinha
"""


import cv2
import numpy as np
from keras.models import load_model

class DrowsinessDetector:
    def __init__(self):
        self.model = load_model('classifeye.h5')
        self.leftEye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
        self.rightEye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
        self.rpred, self.lpred = [], []
        
    
    def detector(self,faces, frame, gray):       
        
        height,width = frame.shape[:2] 
        
        left_eye = self.leftEye.detectMultiScale(gray)
        right_eye = self.rightEye.detectMultiScale(gray)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
    
        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye = r_eye/255
            #print(r_eye)
            r_eye=  r_eye.reshape(24,24,-1)
            #print(r_eye)
            r_eye = np.expand_dims(r_eye,axis=0)
            #returns 0 and 1 (for eyes closed and open)
            self.rpred = self.model.predict_classes(r_eye) 
            break
    
        for (x,y,w,h) in left_eye:
            l_eye = frame[y:y+h,x:x+w]
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye = l_eye/255
            l_eye = l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            #returns 0 and 1 (for eyes closed and open)
            self.lpred = self.model.predict_classes(l_eye)
            break
    
        
        
        return self.rpred,self.lpred 