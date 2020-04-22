# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:13:42 2019
Vigilant Companion System 
Version: 1.0
@developers: DiyaSinha , HarshitTripathi , HarshitSingh

Purpose : This is for the developer to create the dataset for a new user.
"""
import cv2, os 

name = input('Input your name : ')

path = r"C:\Users\cutyp\.spyder-py3\FinalYearProject\userData\\" + name

if not os.path.isdir(path): 
    os.mkdir(path) 
  
# defining the size of images  
(width, height) = (130, 100)     
  
#the cascade use to detect the face 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

webcam = cv2.VideoCapture(0)  
k = 1 
# The program loops until it has 100 images of the face. 
count = 1
while count <= 250:  
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height)) 
        cv2.imwrite('% s/% s.png' % (path, k), face_resize) 
    k += 1
    count += 1
      
    cv2.imshow('VigilantCompanion', im) 
    key = cv2.waitKey(50) 
    
webcam.release()
cv2.destroyAllWindows()
