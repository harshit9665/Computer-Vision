# -*- coding: utf-8 -*-
"""
@author: diyasinha
"""
from datetime import datetime
import UserRecognition , EmoPlayer, DrowsinessDetector
import cv2
import os
from pygame import mixer
import warnings

webcam = cv2.VideoCapture(0)

#loading the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')       

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
f1 = cv2.FONT_HERSHEY_PLAIN
width, height  = 130, 100

name = []
start = datetime.now()
warnings.filterwarnings('ignore')
def get_data():
    """
    get_data: Gets data from the VideoCapture object and classifies them
    to a face or no face. 
    
    returns: tuple (faces in image, frame read, grayscale frame)
    """
    (f, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces, im , gray
    
def rec_user(model):
        """
        rec_user: Gets data from the get_data() and predict them using UserRecognition object. 
        result: predict the name of the user
        """
        start_time = datetime.now()
        time_elapsed = 0
        
        while True:
            faces, im, gray = get_data()
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x,y), (x+w, y+h) ,(255, 0 ,0), 2)
                face = gray[y:y+h , x:x+w]
                face_resize = cv2.resize(face, (width, height))
                prediction = model.recognize_user(face_resize)
                
                cv2.rectangle(im, (x,y), (x+w, y+h) ,(255, 0 ,0), 3)
                
                if prediction == "User not recognized":
                    cv2.putText(im, 'Not recognized', (10,460), font, 1,(255,255,255),1,cv2.LINE_AA)  
                else:                    
                    name.append(prediction)
                    cv2.putText(im, 'Hello % s ' % (prediction), (10,460), font, 1,(255,255,255),1,cv2.LINE_AA)
                    
                cv2.imshow('Vigilant Companion', im)
                time_elapsed = datetime.now() - start_time 

                #print('Time elapsed',format(time_elapsed))
            if str(time_elapsed) >= '0:00:05.000000':
                print('Time elapsed',format(time_elapsed))                
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
def emo_play(model):
    """
        emo_player: Gets data from the get_data() and predict the emotion using EmoPlayer object. 
    
        result: predict the emotion and play the song
        """
    predictions = []
    start_time = datetime.now()
    time_elapsed = 0
    while True:
        
        faces, im, gray = get_data()
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            pred= model.rec_emo(face)
            predictions.append(pred)
        
            cv2.putText(im, "User seems to be %s"%pred, (10,460), font, 1,(255,255,255),1,cv2.LINE_AA)
        
            cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0),3)
            time_elapsed = datetime.now() - start_time 

        if str(time_elapsed) >= '0:00:10.000000':
            print('Time elapsed',format(time_elapsed))                
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('Vigilant Companion', im)
    #webcam.release()
    cv2.destroyAllWindows()
    em = max(predictions, key=predictions.count)
    print("System thinks you're feeling %s" %em )
    model.play_songs(em)
    

def drowsy_detect(model):
    """
        drowsy_detect: Gets data from the get_data() and predict if driver is feeling drowsy or not using DrowsinessDetector object. 
    
        result: predict the drowsiness and increase the counter
        """
    
    path = os.getcwd()   
    score=0
    frame_width=2
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    while True:
        
        faces, im, gray = get_data()
        height,width = im.shape[:2]
        rpred,lpred = model.detector(faces, im, gray)
        cv2.rectangle(im, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
        
            
        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(im,"%s Drowsy(+)  "%prediction,(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
        # if(rpred[0]==1 or lpred[0]==1): eyes are open
        else:
            score=score-1
            cv2.putText(im,"%s Drowsy(-)  "%prediction,(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
            
        if(score<0):
            score=0  
        
        cv2.putText(im,'Counter:'+str(score),(220,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),im)
            try:
                sound.play()
                
                
            except:  # isplaying = False
                pass
            if(frame_width<16):
                frame_width += 2
            else:
                frame_width -= 2
                if(frame_width<2):
                    frame_width=2
            cv2.rectangle(im,(0,0),(width,height),(0,0,255),frame_width) 
        cv2.imshow('Vigilant Companion',im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #webcam.release()
    cv2.destroyAllWindows()
        
             
hour = datetime.now().hour

if hour < 12:
    greeting = "Good morning"
elif hour < 16:
    greeting = "Good afternoon"
elif hour < 20:
    greeting = "Good evening"
else:
    greeting = "Hope you had a good day"   
                   
model = UserRecognition.UserRecognition()
rec_user(model)  
if len(name) == 0:
    print("Not a specified User. Try again later.")
else:
    try:
        prediction = max(name, key=name.count)
        print("Hi {}! Let's go.".format(prediction+','+greeting))
        modelD = DrowsinessDetector.DrowsinessDetector()
        model = EmoPlayer.EmoPlayer()
        emo_play(model)
        drowsy_detect(modelD)
        start_time = datetime.now()
        te = 0
        '''while True:
               drowsy_detect(modelD)
               te = datetime.now() - start_time
               if str(te) >= '0:10:00.000000':
                   start_time = datetime.now()
                   emo_play(model)
               if str(datetime.now() - start_time) >= '1:00:00.000000': 
                   break'''
        
        
    except:
        webcam.release()
webcam.release()
    
