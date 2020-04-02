import cv2
import numpy as np
import webbrowser
import os
from Training import tushar_model
face_classifier=cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
def face_detector(img):
    #function detects the faces and returned the cropped images
    #if no faced detected it returs the input images
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.1,5)
    if faces is ():
        return img,[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    image,face=face_detector(frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        #pass face to prediction model
        #results comprises of a tuple containing the label and the
        results=tushar_model.predict(face)
        print(results)
        if results[1]<500:
            confidence=int(100*(1-(results[1])/400))
            display_string=str(confidence)+'% Confident, It is user'
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        if confidence>85:
            cv2.putText(image, 'Hey', (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
            cv2.imshow('Face Recognition',image)
        else:
            cv2.putText(image,'Locked', (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
            cv2.imshow('Face Recognition',image)
    except:
        #cv2.putText(image, 'No   Face   Found', (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
        #cv2.putText(image, 'Locked', (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
        #cv2.imshow('Face Recognition', image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()
