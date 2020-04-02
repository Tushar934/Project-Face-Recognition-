import cv2
import numpy as np
import os

#load face classifier
face_classifier=cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
#load functions
def face_extractor(img):
    #function detects the faces and returned the cropped images
    #if no faced detected it returs the input images
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.1,5)
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cropped_faces=img[y:y+h,x:x+w]
    return cropped_faces

#intialize web cam
cap=cv2.VideoCapture(0)
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        #Save file in specified directory with unique name
        file_name_path='./images/face'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        #put count on images and display live count
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_4)
        
        cv2.imshow('Face Cropper',face)
    else:
        print('Face not Found')
        pass
    if cv2.waitKey(1)==13 or count==50: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")