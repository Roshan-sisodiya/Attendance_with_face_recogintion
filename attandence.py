import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pywhatkit as pwt
from pywhatkit.remotekit import start_server

path="imageattendance"
images=[]
classname=[]
mylist=os.listdir(path)
print(mylist)
for cls in mylist:
    curimg=cv2.imread(f"{path}/{cls}")
    images.append(curimg)
    classname.append(os.path.splitext(cls)[0])
print(classname)

def findencoding(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open("attandance.csv","r+") as f:
        mydatalist=f.readlines()
        namelist=[]

        for line in mydatalist:
            entry =line.split(",")
            namelist.append(entry[0])

        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')

            f.writelines(f'\n{name},{dtstring}')


encodelistknown=findencoding(images)


cap=cv2.VideoCapture(0)
while True:
    sucess,img =cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    facescurframe = face_recognition.face_locations(imgs)
    encodescurframe=face_recognition.face_encodings(imgs,facescurframe)

    for encodeface,faceloc in zip(encodescurframe,facescurframe):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedis=face_recognition.face_distance(encodelistknown,encodeface)
        #print(facedis)
        matcheindex=np.argmin(facedis)

        if matches[matcheindex]:
            name=classname[matcheindex].upper()
            #print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)
            #sendmsg(name)

    cv2.imshow("webcame",img)
    cv2.waitKey(1)



#faceloc=face_recognition.face_locations(imgrosh)[0]
#encoderosh=face_recognition.face_encodings(imgrosh)[0]
#cv2.rectangle(imgrosh,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

#faceloctest=face_recognition.face_locations(imgtest)[0]
#encodetest=face_recognition.face_encodings(imgtest)[0]
#cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

#results=face_recognition.compare_faces([encoderosh],encodetest)
#facedis=face_recognition.face_distance([encoderosh],encodetest)