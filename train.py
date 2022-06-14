import cv2
import numpy as np
import face_recognition


imgrosh=face_recognition.load_image_file('imagebasic/roshan.jpeg')
imgrosh=cv2.cvtColor(imgrosh,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('imagebasic/roshan.jpeg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgrosh)[0]
encoderosh=face_recognition.face_encodings(imgrosh)[0]
cv2.rectangle(imgrosh,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encoderosh],encodetest)
facedis=face_recognition.face_distance([encoderosh],encodetest)
print(results,facedis)
cv2.putText(imgtest,f"{results} {round(facedis[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Roshan",imgrosh)
cv2.imshow("Roshantest",imgtest)
cv2.waitKey(0)