import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

# from PIL import ImageGrab
csv_path  = 'Attendance.csv'
with open(csv_path, 'w') as f:
    f.write(f"Name, Status, Time, Day, Month, Year")

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    #print(encodeList)
    return encodeList
 
def markAttendance(name):
    with open(csv_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        #print(nameList)
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%I:%M %p')
            year = now.strftime('%Y')
            month = now.strftime('%B')
            day = now.strftime('%A')
            f.writelines(f'\n{name}, {"Yes"}, {dtString}, {day}, {month}, {year}')

 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        #print(matches)
        matchIndex = np.argmin(faceDis)
        #print(matches[matchIndex])
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
 
    cv2.imshow('Webcam', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        data = pd.read_csv(csv_path)
        #print(data)
        break
