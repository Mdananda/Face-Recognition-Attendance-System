# importing librarys
import cv2
import numpy as np
import face_recognition

# function
def resize(img, size) :
    height = int(img.shape[0] * size)
    width = int(img.shape[1]*size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
Img = face_recognition.load_image_file('Jishan.jpg')
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
Img = resize(Img, 0.50)

Img_test = face_recognition.load_image_file('Jishan_test.jpg')
Img_test = resize(Img_test, 0.50)
Img_test = cv2.cvtColor(Img_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_Img = face_recognition.face_locations(Img)[0]
encode_Img = face_recognition.face_encodings(Img)[0]
cv2.rectangle(Img, (faceLocation_Img[3], faceLocation_Img[0]), (faceLocation_Img[1], faceLocation_Img[2]), (255, 0, 255), 3)


faceLocation_Imgtest = face_recognition.face_locations(Img_test)[0]
encode_Imgtest = face_recognition.face_encodings(Img_test)[0]
cv2.rectangle(Img_test, (faceLocation_Img[3], faceLocation_Img[0]), (faceLocation_Img[1], faceLocation_Img[2]), (255, 0, 255), 3)

results = face_recognition.compare_faces([encode_Img], encode_Imgtest)
print(results)
cv2.putText(Img_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', Img)
cv2.imshow('test_img', Img_test)
cv2.waitKey(0) 
cv2.destroyAllWindows()