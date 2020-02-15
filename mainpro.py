import cv2
#car_cascade=cv2.CascadeClassifier('cars.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
sampleNum=0
Id=input("enter your id")
while 1:  
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])


        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
  
        eyes = eye_cascade.detectMultiScale(roi_gray)  
        for (ex,ey,ew,eh) in eyes: 
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        #car= car_cascade.detectMultiScale(gray, 1.1, 6)
    #for (cx,cy,cw,ch) in car: 
     #   cv2.rectangle(img,(cx,cy),(cx+cw,cy+ch),(0,255,255),2)  
       
            

    cv2.imshow('img',img) 
    k = cv2.waitKey(2) & 0xff
    if k == 27: 
        break 
cap.release() 
cv2.destroyAllWindows()

