import tensorflow.python.keras
from PIL import Image, ImageOps
import tensorflow
import cv2
import numpy as np
import datetime
import csv
import notify

notification = True 

fieldnames = ['Date', 'Time', 'Status'] #head ของ ไฟล์ excel วัน/เวลา/สถานะ
time = datetime.datetime.now() #เวลา
f = open('detected.csv', 'w',newline='') #สร้างไฟล์ detected.csv เป็นค่า write, ไม่เว้นบรรทัด
writer = csv.DictWriter(f, fieldnames=fieldnames) #สร้างตัวเเปร writer เป็นตัวเขียนไฟล์
writer.writeheader() #สร้าง head ของ ไฟล์

face_cascade= cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

webcam = cv2.VideoCapture(0) #ใช้ cv2 เปิดเว็ปแคมลงในตัวเเปร webcam


face_classifier = cv2.CascadeClassifier(face_cascade)
np.set_printoptions(suppress=True)        
model = tensorflow.keras.models.load_model('keras_model.h5',compile=False) # ใช้ model keras
size = (224, 224)

while True:
  success, image_bgr = webcam.read() # เก็บค่าว่าสำเร็จหรือไม่ใน success ค่าสีไว้ใน image_bgr
  image_org = image_bgr.copy()
  image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)# รูปขาวดำ
  image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)# รูปสี
  faces = face_classifier.detectMultiScale(image_bw)#ใช้ภาพขาวดำ detect หน้า
  for face in faces:# loop
    x, y, w, h = face
    cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = cface_rgb
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data) #array บอกความถูกต้อง
    print(prediction)
    if prediction[0][0] > prediction[0][1]:
      rows = [{"Date" : time.strftime("%D"), 'Time' : time.strftime("%T"), "Status" : "Detected"}] #ข้อมูลถ้าใส่เเมส

      cv2.putText(image_bgr,'Mask',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2) #Mask เขียว 
      cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2) #กรอบเขียว Mask
      writer.writerows(rows) #บันทึกข้อมูลถ้าใส่เเมส

    else:
      rows = [{"Date" : time.strftime("%D"), 'Time' : time.strftime("%T"), "Status" : "Undetected"}] #ข้อมูลถ้าไม่ใส่เเมส

      cv2.putText(image_bgr,'UnMask',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2) #Unmask เเดง
      cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0,0,255), 2)#กรอบเเดง Mask
      writer.writerows(rows) #บันทึกข้อมูลถ้าไม่ใส่เเมส
      notify.lineNotify('Mask is not detected')

  cv2.imshow("AI Mask Protector", image_bgr) #โชว์โปรเเกรมขึ้นมาชื่อ AI Mask Protector
  cv2.waitKey(1)