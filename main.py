import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import gtts
import playsound
import os
from playsound import playsound
import random



net = cv2.dnn.readNetFromDarknet("yolov8.cfg","yolov8.weights")


class_ = None

classes = open("coco.names").read().strip().split("\n")


cap = cv2.VideoCapture(0)

while 1:
    _, img = cap.read()
    img = cv2.resize(img,(1280,720))
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)

                x = int(center_x - w/2)
                y = int(center_y - h/2)



                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])

            #creating a super random named file

            r1 = random.randint(1,100000000000)
            r2 = random.randint(1,100000000000)
            randfile = str(r2)+"randomtext"+str(r1) +".mp3"
            
            myobj = gtts.gTTS(text=label, lang='en', slow=False)
            myobj.save(randfile)
            time.sleep(2)
            print(label)
  
            playsound(randfile)
            

            if label=='person':
                print(label)
             

            if label=='mobile':
                print(label)
            

            if label=='pen':
                print(label)
               

            if label=='tvmonitor':
                print(label)
                

            if label=='tie':
                print(label)
               
            if label=='cup':
                print(label)
               

            if label=='cell phone':
                print(label)
               

            if label=='mouse':
                print(label)
               

            if label=='bottle':
                print(label)
                
            
            if label=='laptop':
                print(label)
               
                        
            if label=='chair':
                print(label)
               

                        
            if label=='book':
                print(label)
               
           
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
