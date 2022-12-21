import cv2
import time
import numpy as np
import math
import subprocess
from tkinter import *
import tkinter as tk
from multiprocessing import *
from PIL import ImageTk, Image
import ctypes
from time import sleep



class Yolov4:
    def __init__(self, cfg_dir, weights_dir, names_dir):
        self.cfg_dir = cfg_dir
        self.weights_dir = weights_dir
        self.names_dir = names_dir
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_dir,self.weights_dir)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.classes = []
        with open(names_dir, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    def detect(self,frame):
        Conf_threshold = 0
        NMS_threshold = 0.45
        COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        classes, scores, boxes = self.model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (self.classes[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10),cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        if len(classes)==0:
            return frame,1
        else:
            return frame,0

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print('Coorinate:',y, ' ', x)

 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        try: 
            b = frame[y, x,0]
            g = frame[y, x,1]
            r = frame[y, x,2]
            print(b,' ',g,' ',r)
        except:
            print(frame[y, x])

def setup_cam_cap_and_fill(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    _,startup=cap.read()
    cap.set(cv2.CAP_PROP_SATURATION,0)
    cap.set(cv2.CAP_PROP_AUTO_WB,0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,5000)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    cap.set(cv2.CAP_PROP_EXPOSURE, 5) 
    return cap

def setup_cam_shape(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    _,startup=cap.read()
    #print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
    cap.set(cv2.CAP_PROP_AUTO_WB,0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,10000)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    cap.set(cv2.CAP_PROP_EXPOSURE, 40) 
    #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
    return cap

def bottle_cap_inspection(img):
    img=img[0:650,0:720]
    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,mask= cv2.threshold(img_gray, 20,255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    contours=sorted(contours, key=cv2.contourArea, reverse=True)
    #cap=23000
    if len(contours)==0 or cv2.contourArea(contours[0])<10000 :
        #print(cv2.contourArea(contours[0]))
        return 0
    else:

        edge= cv2.Canny(mask,30,100)
        kernel = np.ones((3, 3), np.uint8)
        edge= cv2.dilate(edge,kernel, iterations=1)
        #cv2.imshow("edge",edge)
        
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20  # minimum number of pixels making up a line
        max_line_gap =  10 # maximum gap in pixels between connectable line segments
        lines = cv2.HoughLinesP(edge, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
        
        try:

            #print("first:",lines)
            min_x=lines[0][0][1]
            min_line=lines[0]
            for line in lines:
                [y1,x1,y2,x2]=line[0]

                if 560>x1>500 or 560>x2>500:
                    angle = math.atan2(x2- x1, y2 - y1)*(180/np.pi)
                    if 10>angle >-10:
                        return 1
                        
                
                if min(x1,x2)<min_x:
                    min_x=min(x1,x2)
                    min_line=line
    
            [y1,x1,y2,x2]=min_line[0]
            angle = math.atan2(x2- x1, y2 - y1)*(180/np.pi)
            #cv2.line(img,(y1,x1),(y2,x2),(255,0,0),5)
            if angle >10 or angle<-10 or (x1+x2)/2 <430:
                return 1
            else:
                return 2
        except:
            pass
    
def bottle_fill_inspection(img):
    img=cv2.resize(img,(360,640))[330:640,170:190]
    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,mask= cv2.threshold(img_gray, 10,255, cv2.THRESH_BINARY_INV)
    cv2.imshow("mask",mask)
    percent=np.sum(mask==255)/(img_gray.shape[0]*img_gray.shape[1])
    print(percent)
    if percent<0.5:
        return 0 #underfilled
        
    elif percent>0.65:
        return 1 #overfilled
    else: return 2 #good

def detected(img,type):
    #type=0 <=> cap, type=1 <=> shape
    
    if type==1:
        img=img[320:400,0:360]
        fore = cv2.Canny(img,20,50)
    elif type==0:
        img=cv2.resize(img,(360,640))
        img=img[560:640,0:360]
        fore = cv2.Canny(img,30,80)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (10,10))
    fore = cv2.morphologyEx(fore, cv2.MORPH_DILATE, kernel)   
    fore[0:5,0:360]=255
    fore[75:80,0:360]=255
    cnts,_= cv2.findContours(fore,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        max_cnt=max(cnts, key=cv2.contourArea)
        cv2.drawContours(fore,[max_cnt],0,255,-1)
        #cv2.imshow("h",fore)
        fore=fore[6:74]
        cnts,_= cv2.findContours(fore,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_cnt=max(cnts, key=cv2.contourArea)
        max_cnt=max(cnts, key=cv2.contourArea)
        area=cv2.contourArea(max_cnt)
        rect= cv2.boundingRect(max_cnt)
        
        x,y,w,h = rect
        # fore=cv2.cvtColor(fore,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(fore,(x,y),(x+w,y+h),(0,255,0),2)
        if (180>(x+w/2)>160) and area>8000 :
            print(area)

            #cv2.imshow("detected",img_original)
            # _,imgh=cam_3.read()
            # imgh=cv2.rotate(imgh, cv2.ROTATE_90_CLOCKWISE)
            return 0
            # cv2.imshow("detected2",imgh)
        else:
            return 1
        # cv2.imshow("contour",fore)
    except:
        return 1

def shape():
    
    cam_1 = setup_cam_shape(0)
    cam_2 = setup_cam_shape(1)
    model = Yolov4(weights_dir='yolov4-custom_last.weights',
                   cfg_dir='yolov4-custom.cfg',
                   names_dir='yolo.names')
    print("aa")
    _, frame_1 = cam_1.read()
    _, frame_2 = cam_2.read()
    frame_1,_=model.detect(frame_1)
    frame_2,_=model.detect(frame_2)
    start_stop[0]=1
    k=0
    while(start_stop[0]==1):
        
        _, frame_1 = cam_1.read()
        _, frame_2 = cam_2.read()
        
        frame_1= cv2.rotate(frame_1, cv2.ROTATE_90_CLOCKWISE)
        frame_2= cv2.rotate(frame_2, cv2.ROTATE_90_CLOCKWISE)
       
        detec=detected(frame_1,1)
        if detec==0:
            
            start_time=time.time()
            frame_1,classes_1=model.detect(frame_1)
            frame_2,classes_2=model.detect(frame_2)
            print(classes_1, classes_2)
            if classes_1==1 and classes_2==1:
                result.value=int(str(order[k])+str(1))
            else:
                result.value=int(str(order[k])+str(0))
            if k==9:
                k=0
            else: k=k+1
            print("finised")
 
            b_1[:]=frame_1
            b_2[:]=frame_2
            frame_1=cam_1.read() #to flush out lastest frame
            
            print(time.time()-start_time)
    cam_1.release()
    cam_2.release()
def cap_and_fill():
    cam_3 = setup_cam_cap_and_fill(2)
    
    while(start_stop[0]==0):
        pass

    k=0
    while(start_stop[0]==1):
        print("cc")
        _, frame_3 = cam_3.read()

        frame_3= cv2.rotate(frame_3, cv2.ROTATE_90_COUNTERCLOCKWISE)
              
        detec=detected(frame_3,0)
        if detec==0:
            bottle_cap=bottle_cap_inspection(frame_3)
            bottle_fill=bottle_fill_inspection(frame_3)

            order[k]=int(str(bottle_cap) + str(bottle_fill))
            if k==9:
                k=0
            else: k=k+1

            if bottle_cap==0:
                cv2.putText(frame_3, "No cap", (10,600),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
            elif bottle_cap==1:
                cv2.putText(frame_3, "Defect cap", (10,600),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
            elif bottle_cap==2:
                cv2.putText(frame_3, "Good cap", (10,600),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)

            if bottle_fill==0:
                cv2.putText(frame_3, "Underfilled", (10,630),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,cv2.LINE_AA)
            elif bottle_fill==1:
                cv2.putText(frame_3, "Overfilled", (10,630),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,cv2.LINE_AA)
            elif bottle_fill==2:
                cv2.putText(frame_3, "Good fill", (10,630),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,cv2.LINE_AA)
            frame_3=cv2.resize(frame_3,(360,640))  
            # cv2.imshow("cap_fill",frame_3)
            b_3[:]=frame_3
            
            frame_3=cam_3.read()
    cam_3.release()
def run():
    
    if start_stop[0]==0:
        btn.config(text="Starting...")
        btn["state"]="disabled"
        detect_shape= Process(target=shape,args=())
        detect_cap_and_fill=Process(target=cap_and_fill,args=())
        detect_shape.daemon = True
        detect_cap_and_fill.daemon=True
        detect_shape.start()
        detect_cap_and_fill.start()
        # detect_shape.join()
        # detect_cap_and_fill.join()
    elif start_stop[0]==1:
        start_stop[0]=0
        btn.config(text="Start")

def update():
    
    if start_stop[0]==1:
        btn.config(text="Stop")
        btn["state"]="active"
    #frame= cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #frame=cv2.resize(frame,(640,360))
    cv2image_1=cv2.cvtColor(b_1,cv2.COLOR_BGR2RGBA)
    imgtk_1=ImageTk.PhotoImage(image=Image.fromarray(cv2image_1))
    windows.one=imgtk_1
    canvas_1.create_image((0,0),image=imgtk_1,anchor=NW)

    cv2image_2=cv2.cvtColor(b_2,cv2.COLOR_BGR2RGBA)
    imgtk_2=ImageTk.PhotoImage(image=Image.fromarray(cv2image_2))
    windows.two=imgtk_2
    canvas_2.create_image((0,0),image=imgtk_2,anchor=NW)

    cv2image_3=cv2.cvtColor(b_3,cv2.COLOR_BGR2RGBA)
    imgtk_3=ImageTk.PhotoImage(image=Image.fromarray(cv2image_3))
    windows.three=imgtk_3
    canvas_3.create_image((0,0),image=imgtk_3,anchor=NW)

    text.delete('1.0', END)
    texts=""
    if result.value!=9:
        if str(result.value)[0]=="0":
            texts=texts+"No cap,"
        elif str(result.value)[0]=="1":
            texts=texts+"Defect cap,"
        else:
            texts=texts+"Good cap,"

        if str(result.value)[1]=="0":
            texts=texts+"Underfilled,"
        elif str(result.value)[1]=="1":
            texts=texts+"Overfilled,"
        else:
            texts=texts+"Good filled,"

        if str(result.value)[2]=="0":
            texts=texts+"Bad shape"
        else:
            texts=texts+"Good shape"
    
    text.insert(END, texts)
    windows.after(10,update)
    
    
    
if __name__ == "__main__":
    windows= Tk()
    windows.title("Program")
    windows.geometry("1024x768")
    
    canvas_1= Canvas(windows,width=340,height=640,bg="white")
    canvas_2= Canvas(windows,width=340,height=640,bg="white")
    canvas_3= Canvas(windows,width=340,height=640,bg="white")
    canvas_1.grid(column=1,row=0)
    canvas_2.grid(column=2,row=0)
    canvas_3.grid(column=0,row=0)

    #canvas_1.pack()
    
    shared_1=Array(ctypes.c_uint8,640*360*3,lock=False)
    b_1=np.frombuffer(shared_1, dtype=np.uint8)
    b_1=b_1.reshape((640,360,3))
    shared_2=Array(ctypes.c_uint8,640*360*3,lock=False)
    b_2=np.frombuffer(shared_2, dtype=np.uint8)
    b_2=b_2.reshape((640,360,3))
    shared_3=Array(ctypes.c_uint8,640*360*3,lock=False)
    b_3=np.frombuffer(shared_3, dtype=np.uint8)
    b_3=b_3.reshape((640,360,3))
    
    


    
    result=Value(ctypes.c_uint8,lock=False)
    result.value=9
    text= Text(windows,width=50,height=2)
    text.grid(column=1,row=1)
    # text.insert(END, result.value)



    btn= Button(windows, text="Start",command=run)
    btn.grid(column=1,row=2)
    
    startstop=Array(ctypes.c_uint8,1,lock=False)
    start_stop=np.frombuffer(startstop, dtype=np.uint8)
    start_stop[0]=0

    order=Array(ctypes.c_uint8,20,lock=False)
    order=np.frombuffer(order, dtype=np.uint8)

    update()
    windows.mainloop()
