import cv2
import time
import numpy as np
import math

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
        return frame

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
    cap.set(cv2.CAP_PROP_SATURATION,0)
    return cap

def setup_cam_shape(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    _,startup=cap.read()

    cap.set(cv2.CAP_PROP_AUTO_WB,0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,10000)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    cap.set(cv2.CAP_PROP_EXPOSURE, 40) 
    #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
    return cap

def bottle_cap_inspection(img):

    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,mask= cv2.threshold(img_gray, 15,255, cv2.THRESH_BINARY_INV)

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
        cv2.imshow("edge",edge)
        
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

                if 550>x1>520 or 550>x2>520:
                    angle = math.atan2(x2- x1, y2 - y1)*(180/np.pi)
                    if 10>angle >-10:
                        return 1
                        
                
                if min(x1,x2)<min_x:
                    min_x=min(x1,x2)
                    min_line=line
    
            [y1,x1,y2,x2]=min_line[0]
            angle = math.atan2(x2- x1, y2 - y1)*(180/np.pi)
            cv2.line(img,(y1,x1),(y2,x2),(255,0,0),5)
            if angle >10 or angle<-10 or (x1+x2)/2 <450:
                return 1
            else:
                return 2
        except:
            pass
    
def bottle_fill_inspection(img):
    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,mask= cv2.threshold(img_gray, 10,255, cv2.THRESH_BINARY_INV)
    cv2.imshow("mask",mask)
    percent=np.sum(mask==255)/(img_gray.shape[0]*img_gray.shape[1])
    print(percent)
    if percent<0.1:
        return 0 #underfilled
    elif percent>0.4:
        return 1 #overfilled
    else: return 2 #good
    
    edge=cv2.Canny(mask,10,100)
    M = cv2.moments(mask)
    # calculate x,y coordinate of center
    try:
        cX = int(M["m10"] / M["m00"])
        roi = edge[0:30,cX-20:cX+20]
        M = cv2.moments(roi)
        cY = int(M["m01"] / M["m00"])
        if cY <img_gray.shape[0]*0.5:
            return 1 #overfilled
        return 2 #good
    except:
        return 1

def main():
    global frame

    cam_1=setup_cam_cap_and_fill(1)
    #cam_1=setup_cam_label_and_shape(1)

    while True:
        start_time = time.time()

        _, frame_1 = cam_1.read()
        frame_1= cv2.rotate(frame_1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.setMouseCallback("a", click_event)
        #bottle_cap= bottle_cap_inspection(cv2.resize(frame_1,(360,640))[0:160,0:360])
        frame= cv2.resize(frame_1, (360,640))

        bottle_cap=bottle_cap_inspection(frame_1[0:650,0:720])
        bottle_fill=bottle_fill_inspection(cv2.resize(frame_1,(360,640))[430:460,0:360])
        
        if bottle_cap==0:
            cv2.putText(frame, "No cap", (10,600),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
        elif bottle_cap==1:
            cv2.putText(frame, "Defect cap", (10,600),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
        elif bottle_cap==2:
            cv2.putText(frame, "Good cap", (10,600),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)

        if bottle_fill==0:
            cv2.putText(frame, "Underfilled", (10,630),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,cv2.LINE_AA)
        elif bottle_fill==1:
            cv2.putText(frame, "Overfilled", (10,630),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,cv2.LINE_AA)
        elif bottle_fill==2:
            cv2.putText(frame, "Good fill", (10,630),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,cv2.LINE_AA)
        FPS= 1.0 / (time.time() - start_time)
        cv2.putText(frame, str("%.2f" %FPS), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)

       
        cv2.imshow("a",frame)

        # ESC pressed 
        k= cv2.waitKey(1)
        if k%256==27:
            break

        
    # cam_1.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
