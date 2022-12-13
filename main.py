import cv2
import time
import numpy as np

def setup_cam(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640 )
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


def setup_cam_cap_and_fill(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    #cap.set(cv2.CAP_PROP_BRIGHTNESS, 240) 
    # cap.set(cv2.CAP_PROP_CONTRAST, 255)
    _,startup=cap.read()
    cap.set(cv2.CAP_PROP_SATURATION,0)
    cap.set(cv2.CAP_PROP_AUTO_WB,1)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,5000)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    cap.set(cv2.CAP_PROP_EXPOSURE, 8) 
    #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
    return cap

def detected_shape(img,type):
    #type=0 <=> cap, type=1 <=> shape
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img=img[320:400,0:360]
    #fore = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,51,2)
    # fore = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    if type==1:
        img=img[320:400,0:360]
        fore = cv2.Canny(img,10,20)
    else:
        img=img[560:640,0:360]
        fore = cv2.Canny(img,30,80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (10,10))
    fore = cv2.morphologyEx(fore, cv2.MORPH_DILATE, kernel)
    # cv2.imshow("ia",fore)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
    # fore = cv2.morphologyEx(fore, cv2.MORPH_DILATE, kernel)
    
    fore[0:5,0:360]=255
    fore[75:80,0:360]=255
    cnts,_= cv2.findContours(fore,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        max_cnt=max(cnts, key=cv2.contourArea)
        cv2.drawContours(fore,[max_cnt],0,255,-1)
        cv2.imshow("h",fore)
        fore=fore[6:74]
        cnts,_= cv2.findContours(fore,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_cnt=max(cnts, key=cv2.contourArea)
        max_cnt=max(cnts, key=cv2.contourArea)
        area=cv2.contourArea(max_cnt)
        rect= cv2.boundingRect(max_cnt)
        print(area)
        x,y,w,h = rect
        # fore=cv2.cvtColor(fore,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(fore,(x,y),(x+w,y+h),(0,255,0),2)
        if (180>x+w/2>160) and area>10000 :
            
            cv2.imshow("detected",img)
            _,img=cam_2.read()
            img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("detected2",img)
        # cv2.imshow("contour",fore)
    except:
        pass

    
    

    
if __name__ == "__main__":

    cam_1=setup_cam(3)
    cam_2=setup_cam(1)
    cam_3=setup_cam_cap_and_fill(2)
    while True:
        start_time = time.time()
        _, frame_3 = cam_3.read()
        
        
        frame_3=cv2.rotate(frame_3, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_3=cv2.resize(frame_3,(360,640))
        #detected_shape(cv2.absdiff(frame_3[560:640,0:360],cv2.imread("ignore/bg.jpg")[560:640,0:360]))
        #detected_shape(frame_3[320:400,0:360])
        detected_shape(frame_3,0)
        FPS= 1.0 / (time.time() - start_time)
        cv2.putText(frame_3, str("%.2f" %FPS), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
        cv2.imshow("a",frame_3)
        # ESC pressed 0
        if cv2.waitKey(1)%256==27:
            break
        
    cam_1.release()
    cv2.destroyAllWindows()
