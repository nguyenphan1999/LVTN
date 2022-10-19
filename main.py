import cv2
import time
import numpy as np

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print('Coorinate:',y, ' ', x)

 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        
        # b = frame_1[y, x,0]
        # g = frame_1[y, x,1]
        # r = frame_1[y, x,2]
        # print(b,' ',g,' ',r)

        print(frame_1[y, x])


def setup_cam(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_WB,0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,4500)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,3) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    #cap.set(cv2.CAP_PROP_EXPOSURE, 1000) 
    #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
    return cap
def bottle_cap_inspection(img, type):
    # gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray, 150,255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh",thresh)
    # print(np.sum(thresh==0))
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if type==1:
        # lower mask (0-10)
        lower_red = np.array([0,50,100])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,50,100])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask = mask0+mask1
        
    else:
        img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,mask= cv2.threshold(img_gray, 20,255, cv2.THRESH_BINARY)
        mask=~mask
        #   black mask
        # lower_black=np.array([0,0,0])
        # upper_black=np.array([180,255,150])

        # mask = cv2.inRange(img_hsv, lower_black, upper_black)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('mask',mask)
        
    img[np.where(mask==0)] = 255

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours=sorted(contours, key=cv2.contourArea, reverse=True)
    try:
        if len(contours)>1:
            for i in range(2):
                rect = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img,[box],0,(127,255,0),2)
                #print(i,':',cv2.contourArea(contours[i]))
        else:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(127,255,0),2)
            #print(cv2.contourArea(contours[0]))
    except:
        pass
    return img

def main():
    global frame_1
    cam_1=setup_cam(0)
    

    
    while True:
        # start_time = time.time()

        _, frame_1 = cam_1.read()
        frame_1= cv2.rotate(frame_1, cv2.ROTATE_90_CLOCKWISE)
        frame= cv2.resize(frame_1, (360,640))
        
        cv2.imshow("b",frame)
        
        # FPS= 1.0 / (time.time() - start_time)
        # cv2.putText(frame_1, str("%.2f" %FPS), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
        frame_1= bottle_cap_inspection(frame_1[0:300,0:720],2)
        cv2.imshow("a",frame_1)
        cv2.setMouseCallback("a", click_event)
        # ESC pressed 
        k= cv2.waitKey(1)
        if k%256==27:
            break
        elif k == ord('t'):
            cv2.imwrite("image/cap.jpg", frame_1)
            break
        
    # cam_1.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
