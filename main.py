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
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 200) 
    # cap.set(cv2.CAP_PROP_CONTRAST, 255)
    cap.set(cv2.CAP_PROP_SATURATION,0)
    cap.set(cv2.CAP_PROP_AUTO_WB,0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,5000)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,3) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    #cap.set(cv2.CAP_PROP_EXPOSURE, 1000) 
    #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
    return cap
def bottle_cap_inspection(img):
 
    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    _,mask= cv2.threshold(img_gray, 70,255, cv2.THRESH_BINARY)
    mask=~mask

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('mask',mask)
        
    img[np.where(mask==0)] = 255

    # edge= cv2.Canny(mask,50,100)
    # # cv2.imshow("edge",edge)
    # rho = 1  # distance resolution in pixels of the Hough grid
    # theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    # min_line_length = 30  # minimum number of pixels making up a line
    # max_line_gap = 20  # maximum gap in pixels between connectable line segments
    # line_image = np.copy(img) * 0  # creating a blank to draw lines on
    # lines = cv2.HoughLinesP(edge, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    # try:
    #     for line in lines:
    #         print(line)
    #         print("a")
    #         for x1,y1,x2,y2 in line:
    #             cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)
    # except:
    #     pass
    
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    contours=sorted(contours, key=cv2.contourArea, reverse=True)
    try:
        if cv2.contourArea(contours[0])<15000:
            print("No cap")
        else:
            
                # if len(contours)>1:
                #     for i in range(2):
                #         rect = cv2.minAreaRect(contours[i])
                #         print(rect[2])
                #         box = cv2.boxPoints(rect)
                #         box = np.int0(box)
                #         cv2.drawContours(img,[box],0,(127,255,0),2)
                #         #print(i,':',cv2.contourArea(contours[i]))
                # else:
                #print(cv2.contourArea(contours[0]))
                rect = cv2.minAreaRect(contours[0])
                
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                print(cv2.contourArea(contours[0]))
                print(np.sum(mask!=0))
                print((box[0][1]+box[2][1])/2)
                cv2.drawContours(img,[box],0,(127,255,0),2)
                if rect[2]>-95 and rect[2]<-5:
                    print("Crooked cap")
                elif ((box[0][1]+box[2][1])/2)<195 or ((box[0][1]+box[2][1])/2)> 220:
                    print("Crooked cap")
                else:
                    print("Good cap")
                
    except:
        pass
    return img

def main():
    
    cam_1=setup_cam(0)
    global frame_1

    
    while True:
        start_time = time.time()

        _, frame_1 = cam_1.read()
        frame_1= cv2.rotate(frame_1, cv2.ROTATE_90_CLOCKWISE)
        frame= cv2.resize(frame_1, (360,640))
        
        cv2.imshow("b",frame)
        
        
        #frame_1= bottle_cap_inspection(cv2.resize(frame_1,(360,640))[0:160,0:360])
        frame_1=bottle_cap_inspection(frame_1[0:290,0:720])
        FPS= 1.0 / (time.time() - start_time)
        cv2.putText(frame_1, str("%.2f" %FPS), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
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
