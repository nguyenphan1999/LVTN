import cv2
import time


def setup_cam(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280 )
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_WB,0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,4000)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    cap.set(cv2.CAP_PROP_EXPOSURE, 1000) 
    #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
    return cap
def bottle_cap_inspection(cap):
    cam=cap


def detected (img,fgbg):

    # gpu_img = cv2.cuda_GpuMat()
    # gpu_img.upload(img)

    # fgmask_gpu = fgbg.apply(gpu_img,-1,cuda_stream_0) 
    # fgmask=fgmask_gpu.download()
    # thresh=fgmask

    #Detect using the middle part of the bottle
    thresh=fgbg.apply(img)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts,_= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    for c in cnts:
        rect= cv2.boundingRect(c)
        area= cv2.contourArea(c)
        x,y,w,h = rect
        if area>15000:
            
            cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("1",thresh)
    return 1, thresh
def main():
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500)
    fgbg.setDetectShadows(False) 
    cam_1=setup_cam(0)
    
    while True:
        start_time = time.time()
        _, frame_1 = cam_1.read()
        frame_1=cv2.rotate(frame_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    
        detect, frame_1=detected(cv2.resize(frame_1,(360,640))[256:384,0:360],fgbg)

        FPS= 1.0 / (time.time() - start_time)
        cv2.putText(frame_1, str("%.2f" %FPS), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
        cv2.imshow("a",cv2.resize(frame_1,(300,535)))
        print(cam_1.get(cv2.CAP_PROP_FPS))
        # ESC pressed 
        if cv2.waitKey(1)%256==27:
            break
        
    cam_1.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
