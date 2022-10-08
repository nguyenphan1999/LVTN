import cv2
import time


def setup_cam(cam_id): 
    # Set up Webcam
    cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_WB,0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE,6000)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
    cap.set(cv2.CAP_PROP_EXPOSURE, 1000) 
    #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
    return cap
def bottle_cap_inspection(img):
    img=img

def main():

    cam_1=setup_cam(1)
    
    while True:
        start_time = time.time()
        _, frame_1 = cam_1.read()
        FPS= 1.0 / (time.time() - start_time)
        cv2.putText(frame_1, str("%.2f" %FPS), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
        
        cv2.imshow("a",frame_1)
        
        # ESC pressed 
        if cv2.waitKey(1)%256==27:
            break
        
    cam_1.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
