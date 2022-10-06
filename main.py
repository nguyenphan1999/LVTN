import cv2
import time
import numpy as np
from threading  import Thread


# def setup_cam(cam_id): 
#     # Set up Webcam
#     cap= cv2.VideoCapture(cam_id,cv2.CAP_V4L2)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     cap.set(cv2.CAP_PROP_AUTO_WB,0)
#     cap.set(cv2.CAP_PROP_WB_TEMPERATURE,4000)
#     cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
#     cap.set(cv2.CAP_PROP_EXPOSURE, 1000) 
#     #Notice when setting up webcam: CAP_V4L2, MJPG, CAP_PROP_FPS, CAP_PROP_EXPOSURE for optimal FPS in this order.
#     return cap

class cam:
    def __init__(self,cam_id):
        self.cam_id = cam_id 


    def start_running(self):
        self.capture=cv2.VideoCapture(self.cam_id,cv2.CAP_V4L2)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_AUTO_WB,0)
        self.capture.set(cv2.CAP_PROP_WB_TEMPERATURE,4000)
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 0:Auto, 1:Manual, 2:Shutter, 3:Aperture
        self.capture.set(cv2.CAP_PROP_EXPOSURE, 1000)  
        #_,self.frame = self.capture.read()
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start() 

    def update(self):
        while True:
            _,self.frame=self.capture.read()
    
    def getFrame(self):
        return self.frame

    def stop_and_release(self):
        # Kill the thread
        self.capture.release()
        self.thread.join()


def main():

    cam_1=cam(1)
    cam_2=cam(2)
    cam_1.start_running()
    cam_2.start_running()
    start_time = time.time()
    while True:
        try:
            
            frame_1=cam_1.getFrame()
            frame_2=cam_2.getFrame()
            FPS= 1.0 / (time.time() - start_time)
            start_time = time.time()
            frame_1 = np.concatenate((frame_1, frame_2), axis=1)
            cv2.putText(frame_1, str("%.2f" %FPS), (30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2,cv2.LINE_AA)
            
            cv2.imshow("a",frame_1)

        except:
            print("No frame")
                            # ESC pressed 
        if cv2.waitKey(1)%256==27:
            cam_1.stop_and_release()
            cam_2.stop_and_release()
            cv2.destroyAllWindows()
            break

    
if __name__ == "__main__":
    main()
