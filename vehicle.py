import cv2;
import numpy as np;

#capturing the frames
cap=cv2.VideoCapture("video.mp4")

count_line_position=550#when crossed by vehicles it is counted
min_width=80
min_height=80

#initialize substructor algorithm
algo=cv2.createBackgroundSubtractorMOG2()#isolates the vehicle and removes the background

def center_point(x,y,w,h):#detects an image from its centre
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

vehicle_count=[]#vehicle count
offset=5
counter=0

while True:
    ret,frame1=cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)#making it monochromatic
    blur=cv2.GaussianBlur(grey, (3,3), 5)#smoothen the image
    #apply on each frame
    img_sub=algo.apply(blur)#apply on blur
    dilat=cv2.dilate(img_sub,np.ones((5,5)))#black and white image
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))#to get elliptical objects
    dilatdata=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatdata=cv2.morphologyEx(dilatdata,cv2.MORPH_CLOSE,kernel)
    counter_shape,h=cv2.findContours(dilatdata,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#outlines the boundary of object

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)#drawing the counter line

    for i,c in enumerate(counter_shape):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter=(w>=min_width and h>=min_height)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame1,str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,2,(0,255,0),5)
        center=center_point(x,y,w,h)
        vehicle_count.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)#center of out image

        for x,y in vehicle_count:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                vehicle_count.remove((x,y))#after vehicle crosses the line remove it from the list
                print("vehicle counter: "+str(counter))
    
    cv2.putText(frame1,"COUNTER: "+str(counter),(450,70),cv2.FONT_HERSHEY_TRIPLEX,2,(0,255,0),5)

    cv2.imshow('video',frame1)
    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release()

