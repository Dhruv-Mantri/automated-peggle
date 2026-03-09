import cv2
import numpy as np
import matplotlib.pyplot as plt
import mss # fast screenshots
import pygetwindow as gw

windows = gw.getAllTitles()

img_path = 'iso_ball.png'

peggle_window = gw.getWindowsWithTitle('Peggle Deluxe 1.01')[0]
left, top, width, height = peggle_window.left, peggle_window.top, peggle_window.width, peggle_window.height
print(width, height)


def current_state():
    with mss.mss() as sct:
        monitor = {
            "left": left + 150,
            "top": top + 100,
            "width": width,
            "height": height - 100
        }

        # circle detection
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def nothing(x):
    pass

def hsv_calc():
    # cap = current_state()
    cap = cv2.imread(img_path)
    cv2.namedWindow("Trackbars",)
    cv2.createTrackbar("lh","Trackbars",0,179,nothing)
    cv2.createTrackbar("ls","Trackbars",0,255,nothing)
    cv2.createTrackbar("lv","Trackbars",0,255,nothing)
    cv2.createTrackbar("uh","Trackbars",179,179,nothing)
    cv2.createTrackbar("us","Trackbars",255,255,nothing)
    cv2.createTrackbar("uv","Trackbars",255,255,nothing)
    while True:
        frame = cap
        #frame = cv2.imread('candy.jpg')
        #height, width = frame.shape[:2]
        #frame = cv2.resize(frame,(width/5, height/5), interpolation = cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos("lh","Trackbars")
        ls = cv2.getTrackbarPos("ls","Trackbars")
        lv = cv2.getTrackbarPos("lv","Trackbars")
        uh = cv2.getTrackbarPos("uh","Trackbars")
        us = cv2.getTrackbarPos("us","Trackbars")
        uv = cv2.getTrackbarPos("uv","Trackbars")

        l_blue = np.array([lh,ls,lv])
        u_blue = np.array([uh,us,uv])
        mask = cv2.inRange(hsv, l_blue, u_blue)
        result = cv2.bitwise_or(frame,frame,mask=mask)

        # cv2.imshow("frame",frame)
        cv2.imshow("mask",mask)
        cv2.imshow("result",result)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

hsv_calc()