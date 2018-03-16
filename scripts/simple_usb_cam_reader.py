#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
from config import Config

def main(arg):

    list_cam = Config.get_usb_cam()
    if list_cam is None or len(list_cam) == 0:
        print "Found no attached usb cameras"
        return

    cam_index = int(list_cam[0][-1])
    argc = len(arg)
    if (argc > 1):
        cam_index = int(arg[1])

    cap = cv2.VideoCapture(cam_index)
    while(True):
        ret, frame = cap.read()
        cv2.imshow("current_frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
