#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys

def main(arg):

    cam_index = 0
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
