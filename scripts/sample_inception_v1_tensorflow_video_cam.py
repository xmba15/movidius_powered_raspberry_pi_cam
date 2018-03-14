#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from mvnc import mvncapi as mvnc
from config import Config

# check Movidius NCS device
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()
device = mvnc.Device(devices[0])
device.OpenDevice()

def preprocessing_image(img, reqsize):

    mean = 128
    std = 1/128

    img = img.astype(np.float32)
    dx,dy,dz = img.shape
    delta = float(abs(dy-dx))
    if dx > dy: #crop the x dimension
        img = img[int(0.5 * delta) : dx - int(0.5 * delta), 0 : dy]
    else:
        img = img[0 : dx, int(0.5 * delta) : dy - int(0.5 * delta)]
    img = cv2.resize(img, (reqsize, reqsize))

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean) * std

    return img

# load graph
graph_filename = "tensorflow_inception_v1_graph"
categories_filename = "imagenet_categories.txt"

with open(os.path.join(Config.model_dir, graph_filename), mode = "rb") as f:
    graphfile = f.read()
graph = device.AllocateGraph(graphfile)

# load labels
categories = []
with open(os.path.join(Config.model_dir, categories_filename), 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    print ("Number of categories:", len(categories))

# load images
list_cam = Config.get_usb_cam()

if len(list_cam) != 0:
    cam_index = int(list_cam[0][-1])
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FPS, 1)
    ret, frame = cap.read()
    print ("Start streaming")

    while (ret):
        img = preprocessing_image(frame, Config.inception_v1_image_size)
        graph.LoadTensor(img.astype(np.float16), 'user object')
        output, userobj = graph.GetResult()
        output_result = categories[output.argsort()[::-1][0]]
        # graph.DeallocateGraph()
        print (output_result)
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

graph.DeallocateGraph()
device.CloseDevice()
