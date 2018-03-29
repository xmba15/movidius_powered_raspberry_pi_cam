#! /usr/bin/env python3

# modified code from the following source
# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os

from config import Config

validated_image_list = os.listdir(Config.validated_image_dir)

GRAPH_FILENAME = "facenet_celeb_ncs.graph"

CV_WINDOW_NAME = "FaceNet- Multiple people"

CAMERA_INDEX = 0
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

FACE_MATCH_THRESHOLD = 0.6

def run_inference(image_to_classify, facenet_graph):

    resized_image = preprocess_image(image_to_classify)
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)
    output, userobj = facenet_graph.GetResult()
    return output

def overlay_on_image(display_image, image_info, matching):

    rect_width = 10
    offset = int(rect_width/2)
    if (image_info != None):
        cv2.putText(display_image, image_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if (matching):
        # match, green rectangle
        cv2.rectangle(display_image, (0+offset, 0+offset),
                      (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                      (0, 255, 0), 10)
    else:
        # not a match, red rectangle
        cv2.rectangle(display_image, (0+offset, 0+offset),
                      (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                      (0, 0, 255), 10)

def whiten_image(source_image):

    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

def preprocess_image(src):

    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = whiten_image(preprocessed_image)

    return preprocessed_image

def face_match(face1_output, face2_output):

    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    print('Total Difference is: ' + str(total_diff))
    return total_diff

def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True

def run_camera(valid_output, validated_image_filename, graph):

    camera_device = cv2.VideoCapture(CAMERA_INDEX)

    print ("open camera")

    camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
    camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    actual_camera_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print ('actual camera resolution: ' + str(actual_camera_width) + ' x ' + str(actual_camera_height))

    if ((camera_device == None) or (not camera_device.isOpened())):
        print ('Could not open camera.  Make sure it is plugged in.')
        print ('Also, if you installed python opencv via pip or pip3 you')
        print ('need to uninstall it and install from source with -D WITH_V4L=ON')
        print ('Use the provided script: install-opencv-from_source.sh')
        return

    frame_count = 0

    cv2.namedWindow(CV_WINDOW_NAME)

    found_match = False

    while True :
        # Read image from camera,
        ret_val, vid_image = camera_device.read()
        if (not ret_val):
            print("No image from camera, exiting")
            break

        frame_count += 1
        frame_name = 'camera frame ' + str(frame_count)

        test_output = run_inference(vid_image, graph)

        min_distance = 100
        min_index = -1

        for i in range(0,len(valid_output)):
            distance = face_match(valid_output[i], test_output)
            if distance < min_distance:
                min_distance = distance
                min_index = i

        if (min_distance<=FACE_MATCH_THRESHOLD):
            print('PASS!  File ' + frame_name + ' matches ' + validated_image_list[min_index])
            found_match = True

        else:
            found_match = False
            print('FAIL!  File ' + frame_name + ' does not match any image.')

        overlay_on_image(vid_image, frame_name, found_match)

        # check if the window is visible, this means the user hasn't closed
        # the window via the X button
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            print('window closed')
            break

        # display the results and wait for user to hit a key
        cv2.imshow(CV_WINDOW_NAME, vid_image)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                print('user pressed Q')
                break

    if (found_match):
        cv2.imshow(CV_WINDOW_NAME, vid_image)
        cv2.waitKey(0)

def main():

    use_camera = True
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No NCS devices found')
        quit()

    device = mvnc.Device(devices[0])

    device.OpenDevice()
    graph_file_name = GRAPH_FILENAME
    with open(os.path.join(Config.model_dir, graph_file_name), mode='rb') as f:
        graph_in_memory = f.read()

    graph = device.AllocateGraph(graph_in_memory)

    print ("load device")
    print (validated_image_list)

    valid_output = []
    for i in validated_image_list:
        validated_image = cv2.imread(os.path.join(Config.validated_image_dir, i))
        valid_output.append(run_inference(validated_image, graph))
    if (use_camera):
        run_camera(valid_output, validated_image_list, graph)

    graph.DeallocateGraph()
    device.CloseDevice()

if __name__ == "__main__":
    sys.exit(main())
