# Emotion detector for Raspberry Pi 4
# Author: Elliot Blanford
# Date: 1/18/2021
# Description: Just run it and make faces at the camera! It will print out predictions and
# confidence if it is above threshold. It will also buzz on detecting a change in emotional state

# Original inspiration by Evan Juras
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
# I updated it to work with tensorflow v2, changed it to an emotion detection model, and added feedback device
# a vibrating motor controlled by GPIO pins

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


# Import packages
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow.compat.v1 as tf
import argparse
import sys
from PIL import Image
import RPi.GPIO as GPIO
import time
import tflite_runtime.interpreter as tflite

cascPath = "/home/pi/.local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Set up camera constants
IM_WIDTH = 1280//4
IM_HEIGHT = 192 #720//4

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# translate model output to label
mapper = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

camera = PiCamera()
camera.resolution = (IM_WIDTH, IM_HEIGHT)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))

#set up GPIO
GPIO.setmode(GPIO.BOARD)

confidence_threshold = 50 #in %

for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    t1 = cv2.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = frame_gray
    faces = faceCascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        box_size = max(h, w)
        roi_gray = frame_gray[y:y+box_size, x:x+box_size]
        roi_color = frame[y:y+box_size, x:x+box_size]

    face_gray = cv2.resize(roi_gray, (48,48))
    face_expanded = np.expand_dims(face_gray/255, axis=2).astype('float32')
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="Emotion_Detector/emotions.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], [face_expanded])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    confidence = np.max(output_data[0]) * 100
    # need to show predition on the screen, if it's a 'confident' prediction, i'll show the %
    if confidence > (confidence_threshold):
        emo_state = np.where(output_data[0] == np.max(output_data[0]))[0][0]
        print("Guess: ", mapper[emo_state],
              "(%.02f%%)" % confidence)
    else:
        print("No emotion")

    cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (60, 100), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Emotion detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()
GPIO.cleanup()

cv2.destroyAllWindows()

