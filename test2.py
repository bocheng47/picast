import time
import RPi.GPIO as GPIO
# play music
import pygame.mixer

# temparature, humidity detection
from pigpio_dht import DHT22
# setup dht22
gpio = 4 # BCM Numbering
sensor = DHT22(gpio)

# Import packages for emotion detection
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow.compat.v1 as tf
import argparse
import sys
from PIL import Image
import time
import tflite_runtime.interpreter as tflite

def activeSetup():
        global active
        active=0
        print("Currently Not Active")
        
def activeState():
        global active
        if active==1:
                active=0
                GPIO.output(37,GPIO.LOW)
                print("Currently Not Active")
        elif active==0:
                print ("Activating in 3 seconds")
                for x in range(0,3):
                        GPIO.output(37,GPIO.HIGH)
                        time.sleep(0.5)
                        GPIO.output(37,GPIO.LOW)
                        time.sleep(0.5)
                active=1
                GPIO.output(37,GPIO.HIGH)
                print("Currently Active")
        else: return

def watchDoor():
        global playing
        playing = False
        while True:
                #print(active, GPIO.input(15), playing)
                if active==1 and GPIO.input(15)==1 and playing == False:
                        
                        
                            
                        # open door 
                        
                        # detect humidity, temperature
                        try:
                            result = sensor.read()
                            print(result)
                            
                            if result["valid"] == True:
                                
                                temperature_c = result["temp_c"]
                                humidity = result["humidity"]

                                print(
                                    "Temp: {:.1f} C ,   Humidity: {}% ".format(
                                        temperature_c, humidity
                                    )
                                )
                                                        
                                # detect emotion
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
                                        
                                        print("start playing music!")
                                        playing=True
                                        pygame.mixer.music.play()

                                        break

                                    cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (60, 100), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
                                    
                                    cv2.imshow('Emotion detector', frame)

                                    t2 = cv2.getTickCount()
                                    time1 = (t2 - t1) / freq
                                    frame_rate_calc = 1 / time1

                                    rawCapture.truncate(0)
                                    

                                camera.close()
                                cv2.destroyAllWindows()                               
                                
                                
                                
                        except RuntimeError as error:
                            # Errors happen fairly often, DHT's are hard to read, just keep going
                            print(error.args[0])
                            time.sleep(2.0)
                            continue
                        except Exception as error:
                            dhtDevice.exit()
                            raise error

                if GPIO.input(13)==1:
                        print("stop playing music!")
                        pygame.mixer.music.pause()
                        break
                        
                if GPIO.input(11)==1:
                        print("Active and Not playing")
                        playing = False
                        GPIO.input(15)==0
                        time.sleep(0.5)


GPIO.setmode(GPIO.BOARD)
GPIO.setup(37, GPIO.OUT,initial=GPIO.LOW)
#GPIO.setup(7,GPIO.IN)
GPIO.setup(11,GPIO.IN)
GPIO.setup(13,GPIO.IN)
GPIO.setup(15,GPIO.IN)

activeSetup()

# choose music
pygame.mixer.init(44100,-16,2,4096)
pygame.mixer.music.set_volume(1.0)
name = "Lyrics Chill Mix _ Stayy Mood.mp3"
pygame.mixer.music.load('song/' + name)
print("Loaded track - "+ str(name))

while True:
                
        if(GPIO.input(11)==1):
                activeState()
                time.sleep(0.5)
                
        if(active==1):
                watchDoor()
GPIO.cleanup()