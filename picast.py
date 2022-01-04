import time
import os
import random
import pickle
import numpy as np

import RPi.GPIO as GPIO
GPIO.setwarnings(False)
# play music
import pygame.mixer

# temparature, humidity detection
from pigpio_dht import DHT22
# setup dht22
gpio = 4 # BCM Numbering
sensor = DHT22(gpio)

# Import packages for emotion detection
from picast_camera import Camera

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

def like_detection(input_data):
    
    while True:
        
        if GPIO.input(29)==1:
            # Open the file in append & read mode ('a+')
            with open("picast_data.txt", "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                file_object.write(' '.join(input_data))
                file_object.write("\n")
                print("I love this recommendation!!! \n Save it to the database!")
                
                break
                
        if GPIO.input(13)==1:
                        pygame.mixer.music.pause()
                        active==0
                        break

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
                                mapper = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
                                
                                emo_state, confidence = Camera.emotion_detect()
                                print("Emotion: ", mapper[emo_state],
                                      "(%.02f%%)" % confidence)
                                
                                if emo_state != False and confidence != False:
                                        
                                        # recommend music
                                        song_mapper = {0:'pop', 1:'soft', 2:'funny', 3:'jazz', 4:'lofi'}
                                        
                                        input_data = [[str(temperature_c), str(humidity), str(emo_state)]]
                                        predict_genre = loaded_model.predict(np.array(input_data))[0]
                                        music_folder = 'song/' + song_mapper[predict_genre] + '/'
                                        random_music = random.choice(os.listdir(music_folder))

                                        print("Recommend music: ",random_music)
                                        pygame.mixer.music.load(music_folder + random_music)

                                        print("start playing music!")
                                        pygame.mixer.music.play()
                                        playing=True
                                        active==0
                                        
                                        save_data = [str(temperature_c), str(humidity), str(emo_state), str(predict_genre)]
                                        like_detection(save_data)
                                        
         
                        except RuntimeError as error:
                            # Errors happen fairly often, DHT's are hard to read, just keep going
                            print(error.args[0])
                            time.sleep(2.0)
                            continue
                        except Exception as error:
                            sensor.exit()
                            raise error

                if GPIO.input(13)==1:
                        pygame.mixer.music.pause()
                        active==0
                        break
                        
                if GPIO.input(11)==1:
                        print("Active and Not playing")
                        playing = False
                        GPIO.input(15)==0
                        time.sleep(0.5)


GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.IN) # activate
GPIO.setup(37, GPIO.OUT,initial=GPIO.LOW) # activate light
GPIO.setup(13,GPIO.IN) # stop music detection
GPIO.setup(15,GPIO.IN) # reed switch detection
GPIO.setup(29,GPIO.IN) # like music detection

activeSetup()

# choose music
pygame.mixer.init(44100,-16,2,4096)
pygame.mixer.music.set_volume(1.0)

# load the model from disk
print("load recommendation model...")
filename = 'music_recommendation_picast.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

while True:
                
        if(GPIO.input(11)==1):
                activeState()
                time.sleep(0.5)
                
        if(active==1):
                watchDoor()
GPIO.cleanup()