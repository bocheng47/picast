#This code was written for an article on www.makeuseof.com by Ian Buckley.
import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37,GPIO.OUT)
GPIO.setup(11,GPIO.IN)
GPIO.setup(13,GPIO.IN)
GPIO.setup(15,GPIO.IN)
while True:
        GPIO.output(37,GPIO.HIGH)
        if (GPIO.input(11)==1):
                print("Button on GPIO 11")
                time.sleep(1)
        if (GPIO.input(13)==1):
                print("Button on GPIO 13")
                time.sleep(1)
        if (GPIO.input(15)==1): # door is opened!
                print("Button on GPIO 15")
                time.sleep(1)
GPIO.cleanup()