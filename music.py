# play music
import pygame.mixer
import time

# choose music
pygame.mixer.init(44100,-16,2,4096)
pygame.mixer.music.set_volume(1.0)
name = "Lyrics Chill Mix _ Stayy Mood.mp3"
pygame.mixer.music.load('song/' + name)
print("Loaded track - "+ str(name))

pygame.mixer.music.play()

time.sleep(5)

pygame.mixer.music.stop()