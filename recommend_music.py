import os
import random
import pickle
import numpy as np

input_data = [['25.2','70.1','3']]
song_mapper = {0:'pop', 1:'soft', 2:'funny', 3:'jazz', 4:'lofi'}

# load the model from disk
filename = 'music_recommendation_picast.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
predict_genre = loaded_model.predict(np.array(input_data))[0]
music_folder = 'song/' + song_mapper[predict_genre] + '/'
random_music = random.choice(os.listdir(music_folder))

print("Recommend music: ",random_music)