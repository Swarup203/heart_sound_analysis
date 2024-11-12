#!/usr/bin/env python3

import tensorflow as tf
import librosa
import numpy as np
import argparse
from termcolor import colored

# Load the model
model = tf.keras.models.load_model('my_model.h5')

normal_txt="""
 _   _                            _ 
| \ | | ___  _ __ _ __ ___   __ _| |
|  \| |/ _ \| '__| '_ ` _ \ / _` | |
| |\  | (_) | |  | | | | | | (_| | |
|_| \_|\___/|_|  |_| |_| |_|\__,_|_|
"""              
              
abnormal_txt="""
    _    _                                      _ 
   / \  | |__  _ __   ___  _ __ _ __ ___   __ _| |
  / _ \ | '_ \| '_ \ / _ \| '__| '_ ` _ \ / _` | |
 / ___ \| |_) | | | | (_) | |  | | | | | | (_| | |
/_/   \_\_.__/|_| |_|\___/|_|  |_| |_| |_|\__,_|_|
"""                

# Parameters
SAMPLE_RATE = 2000
DURATION = 3
MFCC_FEATURES = 40
INPUT_SHAPE = (MFCC_FEATURES, 128, 1)

def predict_wav_file(file_path, model):
    # Load the audio file and extract MFCC features
    audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
    mfcc = np.resize(mfcc, (MFCC_FEATURES, 128))
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Reshape for model input

    # Predict using the model
    prediction = model.predict(mfcc)
    predicted_label = np.argmax(prediction)

    # Print the result in a colored banner
    if predicted_label == 1:
        print(colored(abnormal_txt, 'white', 'on_red', attrs=['bold']))  # Red background for abnormal
    else:
        print(colored(normal_txt, 'white', 'on_green', attrs=['bold']))  # Green background for normal

# Main function to handle command-line arguments
if __name__ == '__main__':
    # Set up argument parser to take file path from command line
    parser = argparse.ArgumentParser(description='Predict heart sound condition from .wav file')
    parser.add_argument('file_path', type=str, help='Path to the .wav file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the prediction function with the provided file path
    predict_wav_file(args.file_path, model)

