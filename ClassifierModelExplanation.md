#Audio Classification CNN using TensorFlow
This project implements a 2D Convolutional Neural Network (CNN) using TensorFlow and Keras to classify audio files. The script handles audio loading, feature extraction, data augmentation, model building, and training.

##Project Overview
The model works by:

1.Loading .wav audio files from a specified directory.

2.Applying data augmentation (adding noise, time-shifting) to create a more robust dataset.

3.Converting the audio waveforms into 2D Mel Spectrograms using librosa.

4.Treating these spectrograms as images and feeding them into a 2D CNN.

5.Training the CNN to classify the audio into predefined categories (based on the folder structure).

6.Saving the trained model as an .h5 file.

##Key Features
1.Framework: TensorFlow 2.x (Keras)

2.Feature Extraction: Uses librosa to generate 128-band Mel spectrograms from 4-second audio clips (sampled at 16,000 Hz).

3.Data Augmentation: Automatically triples the training data by creating two augmented versions of each audio file:

4.Noise Injection: Adds random Gaussian noise.

5.Time Shift: Randomly shifts the audio in time.

6.Model Architecture: A sequential 2D CNN with 3 convolutional blocks, each using Conv2D, BatchNormalization, and MaxPooling2D. It uses Dropout for regularization.

7.Smart Training: Implements EarlyStopping and ReduceLROnPlateau callbacks to optimize training, prevent overfitting, and save the best model weights.

##How It Works
###1. Data Preprocessing & Augmentation
Constants: Key parameters like SR (Sample Rate), N_MELS (Number of Mel bands), and DURATION are defined.

load_audio_file: Loads a file with librosa, ensuring it's exactly 4 seconds long by padding if necessary.

add_noise & time_shift: Functions to create augmented audio data.

audio_to_mel: Converts a raw audio array into a log-power Mel spectrogram (in decibels).

Data Loading Loop:

Iterates through each class folder in DATA_PATH.

For each .wav file, it loads the original audio.

It creates a noisy version and a time-shifted version.

All three versions (original, noise, shift) are converted to Mel spectrograms and added to the dataset.

Labels are one-hot encoded using to_categorical.

###2. Model Architecture
The model is a Sequential Keras model with the following structure:

BatchNormalization (Input layer)

Conv Block 1: Conv2D(32, ...) -> BatchNormalization -> MaxPooling2D

Conv Block 2: Conv2D(64, ...) -> BatchNormalization -> MaxPooling2D

Conv Block 3: Conv2D(128, ...) -> BatchNormalization -> MaxPooling2D -> Dropout(0.25)

Flatten

Dense Block: Dense(128, ...) -> BatchNormalization -> Dropout(0.5)

Output Layer: Dense(num_classes, activation='softmax')

###3. Training
Compiler: Uses the adam optimizer and categorical_crossentropy loss, tracking accuracy.

Callbacks:

EarlyStopping: Stops training if validation accuracy doesn't improve for 10 epochs.

ReduceLROnPlateau: Reduces the learning rate if validation accuracy plateaus.

Training: The model is trained for a maximum of 50 epochs with a batch size of 16, using 20% of the data for validation.

Saving: The best trained model is saved to D:\CODES\AI-ML\Audio classifier\models\m9.h5.

##How to Use
1.Update the DATA_PATH variable to point to your training directory.

2.Ensure your data is organized into subfolders named by class (e.g., .../train/cat, .../train/dog).

3.Run the script.
