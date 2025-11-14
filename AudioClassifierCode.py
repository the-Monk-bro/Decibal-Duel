import librosa
import os 
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , Dropout , BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping , ReduceLROnPlateau


print("Importing done")
DATA_PATH  = r"D:\CODES\AI-ML\Audio classifier\the-frequency-quest\train\train"
SR = 16000
N_MELS = 128
HOP_LENGTH  = 256
DURATION  = 4

def load_audio_file (file_path):
    y,_ = librosa.load(file_path , sr = SR , duration=DURATION)
    if len(y) < DURATION * SR : 
        y = np.pad(y, (0, SR * DURATION - len(y)))
    return y

def add_noise( y, noise_factor = 0.005):
    noise = np.random.randn(len(y))
    augmented_data = y+  noise_factor * noise
    augmented_data = augmented_data.astype(type(y[0]))
    return augmented_data

def time_shift(y ,shift_max_ms =100):
    shift_samples= int (SR * shift_max_ms/ 1000)
    shift = np.random.randint (-shift_samples , shift_samples)
    y_shifted = np.roll(y,shift)
    return y_shifted

def audio_to_mel (y_audio):
    mel = librosa.feature.melspectrogram(y=y_audio , sr=SR , hop_length=HOP_LENGTH , n_mels = N_MELS)
    mel_db =  librosa.power_to_db(mel , ref=np.max)
    return mel_db


X = []
y_labels = []

classes = sorted(os.listdir(DATA_PATH))

print ("Starting data aquisition and augmentation")

for idx , cls in enumerate(classes):
    cls_folder = os.path.join(DATA_PATH, cls)
    print (f"Processing class {cls}")
    for file in os.listdir(cls_folder):
        if file.endswith('.wav'):
            file_path = os.path.join (cls_folder ,file)

            y_original = load_audio_file(file_path)
            y_noise = add_noise (y_original)
            y_TimeShift = time_shift(y_original)

            mel_original = audio_to_mel(y_original)
            mel_noise = audio_to_mel(y_noise)
            mel_TimeShift = audio_to_mel(y_TimeShift)

            X.append(mel_original)
            y_labels.append(idx)
            X.append(mel_noise)
            y_labels.append(idx)
            X.append(mel_TimeShift)
            y_labels.append(idx)
            
print("Raw Data aquired")
X= np.array(X)
y_labels = np.array(y_labels)

X= X[..., np.newaxis]
y_cat  = to_categorical(y_labels, num_classes = len(classes))

X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=33)

print("Data converted to feed")

model = Sequential()

model.add(BatchNormalization(input_shape=(N_MELS, X.shape[2], 1))) 


model.add(Conv2D(32, (3, 3), activation='relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D((2, 2)))


model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25)) 


model.add(Flatten())

model.add(Dense(128, activation='relu')) 
model.add(BatchNormalization()) 
model.add(Dropout(0.5))


model.add(Dense(len(classes), activation='softmax'))

model.summary()
   
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

early_stopper = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(monitor= 'val_accuracy' , factor = 0.2, patience = 3 , verbose= 1 , min_lr = 0.00001)


model.fit( X_train , y_train , validation_data = (X_val, y_val) , epochs= 50 , batch_size = 16 , callbacks=[early_stopper, lr_scheduler])
print ("Model is trained")

model.save(r'D:\CODES\AI-ML\Audio classifier\models\m9.h5')
print("Model is saved")






    


 







