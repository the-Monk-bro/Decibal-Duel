import librosa , numpy as np , os, pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

print("Libraries imported")

SR = 16000
N_MELS = 128
HOP_LENGTH = 256
DURATION = 4


DATA_PATH = r'D:\CODES\AI-ML\Audio classifier\the-frequency-quest\test\test'

X = []
def sound_to_specdb (audio):
    y,_  = librosa.load ( audio , sr = SR , duration = DURATION )
    if len(y) < SR *DURATION:
        y = np.pad (y , (0, SR*DURATION - len(y)))
    mel = librosa.feature.melspectrogram(y=y , sr=SR , hop_length=HOP_LENGTH , n_mels = N_MELS)
    mel_db =  librosa.power_to_db(mel , ref=np.max)
    return(mel_db)

test_cases = []
for file in os.listdir(DATA_PATH):
    if file.endswith('.wav'):
        test_cases.append(file)
        audio_path  = os.path.join(DATA_PATH ,file)
        audio_in = sound_to_specdb(audio_path)
        X.append(audio_in)
X= np.array(X)
X = X[..., np.newaxis]

print("Data Aquired")


model = load_model(r'D:\CODES\AI-ML\Audio classifier\models\m9.h5')
print("Model loaded")

pred = model.predict(X)

pred  = np.argmax(pred , axis=1)

print("Predicted results")


classes =[]
for cls_folder in os.listdir(r"D:\CODES\AI-ML\Audio classifier\the-frequency-quest\train\train"):
    classes.append(cls_folder)
classes = np.array(classes)
cls_pred = classes[pred]


res = pd.DataFrame( {'ID':test_cases, 'Class':cls_pred}  )

res.to_csv(r"D:\CODES\AI-ML\Audio classifier\predictions\p9.csv" , index=False)

print("CSV made successfully")

