import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
# lazy initialization 
model = None
def load_model(size="base"):
    global model
    if model is None:
        model = whisper.load_model(size)
    return model

DATA_DIR = os.path.join(os.path.dirname(__file__),"..","data")
INPUT_FILE = os.path.join(DATA_DIR,"input.wav")

def record_audio(filename = INPUT_FILE, duration = 5, fs = 16000):    
    print("Speak!")
    recording = sd.rec(int(fs*duration), samplerate=fs, channels=1,dtype=np.int16, blocking=1)
    print("Done Recording!")
    write(filename, fs, recording)
    return filename

def transcribe(filename = INPUT_FILE):
    result = load_model().transcribe(filename)
    text = result["text"].strip()
    print(text)
    return text

if __name__ == "__main__":
    path = record_audio()
    transcribe(path)