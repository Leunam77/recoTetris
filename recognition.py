import librosa
import numpy as np
from hmmlearn import hmm
import pyaudio
import wave
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
import threading
import os
from joblib import load

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Reemplaza "4" con el número de núcleos que quieres usar

PALABRAS = ["abajo", "derecha", "izquierda", "arriba"]
FORMAT = pyaudio.paInt16     
CHANNELS = 1
RATE = 44100
CHUNK = RATE * 2
WAVE_OUTPUT_FILENAME = "grabacion.wav"

def cargar_modelos_y_scaler():
    modelos = {}
    scaler = None
    try:
        scaler = load("scaler_general.joblib")
    except Exception as e:
        print(f"No se pudo cargar el scaler: {e}")
    for palabra in PALABRAS:
        try:
            modelos[palabra] = load(palabra + "_modelo.joblib")
        except Exception as e:
            print(f"No se pudo cargar el modelo para la palabra {palabra}: {e}")
    return modelos, scaler

def grabar_audio(stream):
    data = stream.read(CHUNK)
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as waveFile:
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(pyaudio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(data)

def cargar_audio():
    sr, audio_data = wav.read(WAVE_OUTPUT_FILENAME)
    audio_data = audio_data.astype(float)
    mfcc_feat = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).T    
    return mfcc_feat

def reconocer(modelos, scaler, stream, audio):
    try:
        #ultima_palabra_reconocida = None
        while True:
            grabar_audio(stream)
            mfcc_feat = cargar_audio()
            mfcc_feat = scaler.transform(mfcc_feat)

            scores = {palabra: modelos[palabra].score(mfcc_feat) for palabra in PALABRAS}
            print(f"Puntajes: {scores}")
            palabra_reconocida = max(scores, key=scores.get)
            print(f"Palabra reconocida: {palabra_reconocida}")
    except Exception as e:
        print(f"Error durante el reconocimiento de voz: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def main():
    modelos, scaler = cargar_modelos_y_scaler()

    if scaler is None:
        print("Scaler no cargado. Terminando el programa.")
        return

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    try:
        threading.Thread(target=reconocer, args=(modelos, scaler, stream, audio)).start()
    except Exception as e:
        print(f"Error al iniciar el reconocimiento de voz: {e}")

if __name__ == "__main__":
    main()