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

PALABRAS = ["abajo", "derecha", "izquierda", "rotar"]
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = RATE * 4
WAVE_OUTPUT_FILENAME = "grabacion.wav"


def cargar_modelos():
    modelos = {}
    for palabra in PALABRAS:
        try:
            modelos[palabra] = load(palabra + "_modelo.joblib")
        except Exception as e:
            print(f"No se pudo cargar el modelo para la palabra {palabra}: {e}")
    return modelos


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


def reconocer(modelos, stream, audio):

    try:
        ultima_palabra_reconocida = None
        scaler = StandardScaler()
        while True:
            print("Habla, mierda")
            grabar_audio(stream)
            mfcc_feat = cargar_audio()
            mfcc_feat = scaler.fit_transform(mfcc_feat)

            scores = {palabra: modelos[palabra].score(mfcc_feat) for palabra in PALABRAS}
            palabra_reconocida = max(scores, key=scores.get)
            if palabra_reconocida in PALABRAS:
                print(palabra_reconocida)
                ultima_palabra_reconocida = palabra_reconocida
    except Exception as e:
        print(f"Error durante el reconocimiento de voz: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def main():
    modelos = cargar_modelos()

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    try:

        threading.Thread(target=reconocer, args=(modelos, stream, audio)).start()
    except Exception as e:
        print(f"Error al iniciar el reconocimiento de voz: {e}")


if __name__ == "__main__":
    main()
