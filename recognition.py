import librosa
import numpy as np
from hmmlearn import hmm
import pyaudio
import wave
import scipy.io.wavfile as wav
import threading
import os
from joblib import load

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Reemplaza "4" con el número de núcleos que quieres usar
def main():
	# Cargar los modelos de HMM entrenados
	modelos = {}
	for palabra in ["arriba", "abajo", "izquierda", "derecha"]:
		modelos[palabra] = load(palabra + "_modelo.joblib")

	# Configurar la grabación de audio
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	CHUNK = RATE * 2

	audio = pyaudio.PyAudio()

	# Iniciar la grabación
	stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,
						frames_per_buffer=CHUNK)

	def reconocer():
		ultima_palabra_reconocida = None  # Inicializar la variable aquí
		while True:
			data = stream.read(CHUNK)
			# Guardar la grabación en un archivo .wav
			WAVE_OUTPUT_FILENAME = "grabacion.wav"
			waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
			waveFile.setnchannels(CHANNELS)
			waveFile.setsampwidth(pyaudio.get_sample_size(FORMAT))  # Asegúrate de que esta línea está correctamente indentada
			waveFile.setframerate(RATE)
			waveFile.writeframes(data)
			waveFile.close()

			# Cargar la grabación y extraer las características de MFCC
			sr, audio_data = wav.read("grabacion.wav")  # Cambiar 'audio' por 'audio_data'
			audio_data = audio_data.astype(float)  # Convertir los datos de audio a punto flotante
			mfcc_feat = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

			# Usar los modelos de HMM para reconocer la palabra en la grabación
			scores = {}
			for palabra in ["arriba", "abajo", "izquierda", "derecha"]:
				scores[palabra] = modelos[palabra].score(mfcc_feat.T)
			palabra_reconocida = max(scores, key=scores.get)

			if palabra_reconocida in ["arriba", "abajo", "izquierda", "derecha"] and palabra_reconocida != ultima_palabra_reconocida:
				print("Palabra reconocida:", palabra_reconocida)
				ultima_palabra_reconocida = palabra_reconocida

	# Iniciar el reconocimiento de voz en un nuevo hilo
	threading.Thread(target=reconocer).start()

if __name__ == "__main__":
	main()