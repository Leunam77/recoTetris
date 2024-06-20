import librosa
import numpy as np
from hmmlearn import hmm
import os
from joblib import dump

os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Reemplaza "4" con el número de núcleos que deseas utilizar
# Definir las palabras que deseas reconocer
palabras = ["arriba", "abajo", "izquierda", "derecha"]

# Crear un modelo de HMM para cada palabra
modelos = {}
for palabra in palabras:
	modelos[palabra] = hmm.GaussianHMM(n_components=7, covariance_type="diag", n_iter=1000)

# Para cada palabra, cargar las grabaciones de audio y extraer las características de MFCC
for palabra in palabras:
	# Asumimos que tienes una carpeta para cada palabra con grabaciones de audio en archivos .wav
	carpeta = os.path.join("grabaciones", palabra)
	archivos = [f for f in os.listdir(carpeta) if f.endswith(".wav")]
	
	datos_entrenamiento = []
	
	for archivo in archivos:
		# Cargar la grabación de audio
		audio, sr = librosa.load(os.path.join(carpeta, archivo))
		
		# Extraer las características de MFCC
		mfcc_feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15)
		
		# Añadir las características a los datos de entrenamiento
		datos_entrenamiento.append(mfcc_feat.T)
	
	# Entrenar el modelo de HMM con los datos de entrenamiento
	modelos[palabra].fit(np.vstack(datos_entrenamiento))
	dump(modelos[palabra], palabra + "_modelo.joblib")