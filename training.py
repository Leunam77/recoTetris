import librosa
import numpy as np
from hmmlearn import hmm
import os
from joblib import dump
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Reemplaza "4" con el número de núcleos que deseas utilizar
# Definir las palabras que deseas reconocer
palabras = ["abajo", "derecha", "izquierda", "arriba"]

# Crear un modelo de HMM para cada palabra
modelos = {}
for palabra in palabras:
	modelos[palabra] = hmm.GaussianHMM(n_components=5)

def procesar_archivo(archivo):
	# Cargar la grabación de audio
	audio, sr = librosa.load(archivo)
	
	# Extraer las características de MFCC
	mfcc_feat = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
	
	return mfcc_feat.T

if __name__ == "__main__":
	# Inicializar el StandardScaler
	scaler = StandardScaler()
	datos_entrenamiento_total = []

	# Recopilar todos los datos de entrenamiento
	for palabra in palabras:
		carpeta = os.path.join("grabaciones", palabra)
		if not os.path.exists(carpeta):
			print(f"La carpeta {carpeta} no existe.")
			continue

		archivos = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.endswith(".wav")]
		
		with Pool() as p:
			datos_entrenamiento = p.map(procesar_archivo, archivos)
			datos_entrenamiento_total.extend(datos_entrenamiento)

	# Ajustar el StandardScaler con todos los datos de entrenamiento
	scaler.fit(np.vstack(datos_entrenamiento_total))
	dump(scaler, "scaler_general.joblib")

	# Para cada palabra, transformar los datos y entrenar el modelo HMM
	for palabra in palabras:
		carpeta = os.path.join("grabaciones", palabra)
		archivos = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.endswith(".wav")]
		
		with Pool() as p:
			datos_entrenamiento = p.map(procesar_archivo, archivos)
		
		#datos_entrenamiento = [scaler.transform(data) for data in datos_entrenamiento]
		modelos[palabra].fit(np.vstack(datos_entrenamiento))
		dump(modelos[palabra], palabra + "_modelo.joblib")