import pygame
from copy import deepcopy
from random import choice, randrange

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
from queue import Queue

os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Reemplaza "4" con el número de núcleos que quieres usar

PALABRAS = ["abajo", "derecha", "izquierda", "rotar"]
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = RATE * 3
WAVE_OUTPUT_FILENAME = "grabacion.wav"

W, H = 10, 16
TILE = 45
GAME_RES = W * TILE, H * TILE
RES = 750, 800
FPS = 60

pygame.init()
sc = pygame.display.set_mode(RES)
game_sc = pygame.Surface(GAME_RES)
clock = pygame.time.Clock()
input_text = ""

grid = [pygame.Rect(x * TILE, y * TILE, TILE, TILE) for x in range(W) for y in range(H)]

figures_pos = [[(-1, 0), (-2, 0), (0, 0), (1, 0)],
               [(0, -1), (-1, -1), (-1, 0), (0, 0)],
               [(-1, 0), (-1, 1), (0, 0), (0, -1)],
               [(0, 0), (-1, 0), (0, 1), (-1, -1)],
               [(0, 0), (0, -1), (0, 1), (-1, -1)],
               [(0, 0), (0, -1), (0, 1), (1, -1)],
               [(0, 0), (0, -1), (0, 1), (-1, 0)]]

figures = [[pygame.Rect(x + W // 2, y + 1, 1, 1) for x, y in fig_pos] for fig_pos in figures_pos]
figure_rect = pygame.Rect(0, 0, TILE - 2, TILE - 2)
field = [[0 for i in range(W)] for j in range(H)]

anim_count, anim_speed, anim_limit = 0, 30, 2000

bg = pygame.image.load('img/bg.jpg').convert()
game_bg = pygame.image.load('img/bg2.jpg').convert()

main_font = pygame.font.Font('font/font.ttf', 65)
font = pygame.font.Font('font/font.ttf', 45)

title_tetris = main_font.render('TETRIS', True, pygame.Color('darkorange'))
title_score = font.render('score:', True, pygame.Color('green'))
title_record = font.render('record:', True, pygame.Color('purple'))

get_color = lambda: (randrange(30, 256), randrange(30, 256), randrange(30, 256))

figure, next_figure = deepcopy(choice(figures)), deepcopy(choice(figures))
color, next_color = get_color(), get_color()

score, lines = 0, 0
scores = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}


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


def reconocer():
    modelos = cargar_modelos()

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    try:
        ultima_palabra_reconocida = None
        scaler = StandardScaler()
        print("Habla, por favor...")
        grabar_audio(stream)
        mfcc_feat = cargar_audio()
        mfcc_feat = scaler.fit_transform(mfcc_feat)

        scores = {palabra: modelos[palabra].score(mfcc_feat) for palabra in PALABRAS}
        palabra_reconocida = max(scores, key=scores.get)
        if palabra_reconocida in PALABRAS:
            print("Palabra reconocida", palabra_reconocida)
            return palabra_reconocida

    except Exception as e:
        print(f"Error durante el reconocimiento de voz: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def check_borders():
    if figure[i].x < 0 or figure[i].x > W - 1:
        return False
    elif figure[i].y > H - 1 or field[figure[i].y][figure[i].x]:
        return False
    return True


def get_record():
    try:
        with open('record') as f:
            return f.readline()
    except FileNotFoundError:
        with open('record', 'w') as f:
            f.write('0')


def set_record(record, score):
    rec = max(int(record), score)
    with open('record', 'w') as f:
        f.write(str(rec))


def process_command(command):
    command = command.strip().lower()
    dx, rotate, quick_drop = 0, False, False
    if command == "izquierda":
        dx = -1
    elif command == "derecha":
        dx = 1
    elif command == "abajo":
        quick_drop = True
    elif command == "rotar":
        rotate = True
    return dx, rotate, quick_drop


def entregarPalabra():
    return reconocer()

def obtener_y_procesar_comandos(queue):
    while True:
        command = entregarPalabra()  # Obtener el comando
        queue.put(command)  # Poner el comando en la cola

# Crear una cola para comunicación entre hilos
queue = Queue()

# Crear una instancia de Thread para obtener y procesar comandos
thread = threading.Thread(target=obtener_y_procesar_comandos, args=(queue,))

# Iniciar el hilo para obtener y procesar comandos
thread.start()

while True:
    record = get_record()
    dx, rotate = 0, False
    sc.blit(bg, (0, 0))
    sc.blit(game_sc, (20, 20))
    game_sc.blit(game_bg, (0, 0))
    # delay for full lines
    for i in range(lines):
        pygame.time.wait(200)
    # control

    quick_drop = False

    # Procesar comandos desde la cola
    while not queue.empty():
        command = queue.get()
        dx, rotate, quick_drop = process_command(command)
          # Procesar el comando recibido

    #command = entregarPalabra()
    #dx, rotate, quick_drop = process_command(command)

    if quick_drop == True:
        print("Comando Abajo")
        anim_limit = 100


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                dx = -1
            elif event.key == pygame.K_RIGHT:
                dx = 1
            elif event.key == pygame.K_DOWN:
                anim_limit = 100
            elif event.key == pygame.K_UP:
                rotate = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                input_text = input_text[:-1]
            elif event.key == pygame.K_RETURN:
                dx, rotate, quick_drop = process_command(input_text)
                input_text = ""
            else:
                input_text += event.unicode

    # move x
    figure_old = deepcopy(figure)
    for i in range(4):
        figure[i].x += dx
        if not check_borders():
            figure = deepcopy(figure_old)
            break
    # move y
    anim_count += anim_speed
    if anim_count > anim_limit:
        anim_count = 0
        figure_old = deepcopy(figure)
        for i in range(4):
            figure[i].y += 1
            if not check_borders():
                for i in range(4):
                    field[figure_old[i].y][figure_old[i].x] = color
                figure, color = next_figure, next_color
                next_figure, next_color = deepcopy(choice(figures)), get_color()
                anim_limit = 2000
                break
    # rotate
    center = figure[0]
    figure_old = deepcopy(figure)
    if rotate:
        for i in range(4):
            x = figure[i].y - center.y
            y = figure[i].x - center.x
            figure[i].x = center.x - x
            figure[i].y = center.y + y
            if not check_borders():
                figure = deepcopy(figure_old)
                break
    # check lines
    line, lines = H - 1, 0
    for row in range(H - 1, -1, -1):
        count = 0
        for i in range(W):
            if field[row][i]:
                count += 1
            field[line][i] = field[row][i]
        if count < W:
            line -= 1
        else:
            anim_speed += 3
            lines += 1
    # compute score
    score += scores[lines]
    # draw grid
    [pygame.draw.rect(game_sc, (40, 40, 40), i_rect, 1) for i_rect in grid]
    # draw figure
    for i in range(4):
        figure_rect.x = figure[i].x * TILE
        figure_rect.y = figure[i].y * TILE
        pygame.draw.rect(game_sc, color, figure_rect)
    # draw field
    for y, raw in enumerate(field):
        for x, col in enumerate(raw):
            if col:
                figure_rect.x, figure_rect.y = x * TILE, y * TILE
                pygame.draw.rect(game_sc, col, figure_rect)
    # draw next figure
    for i in range(4):
        figure_rect.x = next_figure[i].x * TILE + 380
        figure_rect.y = next_figure[i].y * TILE + 185
        pygame.draw.rect(sc, next_color, figure_rect)
    # draw titles
    sc.blit(title_tetris, (485, 50))
    sc.blit(title_score, (535, 680))
    sc.blit(font.render(str(score), True, pygame.Color('white')), (550, 740))
    sc.blit(title_record, (525, 550))
    sc.blit(font.render(record, True, pygame.Color('gold')), (550, 610))
    # game over

    input_surface = font.render(input_text, True, pygame.Color('white'))
    sc.blit(input_surface, (20, GAME_RES[1] + 10))

    for i in range(W):
        if field[0][i]:
            set_record(record, score)
            field = [[0 for i in range(W)] for i in range(H)]
            anim_count, anim_speed, anim_limit = 0, 60, 2000
            score = 0
            for i_rect in grid:
                pygame.draw.rect(game_sc, get_color(), i_rect)
                sc.blit(game_sc, (20, 20))
                pygame.display.flip()
                clock.tick(200)

    pygame.display.flip()
    clock.tick(FPS)
