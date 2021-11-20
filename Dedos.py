import cv2  #
import mediapipe as mp  # pip install mediapipe
import os
import time

nombre_dedos = ["Letra_A", "Letra_E", "Letra_I", "Letra_O", "Letra_U"]
direccion_Entrenamiento = "Dedos\Foto\Entrenamiento"
direccion_Validacion = "Dedos\Foto\Validacion"


def cap_Señas(carpeta):
    cont = 0  # contador numero de fotos

    cap = cv2.VideoCapture(0)  # Camara

    clase_mano = mp.solutions.hands
    manos = clase_mano.Hands()

    dibujo = mp.solutions.drawing_utils


    while (True):
        ret, frame = cap.read()
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copia = frame.copy()
        resultado = manos.process(color)
        posiciones = []

        if resultado.multi_hand_landmarks:
            for mano in resultado.multi_hand_landmarks:
                for id, lm in enumerate(mano.landmark):
                    alto, ancho, c = frame.shape
                    corx, cory = int(lm.x*ancho), int(lm.y*alto)
                    posiciones.append([id, corx, cory])
                    dibujo.draw_landmarks(frame, mano, clase_mano.HAND_CONNECTIONS)
                if len(posiciones) != 0:
                    pto_i1 = posiciones[4]
                    pto_i2 = posiciones[20]
                    pto_i3 = posiciones[12]
                    pto_i4 = posiciones[0]
                    pto_i5 = posiciones[9]
                    x1, y1 = (pto_i5[1]-100), (pto_i5[2]-100)
                    ancho, alto = (x1+200), (y1+200)
                    x2, y2 = (x1+200), (y1+200)
                    dedos_reg = copia[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(carpeta + "\Dedos_{}.jpg".format(cont), dedos_reg)
                cont += 1

        cv2.imshow('Video', frame)
        k = cv2.waitKey(1)
        if k == 27 or cont >= 300:
            break
    cap.release()
    cv2.destroyAllWindows()



for nombre in nombre_dedos:
    input("Foto para " + nombre + " Entrenamiento")
    carpeta_Entrenamiento = direccion_Entrenamiento + '\\' + nombre
    if not os.path.exists(carpeta_Entrenamiento):
        print('Carpeta creada: ', carpeta_Entrenamiento)
        os.makedirs(carpeta_Entrenamiento)
    
    cap_Señas(carpeta_Entrenamiento)        
    

for nombre in nombre_dedos:
    input("Foto para " + nombre + " Validacion")
    carpeta_Validacion = direccion_Validacion + '\\' + nombre
    if not os.path.exists(carpeta_Validacion):
        print('Carpeta creada: ', carpeta_Validacion)
        os.makedirs(carpeta_Validacion)

    cap_Señas(carpeta_Validacion)    
    

    