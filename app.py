# IMPORTACIÓN DE LIBRERÍAS.
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import sleep
import time

# FUNCIÓN QUE RECIBE EL ARREGLO DE ETIQUETAS DE CADA EMOCIÓN QUE PREDICE EL MODELO, LA RUTA DEL MODELO QUE RECONOCE ROSTROS Y LA RUTA DEL MODELO QUE CLASIFICA EL ROSTRO RECONOCIDO.
def faceEmotionRecognition(emotion_labels:list, path_face_recognition:str, path_classifier:str) -> list:
    # IMPORTACIÓN DE LOS MODELOS
    face_recognition = cv2.CascadeClassifier(path_face_recognition)
    classifier = load_model(path_classifier)
    
    # INICIALIZACIÓN DE WEBCAM DEL COMPUTADOR
    capture = cv2.VideoCapture(0)

    # TIEMPO DE PARTIDA
    start_time = time.time()

    # DECLARACIÓN DEL ARREGLO DATATIME VACÍO
    datatime = []

    # CICLO INFINITO QUE EXTRAERÁ FRAMES DE LA WEBCAM
    while True:
        _, frame = capture.read()

        # PASAMOS LA IMAGEN A UNA ESCALA DE GRISES
        frame_gray_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # SE EXTRAEN LAS POSICIONES DEL ROSTRO EN LA IMAGEN CON ESCALA DE GRISES
        faces_detected = face_recognition.detectMultiScale(frame_gray_scale)

        # SE RECORRE CADA ROSTRO ENCONTRADO
        for (x,y,w,h) in faces_detected:
            # SE DIBUJA UN RECTANGULO EN LA IMAGEN ORIGINAL
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

            # EXTRAEMOS SOLAMENTE LA INFORMACIÓN DEL ROSTRO
            face_frame_gray_scale = frame_gray_scale[y:y+h,x:x+w]
            # REDIMENSIONAMOS LA IMAGEN DEL ROSTRO A 48x48 PIXELES
            face_frame_gray_scale = cv2.resize(face_frame_gray_scale,(48,48),interpolation=cv2.INTER_AREA)


            if np.sum([face_frame_gray_scale])!=0:
                # NORMALIZAMOS LA IMAGEN
                face_frame_normalized = face_frame_gray_scale.astype('float')/255.0
                # LA VOLVEMOS UN ARREGLO
                face_frame_normalized = img_to_array(face_frame_normalized)
                face_frame_normalized = np.expand_dims(face_frame_normalized,axis=0)

                # CLASIFICAMOS LA EXPRESIÓN FACIAL CON EL MODELO CLASIFICADOR
                prediction = classifier.predict(face_frame_normalized, verbose=0)[0]

                # EXTRAEMOS LA EXPRESIÓN FACIAL CON MAYOR PROBABILIDAD DE SER
                label = emotion_labels[prediction.argmax()]
                
                # ALMACENAMOS UNA TUPLA CON EL TIEMPO Y LA EXPRESIÓN EN LA LISTA DATATIME.
                datatime.append(
                    (time.time() - start_time, label)
                )
                # POSICIÓN DE LA EXPRESIÓN EN EL RECTANGULO
                label_position = (x,y-10)
                # INSERTAMOS LA EXPRESIÓN EN EL FRAME
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        # MOSTRAMOS EL FRAME.
        cv2.imshow('Emotion Detector',frame)
        # SI SE PRESIONA LA COMBINACIÓN 1 + Q, ENTONCES EL CICLO PARA.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # CIERRA TODO.
    capture.release()
    cv2.destroyAllWindows()

    return datatime

# FUNCIÓN QUE GRAFICA LAS EXPRESIONES FACIALES DETECTADAS VS EL MOMENTO EN EL QUE SE CAPTURÓ.
def showDataTime(datatime):
    categorias = ['Asustado', 'Disgustado', 'Enojado', 'Triste', 'Neutral', 'Feliz', 'Sorpresa']

    tiempos, predicciones = zip(*datatime)

    end_time = tiempos[-1]

    mapeo_categorias = {categoria: i for i, categoria in enumerate(categorias)}
    predicciones_numeros = [mapeo_categorias[prediccion] for prediccion in predicciones]
    plt.scatter(tiempos, predicciones_numeros, linestyle='-', linewidth=0.7, color='red')
    plt.yticks(range(len(categorias)), categorias)
    plt.xticks(np.arange(0, end_time, step=1))
    plt.xlabel('Tiempo')
    plt.ylabel('Predicción')
    plt.title('Gráfico de Tiempo vs Predicción')
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    emotion_labels = ['Enojado','Disgustado','Asustado','Feliz','Neutral', 'Triste', 'Sorpresa']
    path_face_recognition = r'./models/haarcascade_frontalface_default.xml'
    path_classifier = r'./models/model.h5'

    datatime = faceEmotionRecognition(emotion_labels, path_face_recognition, path_classifier)
    showDataTime(datatime)