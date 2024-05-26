# EJERCICIO 3:
# Generamos un nuevo script, que será el encargado de procesar los paneles, localizar los caracteres y ejecutar el clasificador que desarrollaste en el ejercicio anterior
# El mejor clasificador en el ejercicio anterior es "svc" con "raw" y "none" en reducción dimensional

import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RANSACRegressor
# Clasificador de vecinos más cercanos
from sklearn.neighbors import KNeighborsClassifier
# Clasificador de árbol de decisión
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
# Para extraer características HOG de las imágenes
from skimage.feature import hog  
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse

# Función para preprocesar una imagen: convierte a escala de grises, aplica umbralización adaptativa y extrae la ROI más grande.
def pre_procesado_image(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img_gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (25, 25))
        return roi.flatten()
    return cv2.resize(img_gray, (25, 25)).flatten() 

# Función para extraer características HOG de una imagen, por si los otros clasificadores son mejores
def extract_hog_features(img):
    roi = pre_procesado_image(img)
    hog_features = hog(roi.reshape(25, 25), pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

# Paso 1: Detectar caracteres en la imagen usando umbralización y búsqueda de contornos.
def detectar_caracteres(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangulos = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.2 < w/h < 1.0 and w > 5 and h > 5:
            rectangulos.append((x, y, w, h))
    return rectangulos

# Paso 2: Encontrar líneas de texto utilizando RANSAC para agrupar caracteres alineados.
def encontrar_lineas_de_texto(rectangulos):
    centros = [(x + w // 2, y + h // 2) for (x, y, w, h) in rectangulos]
    lineas = []
    while len(centros) > 1:  # Asegurarse de que haya más de un punto
        X = np.array([[c[0]] for c in centros])
        y = np.array([c[1] for c in centros])
        ransac = RANSACRegressor(min_samples=2)
        ransac.fit(X, y)
        inliers = ransac.inlier_mask_
        outliers = ~ransac.inlier_mask_
        linea = [centros[i] for i in range(len(centros)) if inliers[i]]
        lineas.append(linea)
        centros = [centros[i] for i in range(len(centros)) if outliers[i]]
    if len(centros) == 1:  # Añadir el último punto si queda uno
        lineas.append([centros[0]])
    return lineas

# Paso 3: Clasificar cada carácter detectado usando el clasificador entrenado.
def clasificar_caracteres(rectangulos, imagen, clasificador, label_encoder):
    caracteres = []
    for (x, y, w, h) in rectangulos:
        roi = imagen[y:y+h, x:x+w]
        roi_resized = pre_procesado_image(roi)
        caracter_predicho = clasificador.predict([roi_resized])[0]
        caracter = label_encoder.inverse_transform([caracter_predicho])[0]
        caracteres.append((x, y, caracter))
    # Ordenar caracteres de izquierda a derecha y de arriba a abajo.
    caracteres.sort(key=lambda c: (c[1], c[0]))
    lineas = encontrar_lineas_de_texto([(x, y, 0, 0) for (x, y, _) in caracteres])
    texto_ocr = []
    for linea in lineas:
        linea_texto = ''.join([c[2] for c in caracteres if (c[0], c[1]) in linea])
        texto_ocr.append(linea_texto)
    return '+'.join(texto_ocr)

# Función para procesar los paneles, aplicar la detección de caracteres y clasificación en cada uno.
def procesar_paneles(path_paneles, clasificador, label_encoder):
    resultados = []
    for nombre_archivo in os.listdir(path_paneles):
        if nombre_archivo.endswith(".png"):
            ruta_imagen = os.path.join(path_paneles, nombre_archivo)
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
                continue
            rectangulos = detectar_caracteres(imagen)
            texto_ocr = clasificar_caracteres(rectangulos, imagen, clasificador, label_encoder)
            resultados.append(f"{nombre_archivo};0;0;{imagen.shape[1]};{imagen.shape[0]};panel;1.0;{texto_ocr}")
    return resultados

# Función para cargar datos de entrenamiento y extraer características.
def cargar_datos_entrenamiento(path, feature_extractor):
    images = []
    labels = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                label = os.path.basename(root)
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(feature_extractor(img))
                    labels.append(label)
    return np.array(images), np.array(labels)

# Función para visualizar la matriz de confusión de manera detallada.
def plot_confusion_matrix_detallada(cm, title='Confusion matrix', cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]), rotation=90, fontsize=8)
    plt.yticks(tick_marks, range(cm.shape[0]), fontsize=8)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y, x]), xy=(x, y),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=6)

if __name__ == "__main__":
    # Misma estructura de arranque en el Ejercicio 1 y 2
    parser = argparse.ArgumentParser(description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument('--classifier', type=str, default="svc", help='Classifier string name')
    parser.add_argument('--feature', type=str, default="raw", help='Feature extraction method (raw, hog)')
    parser.add_argument('--dim_reduction', type=str, default="none", help='Dimensionality reduction method (none, pca)')
    parser.add_argument('--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument('--validation_path', default="./validation_ocr", help='Select the validation data dir')
    parser.add_argument('--test_path', default="./test_ocr_panels", help='Select the test data dir')
    
    args = parser.parse_args()

    train_path = os.path.abspath(args.train_path)
    validation_path = os.path.abspath(args.validation_path)
    test_path = os.path.abspath(args.test_path)
    print(f"Train path: {train_path}")
    print(f"Validation path: {validation_path}")
    print(f"Test path: {test_path}")

    # Seleccionar el método de extracción de características aunque ya sabemos, cual es el mejor, se deja los demás por si se necesita realizar pruebas
    if args.feature == "raw":
        feature_extractor = pre_procesado_image
    elif args.feature == "hog":
        feature_extractor = extract_hog_features
    else:
        print(f"Error: Feature extractor {args.feature} no soportado.")
        exit()
    # Cargar datos de entrenamiento y validación
    train_images, train_labels = cargar_datos_entrenamiento(train_path, feature_extractor)
    validation_images, validation_labels = cargar_datos_entrenamiento(validation_path, feature_extractor)
    
    print(f"Loaded {len(train_images)} training images and {len(validation_images)} validation images.")

    if len(train_images) == 0 or len(validation_images) == 0:
        print("Error: No se cargaron imágenes. Verifique las rutas y el formato de los archivos.")
        exit()

    # Codificar las etiquetas en números
    label_encoder = LabelEncoder()
    combined_labels = list(train_labels) + list(validation_labels)
    label_encoder.fit(combined_labels)
    
    train_labels_encoded = label_encoder.transform(train_labels)
    validation_labels_encoded = label_encoder.transform(validation_labels)

    # Aplicar reducción de dimensionalidad si es necesario
    if args.dim_reduction == "none":
        train_features = train_images
        validation_features = validation_images
    elif args.dim_reduction == "pca":
        pca = PCA(n_components=50)
        train_features = pca.fit_transform(train_images)
        validation_features = pca.transform(validation_images)
    else:
        print(f"Error: Método de reducción de dimensionalidad {args.dim_reduction} no soportado.")
        exit()

    # Entrenar el clasificador y evaluar la precisión
    if args.classifier == "svc":
        classifier = SVC()
        classifier.fit(train_features, train_labels_encoded)
        predicted_labels = classifier.predict(validation_features)
        accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
        print("Accuracy = ", accuracy)
        cm = confusion_matrix(validation_labels_encoded, predicted_labels)
        plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {args.classifier}, {args.feature}, {args.dim_reduction}")
        plt.show()
    elif args.classifier == "knn":
        classifier = KNeighborsClassifier()
        classifier.fit(train_features, train_labels_encoded)
        predicted_labels = classifier.predict(validation_features)
        accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
        print("Accuracy = ", accuracy)
        cm = confusion_matrix(validation_labels_encoded, predicted_labels)
        plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {args.classifier}, {args.feature}, {args.dim_reduction}")
        plt.show()
    elif args.classifier == "dtree":
        classifier = DecisionTreeClassifier()
        classifier.fit(train_features, train_labels_encoded)
        predicted_labels = classifier.predict(validation_features)
        accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
        print("Accuracy = ", accuracy)
        cm = confusion_matrix(validation_labels_encoded, predicted_labels)
        plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {args.classifier}, {args.feature}, {args.dim_reduction}")
        plt.show()
    else:
        print(f"Error: Clasificador {args.classifier} no soportado.")
        exit()

    # Guardar el modelo entrenado y el LabelEncoder
    joblib.dump(classifier, 'svc_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Procesar los paneles, para luego la función --> "evaluar_resultados_test_ocr_panels.py"
    clasificador = joblib.load('svc_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    resultados = procesar_paneles(test_path, clasificador, label_encoder)
    with open('resultado.txt', 'w') as f:
        for resultado in resultados:
            f.write(f"{resultado}\n")