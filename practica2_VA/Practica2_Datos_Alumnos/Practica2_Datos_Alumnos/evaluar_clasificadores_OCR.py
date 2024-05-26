# Asignatura de Visión Artificial (URJC). Script de evaluación.
import argparse

#import panel_det //Se Comentará debido a que no lo utilizamos en las demás funciones y mnos generá error


import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
# Para calcular precisión y matriz de confusión
from sklearn.metrics import accuracy_score, confusion_matrix
# Para convertir etiquetas categóricas en números
from sklearn.preprocessing import LabelEncoder
# Para manejar rutas de archivos y directorios para su acceso
from pathlib import Path
# Añadimos los tipos de clasificadores:
# Clasificador de máquinas de vectores de soporte
from sklearn.svm import SVC
# Clasificador de vecinos más cercanos
from sklearn.neighbors import KNeighborsClassifier
# Clasificador de árbol de decisión
from sklearn.tree import DecisionTreeClassifier
# Análisis de componentes principales para reducción de dimensionalidad
from sklearn.decomposition import PCA 
# Para extraer características HOG de las imágenes
from skimage.feature import hog  
# Añadimos import os para cargar las imagenes de la carpeta
import os
# Clasificador Ejercicicio1
from ocr_classifier import OCRClassifier
from lda_normal_bayes_classifier import LdaNormalBayesClassifier

# Función para cargar imágenes desde las carpetas, además le pasaremos la caracteristicas por si queremos diferentes features
def carga_imagenes_carpeta(folder,feature):
    images = []
    labels = []
    if not os.path.exists(folder):
        print(f"Error: La carpeta {folder} no existe.")
        return images, labels

    #Comentar o descomentar, si se quiere comprobar que las imagenes se cargan
    #print(f"Cargando imágenes de la carpeta: {folder}")
    for subdir, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".png"):  # Asumiendo que las imágenes están en formato .png
                label = os.path.basename(subdir)  # Usando el nombre de la subcarpeta como etiqueta
                img_path = os.path.join(subdir, filename)
                #print(f"Cargando imagen: {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error al cargar la imagen: {img_path}")
                    continue  # Saltar esta imagen y seguir con las demás
                try:
                    img = feature(img)
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error procesando la imagen {img_path}: {e}")
    print(f"Total de imágenes cargadas: {len(images)}")
    return images, labels

def carga_algunas_imagenes_carpeta(folder, num_images=5):
    images = []
    labels = []
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"Error: La carpeta {folder} no existe.")
        return images, labels

    print(f"Cargando hasta {num_images} imágenes de la carpeta: {folder}")
    all_files = []
    for subdir in folder_path.iterdir():
        if subdir.is_dir():
            for img_path in subdir.glob('*.png'):
                label = subdir.name  # Usando el nombre de la subcarpeta como etiqueta
                all_files.append((img_path, label))

    if len(all_files) > num_images:
        all_files = random.sample(all_files, num_images)

    for img_path, label in all_files:
        if not img_path.exists():
            print(f"Error: La imagen {img_path} no existe.")
            continue
        
        print(f"Cargando imagen: {img_path}")
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error al cargar la imagen: {img_path}")
                continue  # Saltar esta imagen y seguir con las demás
            
            img = pre_procesado_image(img)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error procesando la imagen {img_path}: {e}")
    print(f"Total de imágenes cargadas: {len(images)}")
    return images, labels

# Función para preprocesar la imagen (umbralización y redimensionamiento)
def pre_procesado_image(img):
    # Umbralización
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (25, 25))  # redimensionar a 25x25 píxeles
        return roi.flatten()
    # En caso de que no se encuentren contornos, devolver la imagen redimensionada
    return cv2.resize(img, (25, 25)).flatten()

# Función para extraer características HOG de una imagen
def extract_hog_features(img):
    roi = pre_procesado_image(img)  # Preprocesa la imagen
    hog_features = hog(roi.reshape(25, 25), pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

# plt.cm.get_cmap me indica que esta en desuso por ello cambiamos la entrada
#def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
def plot_confusion_matrix(cm, title='Confusion matrix', cmap='Blues'):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y, x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

#Función plot_confusión_matrix más detallada
def plot_confusion_matrix_detallada(cm, title='Confusion matrix', cmap='Blues'):
#def plot_confusion_matrix_detallada(cm, title='Confusion matrix', cmap='Blues', filename=None):
    '''
    Given a confusion matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.figure(figsize=(10, 10))  # Aumentar el tamaño de la figura
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]), rotation=90, fontsize=8)  # Rotar etiquetas en el eje x y reducir tamaño de fuente
    plt.yticks(tick_marks, range(cm.shape[0]), fontsize=8)  # Reducir tamaño de fuente en el eje y
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
                        fontsize=6)  # Reducir tamaño de fuente dentro de la celda

def visualizar_imagenes(images, labels, num=5):
    if len(images) < num:
        num = len(images)
    indices = random.sample(range(len(images)), num)
    for i in indices:
        plt.imshow(images[i].reshape(25, 25), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.show()
# Función para organizar las imágenes en un diccionario
def organizar_imagenes_en_diccionario(images, labels):
    images_dict = {}
    for img, label in zip(images, labels):
        if label not in images_dict:
            images_dict[label] = []
        images_dict[label].append(img)
    return images_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument(
        #Modificar en la ejecución el default por "svc" - "knn" - "dtree" para probar ejecuciones
        '--classifier', type=str, default="lda_normal_bayes", help='Classifier string name')
    parser.add_argument(
        #Modificar en la ejecución el default
        '--feature', type=str, default="raw", help='Feature extraction method (raw, hog)')
    parser.add_argument(
        #Modificar en la ejecución el default
        '--dim_reduction', type=str, default="none", help='Dimensionality reduction method (none, pca)')
    parser.add_argument(
        '--train_path', default="./train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="./validation_ocr", help='Select the validation data dir')
    
    args = parser.parse_args()

   
        
    # args = parser.parse_args()

    # # 1) Cargar las imágenes de entrenamiento y sus etiquetas. 
    # # También habrá que extraer los vectores de características asociados (en la parte básica 
    # # umbralizar imágenes, pasar findContours y luego redimensionar)
    # train_path = os.path.abspath(args.train_path)
    # validation_path = os.path.abspath(args.validation_path)
    # print(f"Train path: {train_path}")
    # print(f"Validation path: {validation_path}")
    
    # #Con al carga de las imagenes, hay un problema: 
    # #    - Cargando imagen: C:\Users\david\OneDrive\Escritorio\Universidad\Universidad_2023-2024\SegundoCuatri\VisionArtificial\VisionArtificial\practica2_VA_23-24\Practica2_Datos_Alumnos\Practica2_Datos_Alumnos\train_ocr\minúsculas\f\0180.png[ WARN:0@120.978] global loadsave.cpp:248 cv::findDecoder imread_('C:\Users\david\OneDrive\Escritorio\Universidad\Universidad_2023-2024\SegundoCuatri\VisionArtificial\VisionArtificial\practica2_VA_23-24\Practica2_Datos_Alumnos\Practica2_Datos_Alumnos\train_ocr\min├║sculas\f\0180.png'): can't open/read file: check file path/integrity
    # #    Error al cargar la imagen: C:\Users\david\OneDrive\Escritorio\Universidad\Universidad_2023-2024\SegundoCuatri\VisionArtificial\VisionArtificial\practica2_VA_23-24\Practica2_Datos_Alumnos\Practica2_Datos_Alumnos\train_ocr\minúsculas\f\0180.png   
    # #    Este es warning es debido a que se denomina las carpetas con acentos, por ello se renombrará los acentos
    # # Cargar todas las imagenes
    # # Añadimos opciones referentes al ejercicio 2, como el pre_procesado_imagenes o el neuvo el HOG
    # if args.feature == "raw":
    #     feature_extractor = pre_procesado_image
    # elif args.feature == "hog":
    #     feature_extractor = extract_hog_features
    # else:
    #     print(f"Error: Feature extractor {args.feature} no soportado.")
    #     exit()
    # train_images, train_labels = carga_imagenes_carpeta(train_path,feature_extractor)
    # validation_images, validation_labels = carga_imagenes_carpeta(validation_path,feature_extractor)
    
    # # Cargar 5 imagenes "PRUEBAS"
    # # 1) Cargar las imágenes de entrenamiento y sus etiquetas.
    # #train_path = os.path.abspath(args.train_path)   
    # #validation_path = os.path.abspath(args.validation_path)
    # #print(f"Train path: {train_path}")
    # #print(f"Validation path: {validation_path}")

    # #train_images, train_labels = carga_algunas_imagenes_carpeta(train_path, num_images=5)
    # #validation_images, validation_labels = carga_algunas_imagenes_carpeta(validation_path, num_images=5)
    
    # print(f"Loaded {len(train_images)} training images and {len(validation_images)} validation images.")

    # if len(train_images) == 0 or len(validation_images) == 0:
    #     print("Error: No se cargaron imágenes. Verifique las rutas y el formato de los archivos.")
    #     exit()
    # # Visualizar algunas imágenes para comprobar
    # #visualizar_imagenes(train_images, train_labels, num=5)

    # # Convertir las etiquetas a números usando LabelEncoder
    # label_encoder = LabelEncoder()
    # combined_labels = train_labels + validation_labels
    # label_encoder.fit(combined_labels)
    
    # train_labels_encoded = label_encoder.transform(train_labels)
    # validation_labels_encoded = label_encoder.transform(validation_labels)

    # # Organizar imágenes en diccionario para LdaNormalBayesClassifier
    # train_images_dict = organizar_imagenes_en_diccionario(train_images, train_labels_encoded)
    # validation_images_dict = organizar_imagenes_en_diccionario(validation_images, validation_labels_encoded)
    # # 2) Load training and validation data
    # # También habrá que extraer los vectores de características asociados (en la parte básica 
    # # umbralizar imágenes, pasar findContours y luego redimensionar)
    # # 2) Preparar datos de entrenamiento y validación
    # # gt_labels (ground truth) es el mismo que validation_labels
    # # Eliminamos el gt_labels para el ejercicio dos, ya que lo bamos a gestionar con o sin dimensionalidad 
    # #gt_labels = validation_labels
    # # Aplicar reducción de dimensionalidad si es necesario
    # if args.dim_reduction == "none":
    #     train_features = train_images
    #     validation_features = validation_images
    # elif args.dim_reduction == "pca":
    #     pca = PCA(n_components=50)
    #     train_features = pca.fit_transform(train_images)
    #     validation_features = pca.transform(validation_images)
    # else:
    #     print(f"Error: Método de reducción de dimensionalidad {args.dim_reduction} no soportado.")
    #     exit()
    # # 3) Entrenar clasificador
    # # Declaramos los diferentes tipos de clasificador "ESTO ES UNA PARTE DEL EJERICIO 2"
    # # Si lo ejecutamos en visual estudio, dar los campos de "classifier", directamente en el main 
    # # Antes de la ejecución.
    # if args.classifier == "svc": #--> Por defecto
    #     # svc --> Support Vector Classifier
    #     classifier = SVC()
    #     #classifier.fit(train_images, train_labels)
    #     classifier.fit(train_features, train_labels_encoded)
    #     # 4) Ejecutar el clasificador sobre los datos de validación
    #     #predicted_labels = classifier.predict(validation_images)
    #     predicted_labels = classifier.predict(validation_features)
    #     # 5) Evaluar los resultados
    #     #accuracy = accuracy_score(gt_labels, predicted_labels)
    #     accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
    #     print("Accuracy = ", accuracy)
    #     # Mostrar matriz de confusión
    #     #cm = confusion_matrix(gt_labels, predicted_labels)
    #     #plot_confusion_matrix(cm)
    #     #plt.show()
    #     # Mostrar matriz de confusión más detallada
    #     cm = confusion_matrix(validation_labels_encoded, predicted_labels)
    #     #plot_confusion_matrix_detallada(cm)
    #     plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {args.classifier}, {args.feature}, {args.dim_reduction}")
    #     plt.show()
    # elif args.classifier == "knn":
    #     # K-Nearest Neight
    #     classifier = KNeighborsClassifier()
    #     classifier.fit(train_features, train_labels_encoded)
    #     # 4) Ejecutar el clasificador sobre los datos de validación
    #     #predicted_labels = classifier.predict(validation_images)
    #     predicted_labels = classifier.predict(validation_features)
    #     # 5) Evaluar los resultados
    #     #accuracy = accuracy_score(gt_labels, predicted_labels)
    #     accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
    #     print("Accuracy = ", accuracy)
    #     # Mostrar matriz de confusión
    #     #cm = confusion_matrix(gt_labels, predicted_labels)
    #     #plot_confusion_matrix(cm)
    #     #plt.show()
    #     # Mostrar matriz de confusión más detallada
    #     cm = confusion_matrix(validation_labels_encoded, predicted_labels)
    #     #plot_confusion_matrix_detallada(cm)
    #     plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {args.classifier}, {args.feature}, {args.dim_reduction}")
    #     plt.show()
    # elif args.classifier == "dtree":
    #     # Decission Tree
    #     classifier = DecisionTreeClassifier()
    #     classifier.fit(train_features, train_labels_encoded)
    #     # 4) Ejecutar el clasificador sobre los datos de validación
    #     #predicted_labels = classifier.predict(validation_images)
    #     predicted_labels = classifier.predict(validation_features)
    #     # 5) Evaluar los resultados
    #     #accuracy = accuracy_score(gt_labels, predicted_labels)
    #     accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
    #     print("Accuracy = ", accuracy)
    #     # Mostrar matriz de confusión
    #     #cm = confusion_matrix(gt_labels, predicted_labels)
    #     #plot_confusion_matrix(cm)
    #     #plt.show()
    #     # Mostrar matriz de confusión más detallada
    #     cm = confusion_matrix(validation_labels_encoded, predicted_labels)
    #     plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {args.classifier}, {args.feature}, {args.dim_reduction}")
    #     plt.show()
    # elif args.classifier == "lda_normal_bayes":
    #     classifier = LdaNormalBayesClassifier(ocr_char_size=(25, 25))
    #     classifier.train(train_images_dict)

    #     predicted_labels = []
    #     for label, images in validation_images_dict.items():
    #         for img in images:
    #             predicted_labels.append(classifier.predict(img))

    #     accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
    #     print("Accuracy = ", accuracy)

    #     cm = confusion_matrix(validation_labels_encoded, predicted_labels)
    #     plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {args.classifier}, {args.feature}, {args.dim_reduction}")
    #     plt.savefig(f"confusion_matrix_{args.classifier}.png")
    #     plt.show()
    # else:
    #     print(f"Error: Clasificador {args.classifier} no soportado.")
    #     exit()
    
    
    # # Pruebas de Ejecuciones con todas las posibilidades
    # args = parser.parse_args()

    # train_path = os.path.abspath(args.train_path)
    # validation_path = os.path.abspath(args.validation_path)
    # print(f"Train path: {train_path}")
    # print(f"Validation path: {validation_path}")

    # # Posibles configuraciones
    # classifiers = {
    #     "svc": SVC(),
    #     "knn": KNeighborsClassifier(),
    #     "dtree": DecisionTreeClassifier()
    # }

    # features = {
    #     "raw": pre_procesado_image,
    #     "hog": extract_hog_features
    # }

    # dim_reductions = {
    #     "none": None,
    #     "pca": PCA(n_components=50)
    # }

    # for feature_name, feature_extractor in features.items():
    #     train_images, train_labels = carga_imagenes_carpeta(train_path, feature_extractor)
    #     validation_images, validation_labels = carga_imagenes_carpeta(validation_path, feature_extractor)
    #     print(f"Loaded {len(train_images)} training images and {len(validation_images)} validation images.")

    #     if len(train_images) == 0 or len(validation_images) == 0:
    #         print("Error: No se cargaron imágenes. Verifique las rutas y el formato de los archivos.")
    #         exit()

    #     label_encoder = LabelEncoder()
    #     combined_labels = train_labels + validation_labels
    #     label_encoder.fit(combined_labels)
        
    #     train_labels_encoded = label_encoder.transform(train_labels)
    #     validation_labels_encoded = label_encoder.transform(validation_labels)

    #     for dim_red_name, dim_reducer in dim_reductions.items():
    #         if dim_reducer is None:
    #             train_features = train_images
    #             validation_features = validation_images
    #         else:
    #             dim_reducer.fit(train_images)
    #             train_features = dim_reducer.transform(train_images)
    #             validation_features = dim_reducer.transform(validation_images)

    #         for clf_name, classifier in classifiers.items():
    #             print(f"Training classifier: {clf_name}, Feature: {feature_name}, Dimensionality reduction: {dim_red_name}")

    #             classifier.fit(train_features, train_labels_encoded)
    #             predicted_labels = classifier.predict(validation_features)

    #             accuracy = accuracy_score(validation_labels_encoded, predicted_labels)
    #             print(f"Accuracy = {accuracy}")

    #             cm = confusion_matrix(validation_labels_encoded, predicted_labels)
    #             #plot_confusion_matrix(cm, title=f"Confusion matrix: {clf_name}, {feature_name}, {dim_red_name}")
    #             plot_confusion_matrix_detallada(cm, title=f"Confusion matrix: {clf_name}, {feature_name}, {dim_red_name}")
    #             plt.show()
    #             #cm = confusion_matrix(validation_labels_encoded, predicted_labels)
    #             #filename = f"confusion_matrix_{clf_name}_{feature_name}_{dim_red_name}.png"
    #             #plot_confusion_matrix(cm, title=f"Confusion matrix: {clf_name}, {feature_name}, {dim_red_name}", filename=filename)
    #             #print(f"Confusion matrix saved as {filename}")
    
    # BASES-ENLACE: https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
    # Con la prueba de 5 imagenes, no irá indicando una matriz de confusión
    # - La matriz de confusión muestra la comparación entre las etiquetas verdaderas y las estiquetas
    # predichas por el modelo:
    #     - Eje vertical (True label): Representa las clases reales de los datos de validación
    #     - Eje horizontal (Predicted label): Representa las clases predichas por el modelo
    #     - Valores en la matriz: Cada celda (i,j) representa la cantidad de veces que una clase
    #     i fue clasificada como clase j
    # *Mejoras en la creación de la matriz de confusión**
    # SALIDAS PARA EL EJERCICIO 1 Clasificador Propio:
    # Train path: C:\Users\david\OneDrive\Escritorio\Universidad\Universidad_2023-2024\SegundoCuatri\VisionArtificial\VisionArtificial\practica2_VA_23-24\Practica2_Datos_Alumnos\Practica2_Datos_Alumnos\train_ocr
    # Validation path: C:\Users\david\OneDrive\Escritorio\Universidad\Universidad_2023-2024\SegundoCuatri\VisionArtificial\VisionArtificial\practica2_VA_23-24\Practica2_Datos_Alumnos\Practica2_Datos_Alumnos\validation_ocr
    # Training classifier: lda_normal_bayes, Feature: raw, Dimensionality reduction: none
    # Total de imágenes cargadas: 38750
    # Total de imágenes cargadas: 15500
    # Loaded 38750 training images and 15500 validation images.
    # Accuracy =  0.9584516129032258

    # EJERCICIO 2 - CAMBIOS:
    # Método de extracciones de características:
    # - Utilizando "raw" --> Utiliza la imagen preprocesada directamente
    # - Utilizando "hog" --> Se basa en la utilización de Histograma Orientado a gradiantes
    #     (HOG) como características https://scikit-image.org/docs/dev/api/skimage.feature.html
    # Método de reducción de dimensionalidad:
    # - Sin utilizar una reducción dimensionalidad
    # - 'pca' --> Utilizando "principal component Analysis (PCA) con tan finalidad de reducir la dimesnionalidad
    # a 50 componentes ENLACE: https://scikit-learn.org/stable/modules/decomposition.html 
    # Clasificadores : SVC-KNN-DTREE

    # EJECUCION RESULTADOS CARPETA --> Imágenes Ejecución:
    # Train path: C:\Users\david\OneDrive\Escritorio\Universidad\Universidad_2023-2024\SegundoCuatri\VisionArtificial\VisionArtificial\practica2_VA_23-24\Practica2_Datos_Alumnos\Practica2_Datos_Alumnos\train_ocr
    # Validation path: C:\Users\david\OneDrive\Escritorio\Universidad\Universidad_2023-2024\SegundoCuatri\VisionArtificial\VisionArtificial\practica2_VA_23-24\Practica2_Datos_Alumnos\Practica2_Datos_Alumnos\validation_ocr
    # Total de imágenes cargadas: 38750
    # Total de imágenes cargadas: 15500
    # Loaded 38750 training images and 15500 validation images.
    # Training classifier: svc, Feature: raw, Dimensionality reduction: none
    # Accuracy = 0.9615483870967741
    # Training classifier: knn, Feature: raw, Dimensionality reduction: none
    # Accuracy = 0.9536774193548387
    # Training classifier: dtree, Feature: raw, Dimensionality reduction: none
    # Accuracy = 0.9414193548387096
    # Training classifier: svc, Feature: raw, Dimensionality reduction: pca
    # Accuracy = 0.9547096774193549
    # Training classifier: knn, Feature: raw, Dimensionality reduction: pca
    # Accuracy = 0.9509032258064516
    # Training classifier: dtree, Feature: raw, Dimensionality reduction: pca
    # Accuracy = 0.8941290322580645
    # Total de imágenes cargadas: 38750
    # Total de imágenes cargadas: 15500
    # Loaded 38750 training images and 15500 validation images.
    # Training classifier: svc, Feature: hog, Dimensionality reduction: none
    # Accuracy = 0.9489032258064516
    # Training classifier: knn, Feature: hog, Dimensionality reduction: none
    # Accuracy = 0.9418709677419355
    # Training classifier: dtree, Feature: hog, Dimensionality reduction: none
    # Accuracy = 0.8618709677419355
    # Training classifier: svc, Feature: hog, Dimensionality reduction: pca
    # Accuracy = 0.9474838709677419
    # Training classifier: knn, Feature: hog, Dimensionality reduction: pca
    # Accuracy = 0.9374838709677419
    # Training classifier: dtree, Feature: hog, Dimensionality reduction: pca
    # Accuracy = 0.8220645161290323
    # COMENTAR: Ctrl + K + C y DESCOMENTAR: Ctrl + K + U
    
