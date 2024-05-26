# @brief LdaNormalBayesClassifier


# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminarla

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Clasificador para el Reconocimiento Óptico de Caracteres utilizando LDA y el clasificador Bayes con distribución Gaussiana.
    """

    def __init__(self, ocr_char_size):
        """
        Inicializa el LdaNormalBayesClassifier con el tamaño de carácter dado.

        :param ocr_char_size: Tamaño del carácter OCR.
        """
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        """
        Entrena el clasificador OCR dado un diccionario de imágenes de caracteres.
        Las claves del diccionario son las clases de las listas de imágenes (o el carácter correspondiente).

        :param images_dict: Diccionario con etiquetas como claves y listas de imágenes como valores.
        """
        X, y = self._extract_features_and_labels(images_dict)

        # Realiza el entrenamiento LDA
        self.lda = LinearDiscriminantAnalysis()
        X_reduced = self.lda.fit_transform(X, y)

        # Entrena el clasificador Bayes Normal 
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(np.float32(X_reduced), cv2.ml.ROW_SAMPLE, np.int32(y))

    def predict(self, img):
        """
        Clasifica una imagen dada de un carácter ya recortado.

        :param img: Imagen a clasificar.
        :return: Clase predicha de la imagen.
        """
        feature = self._extract_feature(img)
        feature_reduced = self.lda.transform([feature])
        _, result = self.classifier.predict(np.float32(feature_reduced))
        return int(result[0, 0])

    def _extract_features_and_labels(self, images_dict):
        """
        Extrae características y etiquetas del diccionario de imágenes.

        :param images_dict: Diccionario con etiquetas como claves y listas de imágenes como valores.
        :return: Tupla (X, y) donde X es una lista de características e y es una lista de etiquetas.
        """
        X = []
        y = []
        for label, images in images_dict.items():
            for img in images:
                feature = self._extract_feature(img)
                X.append(feature)
                y.append(label)
        return np.array(X), np.array(y)

    def _extract_feature(self, img):
        """
        Extrae una sola característica de una imagen aplanándola en un array 1D.

        :param img: Imagen de la cual extraer características.
        :return: Array 1D aplanado de la imagen.
        """
        return img.flatten()


