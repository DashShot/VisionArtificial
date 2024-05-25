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
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classifier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        """
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).
        """
        X, y = self._extract_features_and_labels(images_dict)

        # Perform LDA training
        self.lda = LinearDiscriminantAnalysis()
        X_reduced = self.lda.fit_transform(X, y)

        # Train classifier
        self.classifier = cv2.ml.NormalBayesClassifier_create()
        self.classifier.train(np.float32(X_reduced), cv2.ml.ROW_SAMPLE, np.int32(y))

    def predict(self, img):
        """
        Given a single image of a character already cropped classify it.

        :img Image to classify
        """
        feature = self._extract_feature(img)
        feature_reduced = self.lda.transform([feature])
        _, result = self.classifier.predict(np.float32(feature_reduced))
        return int(result[0, 0])

    def _extract_features_and_labels(self, images_dict):
        """
        Extract features and labels from the images dictionary.
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
        Extract a single feature from an image.
        """
        return img.flatten()


