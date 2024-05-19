# @brief LdaNormalBayesClassifier


# A continuación se presenta un esquema de la clase necesaria para implementar el clasificador
# propuesto en el Ejercicio1 de la práctica. Habrá que terminarla

import os
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .ocr_classifier import OCRClassifier

class LdaNormalBayesClassifier(OCRClassifier):
    """
    Classifier for Optical Character Recognition using LDA and the Bayes with Gaussian classfier.
    """

    def __init__(self, ocr_char_size):
        super().__init__(ocr_char_size)
        self.lda = None
        self.classifier = None

    def train(self, images_dict):
        """.
        Given character images in a dictionary of list of char images of fixed size, 
        train the OCR classifier. The dictionary keys are the class of the list of images 
        (or corresponding char).

        :images_dict is a dictionary of images (name of the images is the key)
        """

        # Take training images and do feature extraction
# ---------------------------------------------------------------------------------------------------------------------

        preprocessed_images = []

        for filename in os.listdir(images_dict):
            image_path = os.path.join(images_dict, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Error handling (optional): Check if image is loaded successfully
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            # Apply adaptive thresholding
            _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours and bounding rectangles
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get bounding rectangle coordinates
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the character image using the bounding rectangle
                cropped_image = image[y:y + h, x:x + w]

                # Resize the cropped image to a fixed size (e.g., 25x25 pixels)
                resized_image = cv2.resize(cropped_image, (25, 25), interpolation=cv2.INTERAREA)

                # Convert the resized image to a flattened 1D array with values between 0 and 255
                feature_vector = resized_image.flatten() / 255

                # Append the feature vector to the list of preprocessed images
                preprocessed_images.append(feature_vector)        

# Convertir la matriz con los niveles de gris del carácter ya
# redimensionada, con un tamaño de 25x25 y valores entre 0 y 255, en una matriz con una sola
# fila y 625 columnas. Cada una de estas matrices filas se usará como vector de características
# asociado al carácter de entrenamiento.



#--------------------------------------------------------------------------------------------------------------------------

        X = ... # Feature vectors by rows
        y = ... # Labels for each row in X 

        # Perform LDA training

        # Perform Classifier training

        return samples, labels

    def predict(self, img):
        """.
        Given a single image of a character already cropped classify it.

        :img Image to classify
        
        """

#-------------------------------------------------------------------------------------------------------------------
        # Load image in grayscale mode
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # Apply Otsu's thresholding to convert to binary image
        _, imagen_umbralizada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#--------------------------------------------------------------------------------------------------------------------
   
        y = ... # Obtain the estimated label by the LDA + Bayes classifier

        return int(y)



