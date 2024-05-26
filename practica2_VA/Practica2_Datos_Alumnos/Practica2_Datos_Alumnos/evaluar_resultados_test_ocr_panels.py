import csv
import math
import matplotlib.pyplot as plt
import numpy as np


def read_csv_file(file, delim=";"):
    """

    """
    panels_info = dict()
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            #print(row)
            image_name = row[0]
            try:
                panel_text = row[7]
            except:
                panel_text = "" # The OCR could fail in this image
            if panels_info.get(image_name) is None:
                panels_info[image_name] = [panel_text]
            else:
                print('image=', image_name)
                l = panels_info[image_name]
                l.append([panel_text])
                panels_info[image_name] = l

            line_count += 1
    return panels_info


def levenshtein_distance(str1, str2):
    """

    """
    d = dict()
    for i in range(len(str1) + 1):
        d[i] = dict()
        d[i][0] = i
    for i in range(len(str2) + 1):
        d[0][i] = i
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + (not str1[i - 1] == str2[j - 1]))

    return d[len(str1)][len(str2)]


def plot_recognition_distance(p_gt, p):
    """

    """
    #iou_all = []
    norm_dist_all = []
    txt_distance_all = []
    for img_name in p_gt:
        p_info_gt = p_gt[img_name]

        if p.get(img_name) is None:  # Not found a panel in this image.
            txt_distance_all.append(-1) # -1 is not found panel
            continue

        p_info = p[img_name]

        # By now we assume only one detection for each image
        plate_gt = p_info_gt[0]
        if len(p_info) >= 1:  # if we have at least one detection
            plate = p_info[0]

            txt_distance = levenshtein_distance(plate_gt, plate)
            txt_distance_all.append(txt_distance)

    print(txt_distance_all)

    # Plot histogram
    plt.figure()
    hist, bin_edges = np.histogram(np.array(txt_distance_all),  bins=25, density=False)
    plt.step(bin_edges[:-1], hist, where='mid')
    plt.title("Distancia de Levenshtein: texto panel reconocido vs real")
    plt.ylabel("Núm. imágenes")
    plt.xlabel('Distancia de edición (en "número de operaciones")')
    plt.show()
    print("hist=", hist)
    print("bin_edges=", bin_edges)
    print(hist[0:5].sum())
    


if __name__ == "__main__":

    print('PROCESANDO testing_ocr ------------------------------')
    panels_gt = read_csv_file('./test_ocr_panels/gt.txt')
    print(panels_gt)

    panels = read_csv_file('./resultado.txt')
    print(panels)

    plot_recognition_distance(panels_gt, panels)

# #Ejercicio 3 RANSAC
# MIN_MATCH_COUNT = 4
# img2 = cv.imread('meninas_museo2.jpeg',0)          # imagen de query
# img1 = cv.imread('meninas_plantilla.jpeg',0) # imagen de train
# img1 = cv.resize(img1, (2*img1.shape[1],2*img1.shape[0]))
# print(img1.shape)
# print(img2.shape)

# # Iniciar el detector y descriptor de puntos de interés ORB (Oriented FAST and Rotated BRIEF).
# # ORB es básicamente una combinación del detector FAST y el descriptor BRIEF.
# # ORB funciona igual de bien que SIFT en la detección (y mejor que SURF), siendo dos órdenes de magnitud más rápido. 
# detector = cv.ORB_create()
# descriptor = cv.ORB_create()
#Cambiar por mejor Calsificador




# # Encontrar los puntos de interés y los descriptores
# kp1 = detector.detect(img1)
# kp1, des1 = descriptor.compute(img1,kp1)
# kp2 = detector.detect(img2) 
# kp2, des2 = descriptor.compute(img2,kp2)

# # Matcher de puntos de interés BFMatcher con distancia L1 para descriptores binarios
# matcher = cv.BFMatcher(normType=cv.NORM_L1)
# matches = matcher.knnMatch(des1,des2,k=2)

# # Guardar todos los "matches buenos"
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)