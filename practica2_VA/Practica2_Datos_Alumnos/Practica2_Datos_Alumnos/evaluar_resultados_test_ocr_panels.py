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

# #Ejercicio 3 --------------------EJEMPLO BASE DE EJERCICIO 3 ---------------------------------
# ------------SALIDA EJERCICIO 3-----------------
# PROCESANDO testing_ocr ------------------------------
# {'00003_0.png': ['A52+Verin32+Benavente192+Madrid459'], '00006_0.png': ['A52+Verin+Benavente+Madrid'], '00007_0.png': ['A52+Verin27'], '00011_0.png': ['1000m'], '00011_1.png': ['1'], '00011_2.png': ['Monterrel+Verin+Vilaza'], '00013_0.png': ['A52+Benavente+Madrid'], '00016_0.png': ['500m'], '00016_2.png': ['N532+Mandin+Feces+deAbaixo'], '00017_0.png': ['A75+ChavesP'], '00021_0.png': ['1000m'], '00021_1.png': ['479'], '00021_2.png': ['NVI+Castroverde+Corgo'], '00022_1.png': ['479'], '00022_2.png': ['Castroverde+Corgo'], '00024_0.png': ['A6+Becerrea21+Ponferrada98+Madrid478'], '00028_0.png': ['1000m'], '00028_2.png': ['LU710+NeiradeRei+Baleira'], '00031_0.png': ['A6+Ponferrada75+Benavente195+Madrid455'], '00032_0.png': ['1000m'], '00032_1.png': ['451'], '00032_2.png': ['NVI+Becerrea+LU722+NaviadeSuerna+Cervantes'], '00033_0.png': ['A6+Ponferrada+Benavente+Madrid'], '00034_0.png': ['500m'], '00034_1.png': ['444'], '00034_2.png': ['NVUvAsNogais'], '00036_0.png': ['A6+Ponferrada+Benavente+Madrid'], '00038_1.png': ['432'], '00038_2.png': ['Pedrafita+doCebreiro'], '00040_0.png': ['TUNELDE+LAESCRITA'], '00040_1.png': ['L154m'], '00041_0.png': ['A6+Ponferrada+Benavente+Madrid'], '00042_2.png': ['NVI+Villafranca+deiBierzo+Corullon+Cacabelos'], '00043_0.png': ['A6+Ponferrada+Benavente+Madrid'], '00044_2.png': ['N120+ToraldelosVados+OBarcodeValdeorras+Ourense'], '00045_0.png': ['A6+Ponferrada+Benavente+Madrid'], '00046_1.png': ['399'], '00046_2.png': ['LE52O7+Carracedelo+Carracedo+delMonasterio'], '00047_0.png': ['A6+Ponferrada1O+Madrid398'], '00049_0.png': ['A6+Astorga+Madrid+N120Leon'], '00051_0.png': ['A6+Astorga49+Madrid376'], '00052_1.png': ['372'], '00052_2.png': ['N6+Bembibre+Toreno'], '00053_1.png': ['361'], '00053_2.png': ['Folgoso+delaRibera+AlbaresdelaRibera'], '00055_0.png': ['A6+Astorga+Madrid347'], '00055_1.png': ['N120+Leon68'], '00056_2.png': ['N6+Combarros'], '00057_1.png': ['329'], '00057_2.png': ['N6+Astorga'], '00058_0.png': ['A6+Astorga+N12OLeon+Madrid'], '00061_0.png': ['AP71+Leon+Oviedo'], '00062_0.png': ['A66+Benavente+Madrid+Burgos'], '00062_1.png': ['LE30+Leon'], '00062_2.png': ['144'], '00063_0.png': ['LE3O+Leon'], '00063_1.png': ['N120Leon+AP68+Campomanes+Oviedo'], '00064_0.png': ['LE30+Leon+N630N601'], '00067_0.png': ['LE30+Leon+N601Valladolid+N630Oviedo'], '00067_1.png': ['400m'], '00068_0.png': ['LE30+Leon+N601Valladolid+N630Oviedo'], '00070_0.png': ['RondaSur+N630+Oviedo+N601+Valladolid'], '00074_2.png': ['N634+Vilalba'], '00083_0.png': ['E70A8+Abadin5+Oviedo191'], '00086_2.png': ['N634+MondoÃ±edo+Lourenza'], '00093_1.png': ['E1AP9+ACoruÃ±a+Pontevedra'], '00094_1.png': ['E1AP9+ACoruÃ±a+Pontevedra'], '00095_2.png': ['SanMarcos'], '00096_2.png': ['SanMarcos+montedogozo'], '00097_0.png': ['Villalba+A54Oviedo+Lugo+LavacollaSCQ'], '00098_2.png': ['Sionlla'], '00100_0.png': ['500m'], '00100_2.png': ['SC21aeroporto+AC250Siguero+E1AP9N550+ACoruÃ±a'], '00101_0.png': ['A54+APedrouzo+Oviedo+Lugo']}
# {'00003_0.png': [''], '00006_0.png': [''], '00007_0.png': [''], '00011_0.png': [''], '00011_1.png': [''], '00011_2.png': ['iii+ii'], '00013_0.png': ['v'], '00016_0.png': [''], '00016_2.png': [''], '00017_0.png': [''], '00021_0.png': [''], '00021_1.png': [''], '00021_2.png': [''], '00022_1.png': [''], '00022_2.png': [''], '00024_0.png': [''], '00028_0.png': [''], '00028_2.png': ['iii+i'], '00031_0.png': ['ilii+i'], '00032_0.png': [''], '00032_1.png': [''], '00032_2.png': ['li'], '00033_0.png': ['v'], '00034_0.png': [''], '00034_1.png': ['iiiii'], '00034_2.png': ['iiiii'], '00036_0.png': ['v'], '00038_1.png': [''], '00038_2.png': ['i'], '00040_0.png': [''], '00040_1.png': [''], '00041_0.png': [''], '00042_2.png': ['iii+ii'], '00043_0.png': [''], '00044_2.png': ['ii+ii'], '00045_0.png': ['v'], '00046_1.png': [''], '00046_2.png': ['i'], '00047_0.png': [''], '00049_0.png': [''], '00051_0.png': [''], '00052_1.png': [''], '00052_2.png': [''], '00053_1.png': [''], '00053_2.png': [''], '00055_0.png': [''], '00055_1.png': ['i'], '00056_2.png': [''], '00057_1.png': [''], '00057_2.png': [''], '00058_0.png': [''], '00061_0.png': [''], '00062_0.png': ['i'], '00062_1.png': [''], '00062_2.png': [''], '00063_0.png': ['v'], '00063_1.png': [''], '00064_0.png': [''], '00067_0.png': [''], '00067_1.png': [''], '00068_0.png': [''], '00070_0.png': [''], '00074_2.png': [''], '00083_0.png': [''], '00086_2.png': ['i'], '00093_1.png': [''], '00094_1.png': [''], '00095_2.png': ['ii+i'], '00096_2.png': ['v'], '00097_0.png': [''], '00098_2.png': [''], '00100_0.png': [''], '00100_2.png': [''], '00101_0.png': ['v']}
# [34, 26, 11, 5, 1, 19, 19, 4, 26, 11, 5, 3, 21, 3, 17, 36, 5, 20, 36, 5, 3, 41, 29, 4, 5, 11, 29, 3, 19, 17, 5, 30, 40, 30, 46, 29, 3, 41, 25, 26, 22, 3, 18, 3, 36, 20, 11, 12, 3, 10, 26, 16, 26, 9, 3, 9, 31, 18, 35, 
# 4, 35, 36, 12, 23, 24, 25, 25, 9, 21, 36, 7, 4, 45, 24]
# hist= [ 1 13  6  1  3  5  2  0  3  2  5  3  3  8  0  3  3  0  3  5  0  1  2  0
#   2]
# bin_edges= [ 1.   2.8  4.6  6.4  8.2 10.  11.8 13.6 15.4 17.2 19.  20.8 22.6 24.4
#  26.2 28.  29.8 31.6 33.4 35.2 37.  38.8 40.6 42.4 44.2 46. ]
# 24