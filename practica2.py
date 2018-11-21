################################################################################
# Computer Vision
# Ernesto Martinez del Pino
# ernestomar1997@correo.ugr.es
################################################################################

# Importacion de Bibliotecas
import numpy as np
import cv2
from matplotlib import pyplot as plt

def showImg( imagen, nombre ):
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# EJERCICIO 1

# (a) Obtener un conjunto de 1000 puntos SURF y SIFT


print "hola"