################################################################################
# Computer Vision
# Ernesto Martinez del Pino
# ernestomar1997@correo.ugr.es
################################################################################

# Importacion de Bibliotecas
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Objetos SURF y SIFT
SIFT = cv2.xfeatures2d.SIFT_create()
SURF = cv2.xfeatures2d.SURF_create()

# Funcion para pintar las imagenes
def showImg( imagen, nombre ):
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
    cv2.imshow(nombre, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# EJERCICIO 1

# (a) Obtener un conjunto de 1000 puntos SURF y SIFT
img = cv2.imread("./imagenes/Yosemite1.jpg", 0)
#showImg(img, "Imagen Original")

# No es necesario jugar con los parametros de detect puesto que ya son mas de
# 1000 puntos.
points_SIFT = SIFT.detect(img,None)
print "Numero de puntos SIFT: {}".format(len(points_SIFT))
points_SURF = SURF.detect(img,None)
print "Numero de puntos SURF: {}".format(len(points_SURF))



# (b) Identificar puntos por octava y mostrar los circulos con el radio proporcional
img = cv2.drawKeypoints(img, points_SIFT,outImage=None, 
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#showImg(img, "1")
img = cv2.drawKeypoints(img, points_SURF,outImage=None, 
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#showImg(img, "1")
print points_SIFT
print points_SURF

kp, des = SURF.detectAndCompute(img,None)
print len(des[3])

print "hola"