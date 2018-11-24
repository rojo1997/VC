################################################################################
# Computer Vision
# Ernesto Martinez del Pino
# ernestomar1997@correo.ugr.es
################################################################################

# Importacion de Bibliotecas
import numpy as np
import cv2
import scipy.integrate as integrate
import scipy.ndimage as filters
from pylab import *
from matplotlib import pyplot as plt

# Objetos SURF y SIFT
SIFT = cv2.xfeatures2d.SIFT_create()
SURF = cv2.xfeatures2d.SURF_create()

# Funcion para pintar las imagenes
def showImg( imagen, nombre = ""):
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
showImg(img, "1")
img = cv2.drawKeypoints(img, points_SURF,outImage=None, 
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
showImg(img, "1")
print points_SIFT
print points_SURF

# (c) Calcular los decriptores a partir de los vectores de puntos
des_SIFT = SIFT.compute(img, points_SIFT)
des_SURF = SURF.compute(img, points_SURF)
print des_SIFT
print des_SURF

# EJERCICIO 2

# (a) Mostar las imagenes en un canvas y pintar las lineas de los keypoints
def equiv (img1, img2, cross, dist):
    # Calculamos puntos y descriptores de ambas imagenes
    kp1, des1 = SIFT.detectAndCompute(img1.copy(), None)
    kp2, des2 = SIFT.detectAndCompute(img2.copy(), None)
    # Aplicar KNN
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < (dist * n.distance):
            matchesMask[i]=[1,0]
    img3 = None
    draw_params=dict(matchesMask=matchesMask)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2,**draw_params)
    return (img3)


img1 = cv2.imread("./imagenes/Yosemite1.jpg", 0)
img2 = cv2.imread("./imagenes/Yosemite2.jpg", 0)
img3 = equiv(img1, img2, True, 0.7)
showImg(img3, "Concordancias")
# (b) Comentario de la inspeccion ocular

# (c) Comparacion de tecnicas

# EJERCICIO 3
def joinImg (img1, img2):
    # Minimo numero de puntos que aceptamos
    MIN_MATCH_COUNT = 10

    # Calculamos los puntos y los decriptores
    kp1, des1 = SIFT.detectAndCompute(img1.copy(), None)
    kp2, des2 = SIFT.detectAndCompute(img2.copy(), None)

    # Parametros del match
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        return img2
    else:
        return False


img1 = cv2.imread("./imagenes/mosaico002.jpg", 0)
img2 = cv2.imread("./imagenes/mosaico003.jpg", 0)
img3 = cv2.imread("./imagenes/mosaico004.jpg", 0)
img4 = cv2.imread("./imagenes/mosaico005.jpg", 0)

img12 = joinImg (img1, img2)
img123 = joinImg (img3, img12)
img1234 = joinImg (img4, img123)
showImg (img1234, "Hola")
#p2, p3 = getpoints (img2, img3)
#pt1, pt2 = getpoints (img1, img2)


h1, status = cv2.findHomography(p1, p2, cv2.RANSAC)
h2 = cv2.findHomography(p2, p3, cv2.RANSAC)
h3 = cv2.findHomography(pt1, pt2, cv2.RANSAC)

print h1
print status

#kp, des = SURF.detectAndCompute(img,None)
#print len(des[3])
#getNOctaves()
#getNOctaveLayers()

# BONUS

# EJERCICIO 1
img = cv2.imread("./imagenes/cuadro1.jpg", 0)


def HarrisPoints (img, lavel):
    thresh = 255
    dst = cv2.cornerHarris (img, 2, 3, 0.04)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    print dst_norm_scaled
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv2.rectangle(dst_norm_scaled, (j-label,i-label), (j+label,i+label), (0), 2)
    return dst_norm_scaled

def Pyr(img, tam):
    vector = [img]
    for i in range(tam):
        vector.append(cv2.pyrDown(cv2.GaussianBlur(vector[i],(0,0), sigmaX=1)))
        showImg (HarrisPoints(vector[i], i), "Prueba")
    return (vector)

#Pyr(img, 5)

