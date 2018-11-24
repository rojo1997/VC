################################################################################
# Computer Vision                                                              #
# Ernesto Martinez del Pino                                                    #
# ernestomar1997@correo.ugr.es                                                 #
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

# Funcion para obtener la octava y la escala en sift
def unpackSIFTOctave(kpt):
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)

# (c) Calcular los decriptores a partir de los vectores de puntos
des_SIFT = SIFT.compute(img, points_SIFT)
des_SURF = SURF.compute(img, points_SURF)
print des_SIFT
print des_SURF

# EJERCICIO 2

# (a) Mostar las imagenes en un canvas y pintar las lineas de los keypoints

# Funcion que establece las equivalencias por fuerza bruta entre 2 imagenes
def BruteForceCC (img1, img2, num = 100, best = False):
    # Calculamos puntos y descriptores de ambas imagenes
    kp1, des1 = SIFT.detectAndCompute(img1.copy(), None)
    kp2, des2 = SIFT.detectAndCompute(img2.copy(), None)
    # Aplicar fuerza bruta y verificacion
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    # Ordenamos por la distancia
    if best:
        matches = sorted(matches, key = lambda x:x.distance)
    # Generamos la nueva imagen
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:num], None, flags=2)
    # Devolvemos la imagen
    return (img3)
# Funcion que establece las equivalencias usando KNN con filtro
def LA2NN (img1, img2, num = 100, best = False, dist = 0.7):
    # Calculamos puntos y descriptores de ambas imagenes
    kp1, des1 = SIFT.detectAndCompute(img1.copy(), None)
    kp2, des2 = SIFT.detectAndCompute(img2.copy(), None)
    # Aplicar KNN
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Ordenamos por la distancia
    #if best:
    #    matches = sorted(matches, key = lambda x:x.distance)
    # Filtro
    goodMatches = []
    for m, n in matches:
        if m.distance < dist * n.distance:
            goodMatches.append(m)
    goodMatches = goodMatches[:min(100,len(goodMatches)-1)]
    print len(goodMatches)
    # Generamos la nueva imagen
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,goodMatches,None,flags=2)
    return (img3)


img1 = cv2.imread("./imagenes/Yosemite1.jpg", 1)
img2 = cv2.imread("./imagenes/Yosemite2.jpg", 1)
img3 = BruteForceCC(img1, img2)
showImg(img3, "Concordancias FuerzaBruta")
img3 = LA2NN(img1, img2)
showImg(img3, "Concordancias KNN")
img3 = BruteForceCC(img1, img2, best = True)
showImg(img3, "Concordancias FuerzaBruta BEST")

# (b) Comentario de la inspeccion ocular

# (c) Comparacion de tecnicas

# EJERCICIO 3 y 4
def joinImg2 (img1, img2):
    # Calculamos los puntos y los decriptores
    kp1, des1 = SIFT.detectAndCompute(img1, None)
    kp2, des2 = SIFT.detectAndCompute(img2, None)
    # Aplicar fuerza bruta y verificacion
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = np.array(bf.match(des1, des2))
    matches = sorted(matches, key = lambda x:x.distance)[:100]
    # Cambiamos la estructura del dato
    src = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    # Encontramos la homografia
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    # Aplicamos la homografia
    img2 = addBorders(img2, cv2.BORDER_CONSTANT, 200, (255,255,255))
    showImg (img2)
    img2 = cv2.warpPerspective(img1, M, dsize=(img2.shape[1],img2.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
    # Devolvemos la img3 y la homografia
    return (img2.copy(), M)

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

        h,w,c = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,(255,255,255),1, cv2.LINE_AA)
        return img2, mask
    else:
        return False

def addBorders(img, bordertype, bordersize, value):
    row, col = img.shape[:2]
    border = cv2.copyMakeBorder(img, top=0, bottom=bordersize,
    left=0, right=bordersize, borderType= bordertype, value=value )
    return border



img1 = cv2.imread("./imagenes/mosaico002.jpg", 1)
img2 = cv2.imread("./imagenes/mosaico003.jpg", 1)
img3 = cv2.imread("./imagenes/mosaico004.jpg", 1)
img4 = cv2.imread("./imagenes/mosaico005.jpg", 1)
img5 = cv2.imread("./imagenes/mosaico006.jpg", 1)
img6 = cv2.imread("./imagenes/mosaico007.jpg", 1)

imgs = [img1,img2,img3,img4,img5,img5]

#img1 = addBorders(img1, cv2.BORDER_CONSTANT, 200, (255,255,255))
showImg (img1)

# Para ejercicio 1
imgJOIN, M = joinImg2 (img1, img2)
showImg (imgJOIN, "JOIN IMGS")

def joinNimg (imgs):
    original = imgs[0]
    for img in imgs:
        original,M = joinImg2(original,img)
    return (original)

# Para ejercicio 2
imgJOIN = joinNimg(imgs)
showImg (imgJOIN, "JOIN IMGS")

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

