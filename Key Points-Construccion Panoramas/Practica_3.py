# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import random as rng


#########################################################################
############# FUNCIONES LECTURA/PRESENTACIÓN IMÁGENES ###################
#########################################################################

def BRGtoRGB(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def leeImagen(filename,flagColor = 1):
        if flagColor == 0:
            return cv.imread(filename,flagColor).astype(np.float32)
        else:
            return BRGtoRGB(cv.imread(filename,flagColor)).astype(np.float32)

def normaliza(m):
    if len(m.shape) == 3 and m.shape[2] == 3:  # tribanda
        for i in range(3):
            imax, imin = m[:,:,i].max(), m[:,:,i].min()
            if imax == imin:
                m[:,:,i] = 0
            else:
                m[:,:,i] = ((m[:,:,i] - imin)/(imax - imin)) 
    elif len(m.shape) == 2:    # monobanda
        imax, imin = m.max(), m.min()
        if imax == imin:
            m = 0
        else:
            m = ((m - imin)/(imax - imin))
    # Escalamos la matriz
    m *= 255
    return m

def pintaImagen(im,title = "img"):
    
    # Normaliza [0,255] como integer
    img = np.copy(normaliza(im))
    img = np.copy(im)
    if len(img.shape) == 2:
        img = cv.cvtColor(img.astype(np.uint8),cv.COLOR_GRAY2BGR).astype(np.float64)
        img = img.astype(np.uint8)
        img = BRGtoRGB(img)
    else :
        img = img.astype(np.uint8)
   
    
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    cv.waitKey(0)

#########################################################################
##################### FUNCIONES AUXILIARES ##############################
#########################################################################

def convolution2D(img,kx,ky,border):
    """ 
    Convolves img with 2 kernels(one in each axis). Uses different kinds of borders
    """
    # Flip kernel y hace traspuesta
    kx,ky = np.flip(kx).T, np.flip(ky)

    # Convolución por filas y columnas
    blurredR = cv.filter2D(img,-1,kx,  borderType = border)
    blurredC = cv.filter2D(blurredR,-1,ky, borderType = border)
    return blurredC

def gaussian2D(img,sigma,ksize = 0,border = cv.BORDER_CONSTANT):
    if ksize == 0 :
        ksize = int(6*sigma + 1)
    
    kernel = cv.getGaussianKernel(ksize,sigma)
    return convolution2D(img,kernel,kernel,border)

# Función que hace subsample a una imagen
def subsample(img):
    return img[::2, ::2]

# Función que aplica la máscara de dx y dy a una imagen
def maskDerivKernels(img,dx = 1,dy = 1,ksize = 3,border = cv.BORDER_REPLICATE):
    dxdy = cv.getDerivKernels(dx,dy,ksize,normalize = 0)
    return convolution2D(img,dxdy[0],dxdy[1],border)



#########################################################################
#######################    EJERCICIO 1     ##############################
#########################################################################

""" 
Función que devuelve una pirámide Gaussiana a partir de 4 argumentos: 
una imagen dada, un entero que será el número de niveles de la pirámide,
el tamaño del kernel y el sigma del gaussian blur
"""   
def gaussianPyramid(orig,iters,ksize = 3,sigma1 = 1,border = cv.BORDER_DEFAULT):
    pyramid = [orig]
    
    for i in range (0,iters-1):
        blurred = gaussian2D(pyramid[i],sigma1,ksize,border)
        subsampled = subsample(blurred)
        pyramid.append(subsampled)

    return pyramid


""" 
Función que toma como argumentos una imagen de entrada y un valor que será la distancia entre 
máximos locales.
Comprueba para cada pixel de la imagen si los pixeles adyacentes tienen un valor mayor
"""   
def supresionNoMax(im,dist=3):
    
    # Pixeles de las imágenes
    filas=im.shape[0]
    columnas=im.shape[1]
    
    # Realizamos una copia de la imagen
    supre=np.copy(im)
    
    # Para cada pixel de la imagen, comprobamos si los pixeles adyacentes
    # tienen un valor más alto, en cuyo caso ponemos a 0 el pixel en la copia
    for i in range(0,filas):
        for j in range(0,columnas):
            pixel=im[i][j]
            fil_inf=max(i-int((dist-1)/2),0)
            fil_sup=min(i+int((dist-1)/2)+1,filas)
            col_inf=max(j-int((dist-1)/2),0)
            col_sup=min(j+int((dist-1)/2)+1,columnas)
            for k in range(fil_inf,fil_sup):
                for l in range(col_inf,col_sup):
                    if im[k][l]>pixel:
                        supre[i][j]=0.0
                        
    return supre

# Leemos imagenes con las que trabajaremos
y1c = leeImagen("images/Yosemite1.jpg")
y1g = leeImagen("images/Yosemite1.jpg",0)
y2c = leeImagen("images/Yosemite2.jpg")
y2g = leeImagen("images/Yosemite2.jpg",0)



def criterio_harris(l1,l2):
    if (l1+l2 == 0):
        return 0
    return (l1*l2)/(l1+l2)


"""
Función que recibe una imagen con valor 0 en los pixeles no relevantes,
se aplica un alisamiento a la imagen con sigma=4.5 para despues calcular
la orientación del gradiente.
Finalmente se van almacenando en un array de objetos KeyPoint los puntos con
sus orientaciones y su escala para posteriormente poder dibujarlos.
"""
def get_keypoints(img,block_size,level):
    dx = maskDerivKernels(img,1,0)
    dy = maskDerivKernels(img,0,1)

    dx = gaussian2D(dx,4.5)
    dy = gaussian2D(dy,4.5)
    keypoints = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] > 0):
                ox = dx[i,j]
                oy = dy[i,j]
                cos,sen = ox/(math.sqrt(ox**2 + oy**2)),oy/(math.sqrt(ox**2 + oy**2))
                ori = math.atan2(sen,cos)*180/math.pi
                keypoints.append(cv.KeyPoint(j*(2**level),
                                              i*(2**level),
                                              _size = block_size* 2**(level-1)*2,
                                              _angle = ori))
    return keypoints

"""
Función que calcula puntos harris segun valores propios y los reduce al pasarlos
por un umbral y por la supresión de no máximos
"""
def calculate_harris(src,level,block_size = 3,ksize = 3,threshold = 10):
    # Obtenemos los valores y vectores propios(l1,l2, eiv11,eiv12, eiv21,eiv22)
    e_v = cv.cornerEigenValsAndVecs(src,blockSize = block_size,ksize = ksize)
    
    # Calculamos matriz con valor segun criterio harris
    first_m = np.asarray([[ criterio_harris(e_v[i,j,0],e_v[i,j,1]) 
                 for j in range(src.shape[1])] 
                 for i in range(src.shape[0])])
    
    # Nos quedamos con los valores mayores que un umbral
    threshold_m = np.asarray([[ first_m[i,j] if first_m[i,j] > threshold else 0
                     for j in range(first_m.shape[1])]
                     for i in range(first_m.shape[0])])
    

    # Suprimimos los no máximos en una distancia X por X del vecindario
    sup_no_max_m  = supresionNoMax(threshold_m,5)
   
    # Return keypoints
    return get_keypoints(sup_no_max_m,block_size,level)


def ej1():
    
    # Creamos la una pirámide Gaussiana para cada imagen con el número de niveles (3)   
    p1 = gaussianPyramid(y1g,iters = 3)
    p2 = gaussianPyramid(y2g,iters = 3)

    all_keypoints = np.copy(y1g).astype(np.uint8)
    total_kp = 0
    
    # Pintamos puntos encontrados en las 2 imagenes de ejemplo con 3 escalas cada una más
    # una última con todos los puntos de todas los niveles
    
    print("*** IMAGEN YOSEMIR 1 ***")
    for i in range(len(p1)):
        img = p1[i].astype((np.float32))
        kp = calculate_harris(img,i,block_size = 7,ksize = 5,threshold = 30)
        msg = "Nivel " + str(i+1) + ":  " + str(len(kp)) + " keypoints."
        print(msg)
        total_kp += len(kp)
        
        # Dibuja los puntos circunferencia y radio
        copy = np.copy(y1g).astype(np.uint8)
        copy = cv.drawKeypoints(copy,kp,np.array([]),flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.float64)
        all_keypoints = cv.drawKeypoints(all_keypoints,kp,np.array([]),flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        pintaImagen(copy,msg)
    
    pintaImagen(all_keypoints.astype(np.float64),"Keypoints totales:  "+str(total_kp))
    print("Total keypoints encontrados: " + str(total_kp))
    
    
    print("*** IMAGEN YOSEMIR 2 ***")
    all_keypoints = np.copy(y2g).astype(np.uint8)
    total_kp = 0

    for i in range(len(p2)):
        img = p2[i].astype((np.float32))
        kp = calculate_harris(img,i,block_size = 7,ksize = 5,threshold = 30)
        msg = "Nivel " + str(i+1) + ":  " + str(len(kp)) + " keypoints."
        print(msg)
        total_kp += len(kp)
        
        # Draw circles
        copy = np.copy(y2g).astype(np.uint8)
        copy = cv.drawKeypoints(copy,kp,np.array([]),flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).astype(np.float64)
        all_keypoints = cv.drawKeypoints(all_keypoints,kp,np.array([]),flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        pintaImagen(copy,msg)
    
    pintaImagen(all_keypoints.astype(np.float64),"Keypoints totales:  "+str(total_kp))
    print("Total keypoints encontrados: " + str(total_kp))
    
    
    
    #######################################################################################
    """
    # Refinamiento subpixel  
    
    # Elegimos 3 keyPoints aleatorios
    rng.seed(12345)
    corners = []
    for i in range(3):
        r = rng.randint(0,len(kp))
        corners.append(kp[r])

    # Calculamos su refinamiento
    winSize = (11,11)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 50, 0.001)
    corners_1 = cv.cornerSubPix(y2g, corners[0], winSize, zeroZone, criteria)
    
    # Extraemos la región con distancia 11 al punto original
    x,y = corners[0]
    imgP1 = y2g[(x-11):(x+11),(y-11):(y+11)]

    #Interpolamos imágenes y dibujamos los puntos
    
    width = int(imgP1.shape[1] * 10 / 100)
    height = int(imgP1.shape[0] * 10 / 100)
    dsize = (width,height)
    
    interP1 = cv.resize(imgP1,dsize)
    interP2 = cv.resize(imgP2,dsize)
    interP3 = cv.resize(imgP3,dsize)
    
    # Dibujar puntos
    radius = 3
    
    # Punto original
    cv.circle(interP1, (corners[i,0,0], corners[i,0,1]), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)
    # Punto refinado
    cv.circle(interP1, (corners_1[i,0,0], corners[i,0,1]), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)

    # Mostramos resultados
    pintaImagen(interP1,"Refinamiento punto 1")
    pintaImagen(interP2,"Refinamiento punto 2")
    pintaImagen(interP3,"Refinamiento punto 3")
    """

#########################################################################
#######################    EJERCICIO 2     ##############################
#########################################################################

"""
Función que dadas dos imágenes encuentra keypoints y descriptores y los empareja
según el criterio de fuerza brute buscando el descriptor, buscando el descriptor más 
cercano probando uno por uno.
"""
def matchBruteForce(im1,im2):
    # Creamos AKAZE
    akaze = cv.AKAZE_create()
    # Obtenemos los keypoints y los descriptores
    kp1,d1 = akaze.detectAndCompute(im1,None)
    kp2,d2 = akaze.detectAndCompute(im2,None)
    # Creamos objeto BF MAtcher
    bfmatcher = cv.BFMatcher.create(crossCheck = True)
    # Emparejamiento de fuerza bruta
    matches = bfmatcher.match(d1,d2)
    
    return kp1,kp2,d1,d2,matches


"""
Función que dadas dos imágenes encuentra keypoints y descriptores y los empareja
según el criterio de Lowe Average 2NN tomando los dos vecinos más cercano y
quedandose con un número reducido de emparejamientos válidos.
Dado un par de matches m, n, si la distancia de m es menor que la distancia de
 n por el ratio, entonces tomará m como válido.
"""
def matchLoweAvg2NN(im1,im2,ratio = 0.7):
    # Creamos AKAZE
    akaze = cv.AKAZE_create()
    # Obtenemos los keypoints y los descriptores
    kp1,d1 = akaze.detectAndCompute(im1,None)
    kp2,d2 = akaze.detectAndCompute(im2,None)
    # Creamos objeto BF MAtcher
    bfmatcher = cv.BFMatcher.create()
    # Usamos emparejamiento 2NN
    matches = bfmatcher.knnMatch(d1,d2,k=2)
    # Nos quedamos con los emparejamiento no ambiguos
    valid = []
    for m,n in matches:
        if m.distance < n.distance*ratio:
            valid.append([m])
    
    return kp1,kp2,d1,d2,valid
    


def ej2():
    # Fijamos número de emparejamientos a 100
    n = 100
    im1 = y1g.astype(np.uint8)
    im2 = y2g.astype(np.uint8)
    
    # Brute Force matches
    k1,k2,d1,d2,matches = matchBruteForce(im1,im2)
    # Tomamos n matches
    sample = random.sample(matches,n)
    img = cv.drawMatches(im1,k1,im2,k2,sample,None,flags = 2)
    pintaImagen(img.astype(np.float64),"Brute Force + Crosscheck")

    # LoweAvg2NN matches
    k1,k2,d1,d2,matches = matchLoweAvg2NN(im1,im2)
    # Tomamos n matches
    sample = random.sample(matches,n)
    # Dibujamos emparejamientos
    img = cv.drawMatchesKnn(im1,k1,im2,k2,sample,None,flags = 2)
    pintaImagen(img.astype(np.float64),"Lowe-Average 2NN")


#########################################################################
#######################    EJERCICIO 3     ##############################
#########################################################################
"""
Función que elimina las partes sobrantes negras de la imagen fusionada
"""
def remove_extra(img):
    indexR =[]
    indexC = []
    for i in range(img.shape[0]):
        if np.count_nonzero(img[i]) == 0:
            indexR.append(i)
    for i in range(img.shape[1]):
        if np.count_nonzero(img[:,i]) == 0:
            indexC.append(i)
    img = np.delete(img,indexR,axis = 0)
    img = np.delete(img,indexC,axis = 1)
    return img

"""
Crea canvas suficientemente grande para que quepan las imágenes que queremos
juntar
"""
def getCanvas(imgs):
    return np.zeros((sum([img.shape[0] for img in imgs])*2,
                     sum([img.shape[1] for img in imgs])*2)).astype(np.uint8)

"""
Función para crear la homografía identidad
"""
def identity_h(img,canvas):
    tx = canvas.shape[1]/2 - img.shape[1]/2
    ty = canvas.shape[0]/2 - img.shape[0]/2
    id = np.array([[1,0,tx],[0,1,ty],[0,0,1]],dtype = np.float32)
    return id

"""
Función que devuelve la homografía de dos imágenes usando algoritmo RANSAC
"""
def homography(img1,img2):
    # Obtenemos puntos descriptores y emparejamientos
    k1,k2,d1,d2,matches = matchLoweAvg2NN(img1,img2)
    # Ordenamos los puntos de emparejamiento
    orig = np.float32([k1[p[0].queryIdx].pt for p in matches]).reshape(-1,1,2)
    dest = np.float32([k2[p[0].trainIdx].pt for p in matches]).reshape(-1,1,2)
    # Obtenemos la homografía usando RANSAC
    h = cv.findHomography(orig,dest,cv.RANSAC,1)[0]

    return h

"""
Función que dadas 3 imágenes devuelve una única imagen de la unión de estas
"""
def mosaic(img1,img2,img3):
    canvas = getCanvas([img1,img2,img3])
    h = homography(img2,img1)
    id = identity_h(img1,canvas)
    # Introduce img1 en el canvas
    canvas = cv.warpPerspective(img1,id,(canvas.shape[1],canvas.shape[0]),
                                dst = canvas,borderMode = cv.BORDER_TRANSPARENT)
    comp = np.dot(id,h)
    # Introduce img2 en el canvas
    canvas = cv.warpPerspective(img2,comp,(canvas.shape[1],canvas.shape[0]),
                                dst = canvas,borderMode = cv.BORDER_TRANSPARENT)
    
    h2 = homography(img3,img2)
    comp = np.dot(comp,h2)
    # Introduce img3 en el canvas
    canvas = cv.warpPerspective(img3,comp,(canvas.shape[1],canvas.shape[0]),
                                dst = canvas,borderMode = cv.BORDER_TRANSPARENT)
    return canvas


def ej3():
  
    im_1 = leeImagen("images/mosaico002.jpg",0)
    im_2 = leeImagen("images/mosaico003.jpg",0)
    im_3 = leeImagen("images/mosaico004.jpg",0)
    
    c = mosaic(im_1,im_2,im_3)
    c = remove_extra(c)

    pintaImagen(c,"Composición escala de grises de 3 imágenes")


#########################################################################
#######################    EJERCICIO 4     ##############################
#########################################################################

def n_mosaic(imgs):
    # Creamos canvas y calculamos posición de la imagen central
    half = int(len(imgs)/2)
    canvas = getCanvas(imgs)
 
    # Crea homografía identidad y mete img1 en el canvas
    id = identity_h(imgs[half],canvas)
    
    canvas = cv.warpPerspective(imgs[half],id,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv.BORDER_TRANSPARENT)
    
    # Creamos vector de homografias
    homs = [None]*len(imgs)
    homs[half] = id
 
    # Parte izquierda de las imágenes 5,4,3,2,1,0
    for i in range(half)[::-1]:
        h_i = homography(imgs[i],imgs[i+1])
        
        h_i = np.dot(homs[i+1],h_i)
        homs[i] = h_i
        canvas = cv.warpPerspective(imgs[i],h_i,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv.BORDER_TRANSPARENT)
        
    homs[half]=homs[0]

    # Parte derecha de las imágenes 6,7,8,9
    for i in range(half,len(imgs)):
       h_i = homography(imgs[i],imgs[i-1])
       h_i = np.dot(homs[i-1],h_i)

       homs[i] = h_i
       canvas = cv.warpPerspective(imgs[i],h_i,(canvas.shape[1],canvas.shape[0]),dst = canvas,borderMode = cv.BORDER_TRANSPARENT)

    return canvas


def ej4():
    
    im_1 = leeImagen("images/mosaico002.jpg",0)
    im_2 = leeImagen("images/mosaico003.jpg",0)
    im_3 = leeImagen("images/mosaico004.jpg",0)
    im_4 = leeImagen("images/mosaico005.jpg",0)
    im_5 = leeImagen("images/mosaico006.jpg",0)
    im_6 = leeImagen("images/mosaico007.jpg",0)
    im_7 = leeImagen("images/mosaico008.jpg",0)
    im_8 = leeImagen("images/mosaico009.jpg",0)
    im_9 = leeImagen("images/mosaico010.jpg",0)
    im_10 = leeImagen("images/mosaico011.jpg",0)

    
    mosaico_final = n_mosaic([im_1,im_2,im_3,im_4,im_5,im_6,im_7,im_8,im_9,im_10])
    mosaico_final = remove_extra(mosaico_final)
    pintaImagen(mosaico_final,"Mosaico final escala de grises")




#########################################################################
###################### EJECUCIÓN EJERCICIOS #############################
#########################################################################
print("""
      ************* EJERCICIO 1 ***************
      """)
ej1()

input("(Pulsa cualquier tecla para continuar...)")
print("""
      ************* EJERCICIO 2 ***************
      """)
ej2()
input("(Pulsa cualquier tecla para continuar...)")
print("""
      ************* EJERCICIO 3 ***************
      """)
ej3()
input("(Pulsa cualquier tecla para continuar...)")
print("""
      ************* EJERCICIO 4 ***************
      """)
ej4()