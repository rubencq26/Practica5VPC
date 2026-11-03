import numpy as np
import cv2
from enum import Enum


class OpcionRelleno(Enum):
    REPLICATE = 0
    REFLECT = 1
    ZEROS = 2


def filtro_maximos(I, V, OpcionRelleno=OpcionRelleno.ZEROS):
    if I.dtype != np.uint8 and I.dtype != np.float32:
         raise ValueError("La imagen debe ser uint8 o float32")
    if I.ndim != 2:
        raise ValueError("Dimension incorrecta la Imagen debe ser en escala de grises")
    if I.dtype == np.uint8:
        Ic = np.float32(I)

    V = np.asarray(V)
    Ic = I.astype(np.float32)

    borderHeight = V.shape[0] // 2
    borderWidth = V.shape[1] // 2

    rellenos = [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT]

    pad = cv2.copyMakeBorder(Ic, borderHeight, borderHeight, borderWidth, borderWidth, borderType=rellenos[OpcionRelleno.value])

    alto, ancho = Ic.shape
    mascara = (V == 1)
    Io = np.zeros((alto, ancho), np.float32)

    for i in range(alto):
        for j in range(ancho):
            ventana = pad[i:i + V.shape[0], j:j + V.shape[1]]
            Io[i, j] = np.max(ventana[mascara])

    return Io
    

    
def filtro_minimos(I, V, OpcionRelleno=OpcionRelleno.ZEROS):
    if I.dtype != np.uint8 and I.dtype != np.float32:
         raise ValueError("La imagen debe ser uint8 o float32")
    if I.ndim != 2:
        raise ValueError("Dimension incorrecta la Imagen debe ser en escala de grises")
    if I.dtype == np.uint8:
        Ic = np.float32(I)

    V = np.asarray(V)
    Ic = I.astype(np.float32)

    borderHeight = V.shape[0] // 2
    borderWidth = V.shape[1] // 2

    rellenos = [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT]

    pad = cv2.copyMakeBorder(Ic, borderHeight, borderHeight, borderWidth, borderWidth, borderType=rellenos[OpcionRelleno.value])

    alto, ancho = Ic.shape
    mascara = (V == 1)
    Io = np.zeros((alto, ancho), np.float32)

    for i in range(alto):
        for j in range(ancho):
            ventana = pad[i:i + V.shape[0], j:j + V.shape[1]]
            Io[i, j] = np.min(ventana[mascara])

    return Io


def filtro_mediana(I, V, OpcionRelleno=OpcionRelleno.ZEROS):
    if I.dtype != np.uint8 and I.dtype != np.float32:
         raise ValueError("La imagen debe ser uint8 o float32")
    if I.ndim != 2:
        raise ValueError("Dimension incorrecta la Imagen debe ser en escala de grises")
    if I.dtype == np.uint8:
        Ic = np.float32(I)

    V = np.asarray(V)
    Ic = I.astype(np.float32)

    borderHeight = V.shape[0] // 2
    borderWidth = V.shape[1] // 2

    rellenos = [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT]

    pad = cv2.copyMakeBorder(Ic, borderHeight, borderHeight, borderWidth, borderWidth, borderType=rellenos[OpcionRelleno.value])

    alto, ancho = Ic.shape
    mascara = (V == 1)
    Io = np.zeros((alto, ancho), np.float32)

    for i in range(alto):
        for j in range(ancho):
            ventana = pad[i:i + V.shape[0], j:j + V.shape[1]]
            Io[i, j] = np.median(ventana[mascara]).astype(np.uint8)

    return Io
    


    

    
