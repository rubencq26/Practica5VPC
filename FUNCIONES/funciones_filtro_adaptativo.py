import numpy as np
import cv2
from funcion_stdfilt import stdfilt

EPS = np.finfo(np.float64).eps

def _comprobar(I, V):
    if I.ndim != 2:
        raise ValueError("I debe ser una imagen 2D")
    if V.ndim != 2 or V.shape[0] % 2 == 0 or V.shape[1] % 2 == 0:
        raise ValueError("V debe ser 2D y de dimensiones impares")
    if not np.any(V):
        raise ValueError("V debe tener al menos un 1")

def filtro_adaptativo_v1(I: np.ndarray, V: np.ndarray, VarRuido: float, opcionrelleno: str = 'replicate') -> np.ndarray:
    _comprobar(I, V)
    I = I.astype(np.float64)
    mascara = V.astype(bool)
    alto, ancho = I.shape
    nf, nc = V.shape

    relleno = {
        'replicate': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT,
        'constant': cv2.BORDER_CONSTANT

    }

    pad = cv2.copyMakeBorder(I, nf // 2, nf // 2, nc // 2, nc // 2, relleno[opcionrelleno])
    Io = np.zeros((alto, ancho), dtype=np.float64)

    for i in range(alto):
        for j in range(ancho):
            ventana = pad[i:i + nf, j:j + nc]
            valores = ventana[mascara]

            media = np.mean(valores)
            desv = np.std(valores)
            if desv == 0:
                desv = EPS
            var_local = desv ** 2

            factor = np.clip(VarRuido / var_local, 0.0, 1.0)
            Io[i, j] = I[i, j] - factor * (I[i, j] - media)
    return Io


def filtro_adaptativo_v2(I: np.ndarray, V: np.ndarray, VarRuido: float, opcionrelleno: str = 'replicate') -> np.ndarray:
    _comprobar(I, V)
    I = I.astype(np.float64)
    Vb = (V > 0).astype(np.float64)
 
    relleno = {
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT,
            'constant': cv2.BORDER_CONSTANT
    
        }

    media = cv2.filter2D(I, -1, Vb / Vb.sum(), borderType=relleno[opcionrelleno])
 
    
    desv = stdfilt(I, Vb, opcionrelleno)
    desv[desv == 0] = EPS                                
    var_local = desv ** 2
 
    factor = np.clip(VarRuido / var_local, 0.0, 1.0)
    return I - factor * (I - media)

