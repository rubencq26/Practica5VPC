import cv2
import numpy as np

def stdfilt(I, V=None, opcionRelleno='reflect'):

    """
    Calcula la desviación estándar local sobre una vecindad binaria.
    
    Parámetros
    ----------
    I : np.ndarray
        Imagen de entrada 2D, tipo uint8 o float.
    V : np.ndarray
        Matriz de vecindad (de dimensiones impares), compuesta por 0s y 1s.
        Solo los píxeles donde V=1 se usan para calcular la desviación local.
     opcionrelleno : str
        Tipo de padding: 'replicate', 'reflect' o 'zeros' (por defecto 'reflect').
    
    Retorna
    -------
    Io : np.ndarray
        Matriz std resultante tras calcular la std local, tipo float64.
    """

    I = I.astype(np.float64)
    if V is None:
        V = np.ones((3, 3), np.float64)  # vecindad 3x3 por defecto
    else:
        V = (V>0).astype(np.float64)
        
    # Padding según opción
    if opcionRelleno == 'replicate':
        tipoPadding = cv2.BORDER_REPLICATE
    elif opcionRelleno == 'reflect':
        tipoPadding =  cv2.BORDER_REFLECT
    elif opcionRelleno == 'zeros':
        tipoPadding = cv2.BORDER_CONSTANT
    else:
        raise ValueError("Opción de relleno no válida. Use: 'replicate', 'reflect' o 'zeros'.")


    # Kernel promedio (para hacer promedios)
    kp = V / V.sum()

    # Media local y media de los cuadrados
    mean = cv2.filter2D(I, -1, kp, borderType=tipoPadding)
    mean_sq = cv2.filter2D(I*I, -1, kp, borderType=tipoPadding)

    # Varianza local y std
    var = np.maximum(mean_sq - mean**2, 0)
    Istd = np.sqrt(var) 
    return Istd
