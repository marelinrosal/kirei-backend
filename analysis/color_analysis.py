import cv2
import numpy as np

def analyze_color(image_bytes: bytes) -> dict:
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "No se pudo procesar la imagen"}

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_color = image_lab.mean(axis=(0, 1))
    L, A, B = avg_color[0], avg_color[1], avg_color[2]

    if L > 180:
        tono = "muy claro"
    elif L > 150:
        tono = "claro"
    elif L > 120:
        tono = "medio"
    elif L > 90:
        tono = "oscuro"
    else:
        tono = "profundo"

    if B > A + 5:
        subtono = "cálido"
    elif A > B + 5:
        subtono = "frío"
    else:
        subtono = "neutro"

    temporada = determinar_temporada(tono, subtono)

    return {
        "tono":      tono,
        "subtono":   subtono,
        "temporada": temporada
    }


def determinar_temporada(tono: str, subtono: str) -> str:
    if subtono == "cálido":
        if tono in ["claro", "muy claro"]:
            return "Primavera"
        else:
            return "Otoño"
    elif subtono == "frío":
        if tono in ["claro", "muy claro"]:
            return "Verano"
        else:
            return "Invierno"
    else:
        if tono in ["claro", "muy claro", "medio"]:
            return "Verano"
        else:
            return "Otoño"