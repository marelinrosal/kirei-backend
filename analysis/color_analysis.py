import cv2
import numpy as np


# ── Rangos de L* (espacio LAB, 0-100) para cada fototipo ─────────────────────
# OpenCV devuelve L en [0, 255], se normaliza a [0, 100]
FITZPATRICK_L_RANGES = [
    (1, 75, 101),   # (fototipo, L_min, L_max)
    (2, 63, 75),
    (3, 50, 63),
    (4, 38, 50),
    (5, 25, 38),
    (6,  0, 25),
]


def _L_to_fototipo(L_norm: float) -> int:
    """Convierte L* normalizado (0-100) al fototipo base."""
    for fototipo, lo, hi in FITZPATRICK_L_RANGES:
        if lo <= L_norm < hi:
            return fototipo
    return 6 if L_norm < 25 else 1


# ── Análisis de imagen ────────────────────────────────────────────────────────

def _analizar_imagen(image_bytes: bytes) -> dict:
    """
    Devuelve:
        fototipo_img  : int  (1-6)
        L_norm        : float (0-100)
        b_star        : float  — positivo = cálido, negativo = frío
        ok            : bool
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False}

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L_mean = float(np.mean(L)) / 255.0 * 100.0   # normalizar a [0, 100]
    b_mean = float(np.mean(B)) - 128.0            # b* centrado en 0

    return {
        "ok":           True,
        "fototipo_img": _L_to_fototipo(L_mean),
        "L_norm":       round(L_mean, 2),
        "b_star":       round(b_mean, 2),
    }


# ── Análisis de cuestionario ──────────────────────────────────────────────────

def _analizar_cuestionario(resp: dict) -> dict:
    """
    Campos esperados:
        skin     : str  "1"-"6"
        eye      : str  "blue"|"green"|"hazel"|"brown"|"dbrown"|"gray"
        hair     : str  "blonde"|"lbrown"|"brown"|"dbrown"|"black"|"red"
        vein     : str  "cold"|"warm"|"neutral"
        sun      : str  "1"-"4"
        freckles : str  "yes"|"few"|"no"
        base     : str  "rosada"|"beige"|"oliva"|"cafe"

    Devuelve:
        fototipo_q : int   (1-6) desde cuestionario
        subtono    : str   "frio"|"calido"|"neutro"
        confianza_q: float (0-1) según campos respondidos
    """

    # ── Fototipo desde cuestionario ──────────────────────────────────────────
    try:
        skin_val = int(resp.get("skin", 3))
    except (ValueError, TypeError):
        skin_val = 3

    try:
        sun_val = int(resp.get("sun", 2))
    except (ValueError, TypeError):
        sun_val = 2

    freckles = resp.get("freckles", "no")
    freckles_adj = -1 if freckles == "yes" else 0  # pecas → más claro (fototipo menor)

    fototipo_q = round(skin_val * 0.60 + sun_val * 0.40) + freckles_adj
    fototipo_q = max(1, min(6, fototipo_q))

    # ── Subtono por puntuación ───────────────────────────────────────────────
    score = 0.0

    vein = resp.get("vein", "neutral")
    if vein == "cold":    score -= 2.5
    elif vein == "warm":  score += 2.5

    eye = resp.get("eye", "brown")
    eye_map = {
        "blue":   -1.5,
        "gray":   -1.0,
        "green":  -0.5,
        "hazel":  +0.5,
        "brown":  +1.0,
        "dbrown": +1.5,
    }
    score += eye_map.get(eye, 0.0)

    hair = resp.get("hair", "brown")
    hair_map = {
        "blonde": -1.0,
        "red":    -0.5,
        "lbrown": +0.5,
        "brown":  +0.5,
        "dbrown": +1.0,
        "black":  +1.0,
    }
    score += hair_map.get(hair, 0.0)

    base = resp.get("base", "beige")
    base_map = {
        "rosada": -1.0,
        "beige":   0.0,
        "oliva":  +1.0,
        "cafe":   +1.5,
    }
    score += base_map.get(base, 0.0)

    if score <= -1.5:
        subtono = "frio"
    elif score >= 1.5:
        subtono = "calido"
    else:
        subtono = "neutro"

    # Confianza: cuántos campos válidos recibimos (7 campos = máximo)
    campos = ["skin", "eye", "hair", "vein", "sun", "freckles", "base"]
    respondidos = sum(1 for c in campos if resp.get(c, ""))
    confianza_q = round(0.50 + (respondidos / len(campos)) * 0.45, 3)

    return {
        "fototipo_q":  fototipo_q,
        "subtono":     subtono,
        "confianza_q": confianza_q,
    }


# ── Función principal ─────────────────────────────────────────────────────────

def analyze_color(image_bytes: bytes, cuestionario: dict = None) -> dict:
    """
    Combina análisis de imagen (35%) + cuestionario (65%).

    Si no hay imagen válida, usa solo el cuestionario.
    Devuelve:
        fototipo  : int   (1-6)
        subtono   : str   "frio"|"calido"|"neutro"
        confianza : float (0-1)
    """
    if cuestionario is None:
        cuestionario = {}

    img_result = _analizar_imagen(image_bytes)
    q_result   = _analizar_cuestionario(cuestionario)

    fototipo_q = q_result["fototipo_q"]
    subtono    = q_result["subtono"]
    confianza  = q_result["confianza_q"]

    if img_result["ok"]:
        PESO_IMG = 0.35
        PESO_Q   = 0.65
        fototipo_raw = (img_result["fototipo_img"] * PESO_IMG +
                        fototipo_q               * PESO_Q)
        fototipo = max(1, min(6, round(fototipo_raw)))

        # Refinar subtono con b* si el cuestionario dice neutro
        if subtono == "neutro":
            b = img_result["b_star"]
            if b > 8:
                subtono = "calido"
            elif b < -4:
                subtono = "frio"

        confianza = min(0.97, confianza + 0.08)   # bonus por tener imagen
    else:
        fototipo = fototipo_q

    return {
        "fototipo":  fototipo,
        "subtono":   subtono,
        "confianza": round(confianza, 2),
    }