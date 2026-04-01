import cv2
import numpy as np

FITZPATRICK_L_RANGES = [
    (1, 75, 101),
    (2, 63, 75),
    (3, 50, 63),
    (4, 38, 50),
    (5, 25, 38),
    (6,  0, 25),
]

def _L_to_fototipo(L_norm: float) -> int:
    for fototipo, lo, hi in FITZPATRICK_L_RANGES:
        if lo <= L_norm < hi:
            return fototipo
    return 6 if L_norm < 25 else 1


def _analizar_imagen(image_bytes: bytes) -> dict:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False}
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_mean = float(np.mean(L)) / 255.0 * 100.0
    b_mean = float(np.mean(B)) - 128.0
    return {
        "ok":           True,
        "fototipo_img": _L_to_fototipo(L_mean),
        "L_norm":       round(L_mean, 2),
        "b_star":       round(b_mean, 2),
    }


def _analizar_cuestionario(resp: dict) -> dict:
    # ── Fototipo ──────────────────────────────────────────────────────────────
    try:
        skin_val = int(resp.get("skin", 3))
    except (ValueError, TypeError):
        skin_val = 3

    freckles     = resp.get("freckles", "no")
    freckles_adj = -1 if freckles == "yes" else 0

    # skin ahora tiene más peso (0.85) porque ya no hay sun para complementarlo
    fototipo_q = round(skin_val * 0.85) + freckles_adj
    fototipo_q = max(1, min(6, fototipo_q))

    # ── Subtono ───────────────────────────────────────────────────────────────
    score = 0.0

    vein = resp.get("vein", "neutral")
    if vein == "cold":   score -= 2.5
    elif vein == "warm": score += 2.5

    eye_map = {
        "blue":   -1.5,
        "gray":   -1.0,
        "green":  -0.5,
        "hazel":   0.5,
        "brown":   1.0,
        "dbrown":  1.5,
    }
    score += eye_map.get(resp.get("eye", "brown"), 0.0)

    hair_map = {
        "blonde": -1.0,
        "red":    -0.5,
        "lbrown":  0.5,
        "brown":   0.5,
        "dbrown":  1.0,
        "black":   1.0,
    }
    score += hair_map.get(resp.get("hair", "brown"), 0.0)

    base_map = {
        "rosada": -1.0,
        "beige":   0.0,
        "oliva":   1.0,
        "cafe":    1.5,
    }
    score += base_map.get(resp.get("base", "beige"), 0.0)

    if score <= -1.5:
        subtono = "frio"
    elif score >= 1.5:
        subtono = "calido"
    else:
        subtono = "neutro"

    # ── Confianza ─────────────────────────────────────────────────────────────
    # tipo_piel reemplaza a sun en el conteo de campos respondidos
    campos      = ["skin", "eye", "hair", "vein", "tipo_piel", "freckles", "base"]
    respondidos = sum(1 for c in campos if resp.get(c, ""))
    confianza_q = round(0.50 + (respondidos / len(campos)) * 0.45, 3)

    return {
        "fototipo_q":  fototipo_q,
        "subtono":     subtono,
        "confianza_q": confianza_q,
    }


def analyze_color(image_bytes: bytes, cuestionario: dict = None) -> dict:
    if cuestionario is None:
        cuestionario = {}

    img_result = _analizar_imagen(image_bytes)
    q_result   = _analizar_cuestionario(cuestionario)

    fototipo_q = q_result["fototipo_q"]
    subtono    = q_result["subtono"]
    confianza  = q_result["confianza_q"]

    if img_result["ok"]:
        # skin (0.85) + imagen (0.35) para el fototipo final
        fototipo_raw = img_result["fototipo_img"] * 0.35 + fototipo_q * 0.65
        fototipo     = max(1, min(6, round(fototipo_raw)))

        # Si el cuestionario dejó subtono neutro, la imagen puede desempatarlo
        if subtono == "neutro":
            b = img_result["b_star"]
            if b > 8:    subtono = "calido"
            elif b < -4: subtono = "frio"

        confianza = min(0.97, confianza + 0.08)
    else:
        fototipo = fototipo_q

    return {
        "fototipo":  fototipo,
        "subtono":   subtono,
        "confianza": round(confianza, 2),
    }