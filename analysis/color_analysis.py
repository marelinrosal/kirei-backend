import cv2
import numpy as np

# ── Rangos Fitzpatrick por luminosidad L* ─────────────────────────────────────
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


# ── Análisis de imagen ────────────────────────────────────────────────────────
def _analizar_imagen(image_bytes: bytes) -> dict:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False}
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_mean = float(np.mean(L)) / 255.0 * 100.0
    a_mean = float(np.mean(A)) - 128.0   # eje rojo-verde
    b_mean = float(np.mean(B)) - 128.0   # eje amarillo-azul
    return {
        "ok":           True,
        "fototipo_img": _L_to_fototipo(L_mean),
        "L_norm":       round(L_mean, 2),
        "a_star":       round(a_mean, 2),
        "b_star":       round(b_mean, 2),
    }


# ── Análisis de cuestionario ──────────────────────────────────────────────────
def _analizar_cuestionario(resp: dict) -> dict:

    # ── 1. Fototipo ───────────────────────────────────────────────────────────
    try:
        skin_val = int(resp.get("skin", 3))
    except (ValueError, TypeError):
        skin_val = 3

    try:
        sun_val = int(resp.get("sun", 2))
    except (ValueError, TypeError):
        sun_val = 2

    freckles     = resp.get("freckles", "no")
    freckles_adj = -1 if freckles == "yes" else 0

    # Antebrazo: indica claridad real de piel sin bronceado
    forearm_map = {
        "rosy":   0,    # muy clara / fototipo 1-2
        "beige":  1,    # clara-media / fototipo 2-3
        "peachy": 2,    # media / fototipo 3-4
        "tan":    3,    # media-oscura / fototipo 4-5
    }
    forearm_val = forearm_map.get(resp.get("forearm", ""), -1)

    # Reflejos de cabello: indicador fino de claridad/oscuridad
    shine_map = {
        "golden": -0.3,   # tiende a piel más clara/cálida
        "red":    -0.1,
        "none":    0.0,
        "ashy":    0.2,   # tiende a piel más fría/oscura
    }
    shine_adj = shine_map.get(resp.get("hair_shine", ""), 0.0)

    # Score base fototipo (1=muy clara, 6=muy oscura)
    fototipo_q_raw = skin_val * 0.50 + sun_val * 0.30
    if forearm_val >= 0:
        fototipo_q_raw += forearm_val * 0.20
    fototipo_q_raw += shine_adj
    fototipo_q = round(fototipo_q_raw) + freckles_adj
    fototipo_q = max(1, min(6, fototipo_q))

    # ── 2. Temporada colorimétrica ────────────────────────────────────────────
    # Se determina combinando: piel (skin+forearm), ojos, cabello, reflejos, pecas
    # Escala interna: warmth (-3 frío … +3 cálido), depth (0 claro … 6 oscuro)

    warmth = 0.0
    depth  = float(skin_val)  # base: qué tan oscura es la piel (1=claro,6=oscuro)

    # Ojos
    eye_warmth = {
        "blue":   -2.0, "gray": -1.5, "green": -0.5,
        "hazel":   0.5, "brown": 1.0, "dbrown": 1.5,
    }
    eye_depth = {
        "blue": 0, "gray": 0, "green": 1,
        "hazel": 2, "brown": 3, "dbrown": 4,
    }
    eye = resp.get("eye", "brown")
    warmth += eye_warmth.get(eye, 0.0)
    depth  += eye_depth.get(eye, 2)

    # Cabello
    hair_warmth = {
        "blonde":  -1.0, "red":    1.5, "lbrown": 0.5,
        "brown":    0.5, "dbrown": 0.0, "black": -0.5,
    }
    hair_depth = {
        "blonde": 0, "red": 2, "lbrown": 2,
        "brown":  3, "dbrown": 4, "black": 5,
    }
    hair = resp.get("hair", "brown")
    warmth += hair_warmth.get(hair, 0.0)
    depth  += hair_depth.get(hair, 2)

    # Reflejos (señal fina)
    shine_warmth = {
        "golden":  1.0, "red":  0.5,
        "ashy":   -1.0, "none": 0.0,
    }
    warmth += shine_warmth.get(resp.get("hair_shine", ""), 0.0)

    # Antebrazo (señal fina de warmth)
    forearm_warmth = {
        "rosy":  -1.0, "beige": 0.0,
        "peachy": 1.0, "tan":   0.5,
    }
    warmth += forearm_warmth.get(resp.get("forearm", ""), 0.0)

    # Pecas: asociadas a primavera/otoño (cálido)
    if freckles == "yes":
        warmth += 1.0

    # Contraste tonal piel × cabello
    hair_darkness_scale = {
        "blonde": 1, "red": 3, "lbrown": 3,
        "brown":  4, "dbrown": 5, "black": 6,
    }
    contraste = abs(hair_darkness_scale.get(hair, 3) - skin_val)

    # ── Clasificación en 12 subtemporadas ────────────────────────────────────
    # Lógica: primero eje frío/cálido (warmth), luego eje claro/oscuro (depth)
    # y contraste para diferenciar "brillante" vs "suave"

    temporada, subtemporada = _clasificar_temporada(warmth, depth, contraste)

    # ── Confianza ─────────────────────────────────────────────────────────────
    campos      = ["skin", "eye", "hair", "hair_shine", "forearm", "sun", "freckles"]
    respondidos = sum(1 for c in campos if resp.get(c, ""))
    confianza_q = round(0.45 + (respondidos / len(campos)) * 0.50, 3)

    return {
        "fototipo_q":   fototipo_q,
        "temporada":    temporada,
        "subtemporada": subtemporada,
        "confianza_q":  confianza_q,
    }


def _clasificar_temporada(warmth: float, depth: float, contraste: int) -> tuple[str, str]:
    """
    warmth  < -1.5  → frío  |  > 1.5 → cálido  |  medio → neutro
    depth   < 8     → claro |  8-13  → medio    |  > 13  → oscuro
    contraste ≥ 4   → alto  |  < 4   → bajo
    """
    es_calido = warmth > 1.5
    es_frio   = warmth < -1.5
    es_claro  = depth < 8
    es_oscuro = depth > 13
    alto_contraste = contraste >= 4

    if es_frio:
        if es_oscuro or alto_contraste:
            return "invierno", "invierno_oscuro"
        elif es_claro:
            return "verano", "verano_frio"
        else:
            return "invierno", "invierno_frio"

    elif es_calido:
        if es_claro:
            if alto_contraste:
                return "primavera", "primavera_brillante"
            else:
                return "primavera", "primavera_calida"
        elif es_oscuro:
            return "otono", "otono_oscuro"
        else:
            if alto_contraste:
                return "primavera", "primavera_brillante"
            else:
                return "otono", "otono_calido"

    else:  # neutro
        if es_claro:
            return "verano", "verano_claro"
        elif es_oscuro:
            return "otono", "otono_suave"
        else:
            if alto_contraste:
                return "invierno", "invierno_brillante"
            else:
                return "verano", "verano_suave"


# ── Función principal ─────────────────────────────────────────────────────────
def analyze_color(image_bytes: bytes, cuestionario: dict = None) -> dict:
    if cuestionario is None:
        cuestionario = {}

    img_result = _analizar_imagen(image_bytes)
    q_result   = _analizar_cuestionario(cuestionario)

    fototipo_q   = q_result["fototipo_q"]
    temporada    = q_result["temporada"]
    subtemporada = q_result["subtemporada"]
    confianza    = q_result["confianza_q"]

    if img_result["ok"]:
        # Fototipo: imagen (35%) + cuestionario (65%)
        fototipo_raw = img_result["fototipo_img"] * 0.35 + fototipo_q * 0.65
        fototipo     = max(1, min(6, round(fototipo_raw)))

        # Refinar temporada con datos de imagen si hay señal clara
        b = img_result["b_star"]
        a = img_result["a_star"]
        L = img_result["L_norm"]

        # Si L es muy alto (piel muy clara) y la temporada es oscura → corregir
        if L > 70 and temporada in ("otono", "invierno") and subtemporada in ("otono_oscuro", "invierno_oscuro"):
            subtemporada = subtemporada.replace("oscuro", "suave").replace("_oscuro", "_frio")
            subtemporada = {
                "otono_suave":   "otono_suave",
                "invierno_frio": "invierno_frio",
            }.get(subtemporada, subtemporada)

        confianza = min(0.97, confianza + 0.08)
    else:
        fototipo = fototipo_q

    return {
        "fototipo":    fototipo,
        "temporada":   temporada,
        "subtemporada": subtemporada,
        "confianza":   round(confianza, 2),
    }