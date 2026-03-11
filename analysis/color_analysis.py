import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# analyze_color
# Recibe bytes de la imagen de la mejilla, analiza en espacio LAB y devuelve
# tono, subtono y una clasificación base de temporada.
# ─────────────────────────────────────────────────────────────────────────────

def analyze_color(image_bytes: bytes) -> dict:
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "No se pudo procesar la imagen"}

    # Convertir a LAB (más cercano a percepción humana del color)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_color  = image_lab.mean(axis=(0, 1))
    L, A, B    = float(avg_color[0]), float(avg_color[1]), float(avg_color[2])

    # ── Tono de piel por luminosidad (canal L) ────────────────────────────────
    if L > 195:
        tono = "muy claro"
    elif L > 175:
        tono = "claro"
    elif L > 150:
        tono = "claro medio"
    elif L > 125:
        tono = "medio"
    elif L > 100:
        tono = "medio oscuro"
    elif L > 75:
        tono = "oscuro"
    else:
        tono = "profundo"

    # ── Subtono por canales A y B ─────────────────────────────────────────────
    # Canal A: negativo=verde, positivo=rojo/rosado  → indica frío/cálido
    # Canal B: negativo=azul,  positivo=amarillo     → refuerza cálido
    diferencia = B - A

    if diferencia > 8:
        subtono = "cálido"
        intensidad_calido = min((diferencia - 8) / 15, 1.0)   # 0–1
        intensidad_frio   = 0.0
    elif diferencia < -8:
        subtono = "frío"
        intensidad_frio   = min((-diferencia - 8) / 15, 1.0)
        intensidad_calido = 0.0
    else:
        subtono = "neutro"
        intensidad_calido = 0.0
        intensidad_frio   = 0.0

    # Clasificación base de temporada (se refinará con el cuestionario)
    temporada_base = _temporada_base(tono, subtono)

    return {
        "tono":             tono,
        "subtono":          subtono,
        "temporada_base":   temporada_base,
        "L":                round(L, 2),
        "A":                round(A, 2),
        "B":                round(B, 2),
        "intensidad_calido": round(intensidad_calido, 2),
        "intensidad_frio":   round(intensidad_frio, 2),
    }


def _temporada_base(tono: str, subtono: str) -> str:
    tonos_claros = {"muy claro", "claro", "claro medio"}
    tonos_medios = {"medio"}
    tonos_oscuros = {"medio oscuro", "oscuro", "profundo"}

    if subtono == "cálido":
        if tono in tonos_claros:
            return "Primavera"
        else:
            return "Otoño"
    elif subtono == "frío":
        if tono in tonos_claros:
            return "Verano"
        else:
            return "Invierno"
    else:  # neutro
        if tono in tonos_claros:
            return "Verano"
        elif tono in tonos_medios:
            return "Primavera"
        else:
            return "Otoño"


# ─────────────────────────────────────────────────────────────────────────────
# determinar_subtemporada
#
# Sistema de puntuación ponderada para las 12 subtemporadas.
# Combina el resultado de la imagen con las 10 respuestas del cuestionario.
#
# Las 12 subtemporadas:
#   CÁLIDAS:  Primavera Verdadera, Primavera Luminosa, Primavera Cálida
#             Otoño Verdadero,     Otoño Suave,        Otoño Cálido
#   FRÍAS:    Verano Verdadero,    Verano Suave,       Verano Frío
#             Invierno Verdadero,  Invierno Brillante, Invierno Oscuro
# ─────────────────────────────────────────────────────────────────────────────

SUBTEMPORADAS = [
    # Primaveras (cálidas, claras/luminosas)
    "Primavera Verdadera",
    "Primavera Luminosa",
    "Primavera Cálida",
    # Veranos (fríos, suaves/apagados)
    "Verano Verdadero",
    "Verano Suave",
    "Verano Frío",
    # Otoños (cálidos, profundos/terrosos)
    "Otoño Verdadero",
    "Otoño Suave",
    "Otoño Cálido",
    # Inviernos (fríos, profundos/contrastados)
    "Invierno Verdadero",
    "Invierno Brillante",
    "Invierno Oscuro",
]


def determinar_subtemporada(resultado_imagen: dict, cuestionario: dict) -> str:
    """
    Retorna la subtemporada más probable como string.
    Usa un diccionario de puntuaciones {subtemporada: float}.
    """
    scores = {s: 0.0 for s in SUBTEMPORADAS}

    # ── 1. Señales de la imagen (peso alto) ───────────────────────────────────
    subtono          = resultado_imagen.get("subtono", "neutro")
    tono             = resultado_imagen.get("tono", "medio")
    int_calido       = resultado_imagen.get("intensidad_calido", 0.0)
    int_frio         = resultado_imagen.get("intensidad_frio", 0.0)
    L                = resultado_imagen.get("L", 140.0)

    # Subtono cálido → favorece Primaveras y Otoños
    if subtono == "cálido":
        for s in ["Primavera Verdadera", "Primavera Luminosa", "Primavera Cálida"]:
            scores[s] += 3.0 + int_calido * 2
        for s in ["Otoño Verdadero", "Otoño Suave", "Otoño Cálido"]:
            scores[s] += 2.5 + int_calido * 1.5

    # Subtono frío → favorece Veranos e Inviernos
    elif subtono == "frío":
        for s in ["Verano Verdadero", "Verano Suave", "Verano Frío"]:
            scores[s] += 3.0 + int_frio * 2
        for s in ["Invierno Verdadero", "Invierno Brillante", "Invierno Oscuro"]:
            scores[s] += 2.5 + int_frio * 1.5

    # Subtono neutro → distribuido entre los cuatro grupos
    else:
        for s in SUBTEMPORADAS:
            scores[s] += 1.0

    # Luminosidad alta → Primavera o Verano
    if L > 160:
        for s in ["Primavera Verdadera", "Primavera Luminosa", "Verano Suave", "Verano Verdadero"]:
            scores[s] += 2.0
    elif L > 130:
        for s in ["Primavera Cálida", "Verano Verdadero", "Otoño Suave"]:
            scores[s] += 1.5
    else:
        for s in ["Otoño Verdadero", "Otoño Cálido", "Invierno Oscuro", "Invierno Verdadero"]:
            scores[s] += 2.0

    # ── 2. Color de ojos ──────────────────────────────────────────────────────
    ojos = cuestionario.get("color_ojos", "").lower()

    if any(x in ojos for x in ["azul", "gris azul"]):
        scores["Verano Verdadero"]    += 2.5
        scores["Verano Frío"]         += 2.0
        scores["Invierno Verdadero"]  += 2.0
        scores["Invierno Brillante"]  += 1.5

    elif any(x in ojos for x in ["gris", "gris verde"]):
        scores["Verano Verdadero"]    += 2.0
        scores["Verano Suave"]        += 2.0
        scores["Invierno Verdadero"]  += 1.5

    elif any(x in ojos for x in ["verde", "verde oliva"]):
        scores["Primavera Verdadera"] += 2.0
        scores["Primavera Cálida"]    += 1.5
        scores["Otoño Verdadero"]     += 2.0
        scores["Otoño Suave"]         += 1.5

    elif any(x in ojos for x in ["miel", "avellana", "café claro", "ambar"]):
        scores["Primavera Verdadera"] += 2.5
        scores["Primavera Luminosa"]  += 2.0
        scores["Otoño Cálido"]        += 2.0

    elif any(x in ojos for x in ["café oscuro", "café", "negro"]):
        scores["Otoño Verdadero"]     += 2.0
        scores["Invierno Oscuro"]     += 2.5
        scores["Invierno Verdadero"]  += 2.0

    # ── 3. Color de cabello ───────────────────────────────────────────────────
    cabello = cuestionario.get("color_cabello", "").lower()

    if any(x in cabello for x in ["rubio dorado", "rubio cálido", "rubio miel"]):
        scores["Primavera Verdadera"] += 2.5
        scores["Primavera Luminosa"]  += 2.0
        scores["Primavera Cálida"]    += 1.5

    elif any(x in cabello for x in ["rubio cenizo", "rubio frío", "rubio platino"]):
        scores["Verano Verdadero"]    += 2.5
        scores["Verano Frío"]         += 2.0
        scores["Invierno Brillante"]  += 1.5

    elif any(x in cabello for x in ["castaño claro", "castaño dorado"]):
        scores["Primavera Cálida"]    += 2.0
        scores["Otoño Suave"]         += 2.0
        scores["Verano Suave"]        += 1.5

    elif any(x in cabello for x in ["castaño oscuro", "castaño"]):
        scores["Otoño Verdadero"]     += 2.0
        scores["Verano Verdadero"]    += 1.5
        scores["Invierno Verdadero"]  += 1.5

    elif any(x in cabello for x in ["negro azulado", "negro"]):
        scores["Invierno Verdadero"]  += 2.5
        scores["Invierno Oscuro"]     += 2.5
        scores["Invierno Brillante"]  += 2.0

    elif any(x in cabello for x in ["pelirrojo", "rojo"]):
        scores["Otoño Verdadero"]     += 3.0
        scores["Otoño Cálido"]        += 2.5
        scores["Primavera Cálida"]    += 2.0

    # ── 4. Venas de la muñeca ─────────────────────────────────────────────────
    venas = cuestionario.get("venas_muneca", "").lower()

    if "azul" in venas or "morado" in venas or "violeta" in venas:
        for s in ["Verano Verdadero", "Verano Frío", "Invierno Verdadero",
                  "Invierno Brillante", "Invierno Oscuro"]:
            scores[s] += 2.5

    elif "verde" in venas:
        for s in ["Primavera Verdadera", "Primavera Cálida",
                  "Otoño Verdadero", "Otoño Cálido"]:
            scores[s] += 2.5

    elif "azul y verde" in venas or "mixto" in venas:
        for s in ["Primavera Luminosa", "Verano Suave",
                  "Otoño Suave", "Invierno Verdadero"]:
            scores[s] += 2.0

    # ── 5. Joyería (plata vs oro) ─────────────────────────────────────────────
    joyeria = cuestionario.get("joyeria", "").lower()

    if "plata" in joyeria or "blanco" in joyeria:
        for s in ["Verano Verdadero", "Verano Frío", "Invierno Verdadero",
                  "Invierno Brillante"]:
            scores[s] += 2.0

    elif "oro" in joyeria or "dorado" in joyeria:
        for s in ["Primavera Verdadera", "Primavera Cálida",
                  "Otoño Verdadero", "Otoño Cálido"]:
            scores[s] += 2.0

    elif "ambas" in joyeria or "las dos" in joyeria:
        for s in ["Primavera Luminosa", "Verano Suave",
                  "Otoño Suave", "Invierno Verdadero"]:
            scores[s] += 1.5

    # ── 6. Cejas naturales ────────────────────────────────────────────────────
    cejas = cuestionario.get("color_cejas", "").lower()

    if any(x in cejas for x in ["negro", "negro azulado"]):
        scores["Invierno Verdadero"]  += 2.0
        scores["Invierno Oscuro"]     += 2.0

    elif any(x in cejas for x in ["castaño oscuro", "castaño"]):
        scores["Otoño Verdadero"]     += 1.5
        scores["Verano Verdadero"]    += 1.5
        scores["Invierno Verdadero"]  += 1.0

    elif any(x in cejas for x in ["castaño claro", "castaño dorado"]):
        scores["Primavera Cálida"]    += 2.0
        scores["Otoño Suave"]         += 1.5

    elif any(x in cejas for x in ["rubio", "dorado"]):
        scores["Primavera Verdadera"] += 2.0
        scores["Primavera Luminosa"]  += 1.5

    elif any(x in cejas for x in ["pelirrojo", "rojizo"]):
        scores["Otoño Verdadero"]     += 2.5
        scores["Otoño Cálido"]        += 2.0

    # ── 7. Reacción al sol ────────────────────────────────────────────────────
    sol = cuestionario.get("reaccion_sol", "").lower()

    if "enrojece" in sol or "se quema" in sol:
        for s in ["Verano Verdadero", "Verano Frío", "Invierno Brillante"]:
            scores[s] += 1.5

    elif "broncea" in sol:
        for s in ["Otoño Verdadero", "Otoño Cálido", "Primavera Cálida"]:
            scores[s] += 1.5

    elif "ambas" in sol:
        for s in ["Primavera Verdadera", "Verano Suave", "Otoño Suave"]:
            scores[s] += 1.0

    # ── 8. Contraste natural del rostro ───────────────────────────────────────
    contraste = cuestionario.get("contraste_rostro", "").lower()

    if "alto" in contraste:
        for s in ["Invierno Verdadero", "Invierno Brillante",
                  "Invierno Oscuro", "Primavera Luminosa"]:
            scores[s] += 2.0

    elif "medio" in contraste:
        for s in ["Primavera Verdadera", "Otoño Verdadero",
                  "Verano Verdadero"]:
            scores[s] += 1.5

    elif "bajo" in contraste or "suave" in contraste:
        for s in ["Verano Suave", "Otoño Suave", "Primavera Cálida"]:
            scores[s] += 2.0

    # ── 9. Piel sin maquillaje ────────────────────────────────────────────────
    piel_natural = cuestionario.get("piel_sin_maquillaje", "").lower()

    if any(x in piel_natural for x in ["rosada", "rosácea", "rojiza"]):
        for s in ["Verano Verdadero", "Verano Suave",
                  "Invierno Brillante", "Primavera Luminosa"]:
            scores[s] += 2.0

    elif any(x in piel_natural for x in ["amarilla", "dorada", "bronceada"]):
        for s in ["Primavera Cálida", "Otoño Verdadero", "Otoño Cálido"]:
            scores[s] += 2.0

    elif any(x in piel_natural for x in ["melocotón", "durazno", "cálida"]):
        for s in ["Primavera Verdadera", "Primavera Luminosa",
                  "Otoño Suave"]:
            scores[s] += 2.0

    elif any(x in piel_natural for x in ["oliva", "verde", "neutra"]):
        for s in ["Otoño Suave", "Verano Verdadero",
                  "Invierno Verdadero"]:
            scores[s] += 1.5

    elif any(x in piel_natural for x in ["marfil", "porcelana", "muy clara"]):
        for s in ["Verano Frío", "Invierno Brillante",
                  "Primavera Luminosa"]:
            scores[s] += 2.0

    # ── 10. Estilo de maquillaje (señal débil, solo desempata) ────────────────
    estilo = cuestionario.get("estilo_maquillaje", "").lower()

    if "natural" in estilo:
        scores["Primavera Verdadera"] += 0.5
        scores["Verano Suave"]        += 0.5
    elif "glamour" in estilo:
        scores["Invierno Brillante"]  += 0.5
        scores["Primavera Luminosa"]  += 0.5
    elif "editorial" in estilo:
        scores["Invierno Verdadero"]  += 0.5
        scores["Invierno Oscuro"]     += 0.5
    elif "romántico" in estilo:
        scores["Verano Verdadero"]    += 0.5
        scores["Primavera Verdadera"] += 0.5

    # ── Resultado: subtemporada con mayor puntaje ─────────────────────────────
    ganadora = max(scores, key=lambda s: scores[s])
    return ganadora


# ─────────────────────────────────────────────────────────────────────────────
# obtener_colores — paletas completas para las 12 subtemporadas
# ─────────────────────────────────────────────────────────────────────────────

PALETAS = {
    # ── PRIMAVERAS ────────────────────────────────────────────────────────────
    "Primavera Verdadera": {
        "favorables": ["coral cálido", "durazno", "verde lima", "turquesa claro",
                       "dorado suave", "camel", "amarillo mantequilla", "salmón"],
        "desfavorables": ["negro puro", "borgoña oscuro", "gris carbón",
                          "azul marino", "blanco puro"]
    },
    "Primavera Luminosa": {
        "favorables": ["coral luminoso", "melocotón brillante", "amarillo claro",
                       "verde menta", "violeta suave", "dorado brillante",
                       "turquesa", "nude rosado"],
        "desfavorables": ["negro puro", "marrón oscuro", "gris frío",
                          "verde oliva oscuro"]
    },
    "Primavera Cálida": {
        "favorables": ["terracota suave", "naranja cálido", "camel oscuro",
                       "verde musgo claro", "mostaza suave", "coral oscuro",
                       "bronce claro", "melocotón oscuro"],
        "desfavorables": ["rosa frío", "lila", "gris azulado",
                          "negro", "blanco frío"]
    },
    # ── VERANOS ───────────────────────────────────────────────────────────────
    "Verano Verdadero": {
        "favorables": ["rosa palo", "lavanda", "azul pizarra", "malva",
                       "gris perla", "azul acero suave", "rosa mauve",
                       "lila suave"],
        "desfavorables": ["naranja", "amarillo intenso", "verde oliva",
                          "café cálido", "dorado"]
    },
    "Verano Suave": {
        "favorables": ["nude rosado", "rosa polvoriento", "verde salvia",
                       "azul grisáceo", "lavanda grisácea", "beige rosado",
                       "malva suave", "ciruela suave"],
        "desfavorables": ["negro puro", "naranja intenso", "amarillo brillante",
                          "rojo tomate", "dorado"]
    },
    "Verano Frío": {
        "favorables": ["fucsia suave", "rosa frambuesa", "azul hielo",
                       "lila intenso", "plata", "gris azulado",
                       "rosa cereza", "azul lavanda"],
        "desfavorables": ["naranja", "café cálido", "verde oliva",
                          "dorado", "mostaza"]
    },
    # ── OTOÑOS ────────────────────────────────────────────────────────────────
    "Otoño Verdadero": {
        "favorables": ["terracota", "mostaza", "verde oliva", "café chocolate",
                       "naranja quemado", "bronce", "óxido", "verde cazador"],
        "desfavorables": ["rosa pastel", "azul eléctrico", "plateado",
                          "blanco puro", "fucsia"]
    },
    "Otoño Suave": {
        "favorables": ["terracota suave", "melocotón oscuro", "verde musgo",
                       "camel", "marrón medio", "durazno oscuro",
                       "verde salvia oscuro", "nude cálido"],
        "desfavorables": ["negro puro", "rosa frío", "azul marino",
                          "plateado", "fucsia"]
    },
    "Otoño Cálido": {
        "favorables": ["naranja intenso", "óxido", "dorado intenso",
                       "verde lima oscuro", "mostaza fuerte", "bronce oscuro",
                       "terracota oscura", "marrón rojizo"],
        "desfavorables": ["rosa pastel", "lila", "azul hielo",
                          "plateado", "blanco frío"]
    },
    # ── INVIERNOS ─────────────────────────────────────────────────────────────
    "Invierno Verdadero": {
        "favorables": ["negro", "blanco puro", "borgoña", "azul marino",
                       "rojo intenso", "plateado", "esmeralda", "azul real"],
        "desfavorables": ["naranja", "durazno", "café cálido",
                          "dorado", "verde oliva"]
    },
    "Invierno Brillante": {
        "favorables": ["fucsia eléctrico", "azul zafiro", "verde esmeralda",
                       "rojo cereza", "violeta intenso", "plateado brillante",
                       "negro", "blanco brillante"],
        "desfavorables": ["naranja", "mostaza", "café cálido",
                          "verde oliva", "durazno"]
    },
    "Invierno Oscuro": {
        "favorables": ["negro", "borgoña oscuro", "azul medianoche",
                       "verde botella", "ciruela oscuro", "granate",
                       "marrón oscuro frío", "gris antracita"],
        "desfavorables": ["rosa pastel", "amarillo", "naranja",
                          "durazno", "dorado"]
    },
}


def obtener_colores(subtemporada: str) -> dict:
    """Devuelve la paleta de colores para la subtemporada dada."""
    # Fallback: si la subtemporada no está en el dict, usar la estación base
    if subtemporada not in PALETAS:
        for base in ["Primavera", "Verano", "Otoño", "Invierno"]:
            if base in subtemporada:
                # Devolver la paleta de la temporada verdadera como fallback
                return PALETAS.get(f"{base} Verdadero",
                       PALETAS.get(f"{base} Verdadera",
                       {"favorables": [], "desfavorables": []}))
        return {"favorables": [], "desfavorables": []}
    return PALETAS[subtemporada]