from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

from analysis.color_analysis import analyze_color, determinar_subtemporada, obtener_colores
from database.supabase_client import insertar, consultar

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Kirei backend funcionando ✅"}


@app.post("/analizar")
async def analizar(
    file:        UploadFile = File(...),
    usuario_id:  str        = Form(...),
    cuestionario: str       = Form(...)
):
    # ── 1. Leer imagen ────────────────────────────────────────────────────────
    image_bytes = await file.read()
    resultado_color = analyze_color(image_bytes)

    if "error" in resultado_color:
        return {"error": resultado_color["error"]}

    # ── 2. Parsear cuestionario ───────────────────────────────────────────────
    try:
        datos_cuestionario = json.loads(cuestionario)
    except json.JSONDecodeError:
        return {"error": "El cuestionario no es un JSON válido"}

    # ── 3. Determinar subtemporada con sistema ponderado ──────────────────────
    subtemporada = determinar_subtemporada(resultado_color, datos_cuestionario)

    # ── 4. Guardar análisis en Supabase ───────────────────────────────────────
    analisis = insertar("analisis", {
        "usuario_id": usuario_id,
        "tono":       resultado_color["tono"],
        "subtono":    resultado_color["subtono"],
        "temporada":  subtemporada          # ahora guarda la subtemporada completa
    })

    if isinstance(analisis, dict) and "code" in analisis:
        return {"error": f"Error en Supabase: {analisis.get('message', 'desconocido')}"}

    analisis_id = analisis[0]["id"] if isinstance(analisis, list) else analisis["id"]

    # ── 5. Guardar respuestas del cuestionario ────────────────────────────────
    insertar("respuestas_cuestionario", {
        "analisis_id":          analisis_id,
        # Preguntas originales
        "color_ojos":           datos_cuestionario.get("color_ojos", ""),
        "color_cabello":        datos_cuestionario.get("color_cabello", ""),
        "reaccion_sol":         datos_cuestionario.get("reaccion_sol", ""),
        "estilo_maquillaje":    datos_cuestionario.get("estilo_maquillaje", ""),
        # Preguntas nuevas (pueden venir vacías si el cliente es antiguo)
        "venas_muneca":         datos_cuestionario.get("venas_muneca", ""),
        "joyeria":              datos_cuestionario.get("joyeria", ""),
        "color_cejas":          datos_cuestionario.get("color_cejas", ""),
        "contraste_rostro":     datos_cuestionario.get("contraste_rostro", ""),
        "piel_sin_maquillaje":  datos_cuestionario.get("piel_sin_maquillaje", ""),
    })

    # ── 6. Obtener paleta de colores ──────────────────────────────────────────
    colores = obtener_colores(subtemporada)

    # ── 7. Guardar recomendaciones de colores ─────────────────────────────────
    for color in colores["favorables"]:
        insertar("recomendaciones", {
            "analisis_id":  analisis_id,
            "tipo":         "color",
            "valor":        color,
            "es_favorable": True
        })
    for color in colores["desfavorables"]:
        insertar("recomendaciones", {
            "analisis_id":  analisis_id,
            "tipo":         "color",
            "valor":        color,
            "es_favorable": False
        })

    # ── 8. Obtener productos recomendados desde Supabase ──────────────────────
    # Busca por subtemporada exacta primero; si no hay, busca por estación base
    productos = consultar("productos", {"temporada": subtemporada})
    if not productos:
        temporada_base = subtemporada.split()[0]   # "Primavera", "Verano", etc.
        productos = consultar("productos", {"temporada": temporada_base})

    # ── 9. Respuesta ──────────────────────────────────────────────────────────
    return {
        "analisis_id":           analisis_id,
        "tono":                  resultado_color["tono"],
        "subtono":               resultado_color["subtono"],
        "temporada":             subtemporada,
        "colores_favorables":    colores["favorables"],
        "colores_desfavorables": colores["desfavorables"],
        "productos":             productos
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)