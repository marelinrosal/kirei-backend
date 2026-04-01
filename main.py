from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

from analysis.color_analysis import analyze_color
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
    return {"message": "Kirei backend — Fitzpatrick ✅"}


# ── Paletas por subtono ───────────────────────────────────────────────────────

PALETAS = {
    "frio": {
        "favorables": [
            "rosa palo",
            "lavanda",
            "azul cielo",
            "lila",
            "gris perla",
            "borgoña profundo",
            "azul pizarra",
        ],
        "desfavorables": [
            "naranja",
            "mostaza",
            "verde oliva",
            "dorado",
            "café cálido",
        ],
    },
    "calido": {
        "favorables": [
            "terracota",
            "coral",
            "durazno",
            "mostaza",
            "verde oliva",
            "dorado",
            "camel",
        ],
        "desfavorables": [
            "rosa frío",
            "lila",
            "azul eléctrico",
            "plateado",
            "gris frío",
        ],
    },
    "neutro": {
        "favorables": [
            "nude",
            "blush rosado",
            "taupe",
            "vino",
            "verde salvia",
            "azul marino",
        ],
        "desfavorables": [
            "naranja neón",
            "amarillo limón",
            "fucsia intenso",
        ],
    },
}


# ── Endpoint principal ────────────────────────────────────────────────────────

@app.post("/analizar")
async def analizar(
    file:         UploadFile = File(...),
    usuario_id:   str        = Form(...),
    cuestionario: str        = Form(...)
):
    image_bytes        = await file.read()
    datos_cuestionario = json.loads(cuestionario)

    # ── Análisis colorimétrico ────────────────────────────────────────────────
    resultado = analyze_color(image_bytes, datos_cuestionario)

    fototipo  = resultado["fototipo"]
    subtono   = resultado["subtono"]
    confianza = resultado["confianza"]

    paleta = PALETAS.get(subtono, PALETAS["neutro"])

    # ── Guardar análisis en Supabase ──────────────────────────────────────────
    analisis_row = insertar("analisis", {
        "usuario_id": usuario_id,
        "fototipo":   fototipo,
        "subtono":    subtono,
        "confianza":  confianza,
    })

    if isinstance(analisis_row, dict) and "code" in analisis_row:
        return {"error": f"Error en Supabase: {analisis_row.get('message', '')}"}

    analisis_id = (
        analisis_row[0]["id"]
        if isinstance(analisis_row, list)
        else analisis_row["id"]
    )

    # ── Guardar respuestas del cuestionario ───────────────────────────────────
    insertar("respuestas_cuestionario", {
        "analisis_id": analisis_id,
        "skin":        datos_cuestionario.get("skin"),
        "eye":         datos_cuestionario.get("eye"),
        "hair":        datos_cuestionario.get("hair"),
        "vein":        datos_cuestionario.get("vein"),
        "sun":         datos_cuestionario.get("sun"),
        "freckles":    datos_cuestionario.get("freckles"),
        "base":        datos_cuestionario.get("base"),
    })

    # ── Guardar recomendaciones de color ──────────────────────────────────────
    for color in paleta["favorables"]:
        insertar("recomendaciones", {
            "analisis_id":  analisis_id,
            "tipo":         "color",
            "valor":        color,
            "es_favorable": True,
        })
    for color in paleta["desfavorables"]:
        insertar("recomendaciones", {
            "analisis_id":  analisis_id,
            "tipo":         "color",
            "valor":        color,
            "es_favorable": False,
        })

    # ── Consultar productos recomendados ──────────────────────────────────────
    productos = consultar("productos", {"subtono": subtono})

    return {
        "analisis_id":           analisis_id,
        "fototipo":              fototipo,
        "subtono":               subtono,
        "confianza":             confianza,
        "colores_favorables":    paleta["favorables"],
        "colores_desfavorables": paleta["desfavorables"],
        "productos":             productos,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)