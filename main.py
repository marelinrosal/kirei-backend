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
    return {"message": "Kirei backend funcionando ✅"}


@app.post("/analizar")
async def analizar(
    file:         UploadFile = File(...),
    usuario_id:   str        = Form(...),
    cuestionario: str        = Form(...)
):
    image_bytes     = await file.read()
    resultado_color = analyze_color(image_bytes)

    if "error" in resultado_color:
        return {"error": resultado_color["error"]}

    datos_cuestionario = json.loads(cuestionario)

    temporada = afinar_temporada(
        resultado_color["temporada"],
        resultado_color["subtono"],
        datos_cuestionario
    )

    analisis = insertar("analisis", {
        "usuario_id": usuario_id,
        "tono":       resultado_color["tono"],
        "subtono":    resultado_color["subtono"],
        "temporada":  temporada
    })

    if isinstance(analisis, dict) and "code" in analisis:
        return {"error": f"Error en Supabase: {analisis['message']}"}

    analisis_id = analisis[0]["id"] if isinstance(analisis, list) else analisis["id"]

    insertar("respuestas_cuestionario", {
        "analisis_id":       analisis_id,
        "color_ojos":        datos_cuestionario["color_ojos"],
        "color_cabello":     datos_cuestionario["color_cabello"],
        "reaccion_sol":      datos_cuestionario["reaccion_sol"],
        "estilo_maquillaje": datos_cuestionario["estilo_maquillaje"]
    })

    productos = consultar("productos", {"temporada": temporada})
    colores   = obtener_colores(temporada)

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

    return {
        "analisis_id":           analisis_id,
        "tono":                  resultado_color["tono"],
        "subtono":               resultado_color["subtono"],
        "temporada":             temporada,
        "colores_favorables":    colores["favorables"],
        "colores_desfavorables": colores["desfavorables"],
        "productos":             productos
    }


def afinar_temporada(temporada: str, subtono: str, cuestionario: dict) -> str:
    color_ojos    = cuestionario.get("color_ojos", "").lower()
    color_cabello = cuestionario.get("color_cabello", "").lower()
    reaccion_sol  = cuestionario.get("reaccion_sol", "").lower()

    indicadores_frios = [
        "azul"  in color_ojos,
        "gris"  in color_ojos,
        "cenizo" in color_cabello,
        "negro"  in color_cabello,
        "se enrojece" in reaccion_sol
    ]
    indicadores_calidos = [
        "café"     in color_ojos,
        "miel"     in color_ojos,
        "verde"    in color_ojos,
        "dorado"   in color_cabello,
        "castaño"  in color_cabello,
        "se broncea" in reaccion_sol
    ]

    frios   = sum(indicadores_frios)
    calidos = sum(indicadores_calidos)

    if calidos > frios + 1:
        if temporada in ["Verano", "Invierno"]:
            return "Primavera" if "claro" in temporada else "Otoño"
    elif frios > calidos + 1:
        if temporada in ["Primavera", "Otoño"]:
            return "Verano" if "claro" in temporada else "Invierno"

    return temporada


def obtener_colores(temporada: str) -> dict:
    paletas = {
        "Primavera": {
            "favorables":    ["coral", "durazno", "dorado", "verde lima",
                              "turquesa", "camel"],
            "desfavorables": ["negro puro", "borgoña oscuro", "gris frío",
                              "azul marino"]
        },
        "Verano": {
            "favorables":    ["rosa palo", "lavanda", "azul pizarra",
                              "malva", "gris perla"],
            "desfavorables": ["naranja", "amarillo intenso", "verde oliva",
                              "café cálido"]
        },
        "Otoño": {
            "favorables":    ["terracota", "mostaza", "verde oliva",
                              "café chocolate", "naranja quemado"],
            "desfavorables": ["rosa pastel", "azul eléctrico", "plateado",
                              "blanco puro"]
        },
        "Invierno": {
            "favorables":    ["negro", "blanco puro", "borgoña", "azul marino",
                              "rojo intenso", "plateado"],
            "desfavorables": ["naranja", "durazno", "café cálido", "dorado",
                              "verde oliva"]
        }
    }
    return paletas.get(temporada, paletas["Verano"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)