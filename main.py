from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import traceback

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


PALETAS = {
    "frio": {
        "favorables":    ["rosa palo", "lavanda", "azul cielo", "lila",
                          "gris perla", "borgoña profundo", "azul pizarra"],
        "desfavorables": ["naranja", "mostaza", "verde oliva", "dorado", "café cálido"],
    },
    "calido": {
        "favorables":    ["terracota", "coral", "durazno", "mostaza",
                          "verde oliva", "dorado", "camel"],
        "desfavorables": ["rosa frío", "lila", "azul eléctrico", "plateado", "gris frío"],
    },
    "neutro": {
        "favorables":    ["nude", "blush rosado", "taupe", "vino",
                          "verde salvia", "azul marino"],
        "desfavorables": ["naranja neón", "amarillo limón", "fucsia intenso"],
    },
}


@app.post("/analizar")
async def analizar(
    file:         UploadFile = File(...),
    usuario_id:   str        = Form(...),
    cuestionario: str        = Form(...)
):
    try:
        # 1. Leer imagen
        image_bytes = await file.read()
        print(f"[KIREI] Imagen recibida: {len(image_bytes)} bytes")

        # 2. Parsear cuestionario
        try:
            datos = json.loads(cuestionario)
        except Exception as e:
            return JSONResponse(status_code=400,
                content={"error": f"Cuestionario JSON inválido: {str(e)}"})

        print(f"[KIREI] Cuestionario: {datos}")
        print(f"[KIREI] usuario_id: {usuario_id}")

        # 3. Análisis colorimétrico
        try:
            resultado = analyze_color(image_bytes, datos)
        except Exception as e:
            print(f"[KIREI] ERROR analyze_color: {traceback.format_exc()}")
            return JSONResponse(status_code=500,
                content={"error": f"Error en análisis de color: {str(e)}"})

        fototipo  = resultado["fototipo"]
        subtono   = resultado["subtono"]
        confianza = resultado["confianza"]
        paleta    = PALETAS.get(subtono, PALETAS["neutro"])

        print(f"[KIREI] fototipo={fototipo}, subtono={subtono}, confianza={confianza}")

        # 4. Insertar en tabla analisis
        try:
            analisis_row = insertar("analisis", {
                "usuario_id": usuario_id,
                "fototipo":   fototipo,
                "subtono":    subtono,
                "confianza":  confianza,
            })
            print(f"[KIREI] analisis insertado: {analisis_row}")
        except Exception as e:
            print(f"[KIREI] ERROR insertando analisis: {traceback.format_exc()}")
            return JSONResponse(status_code=500,
                content={"error": f"Error guardando análisis: {str(e)}"})

        # Validar respuesta de Supabase
        if analisis_row is None:
            print("[KIREI] Supabase devolvió None al insertar análisis")
            return JSONResponse(status_code=500,
                content={"error": "Supabase no devolvió datos al insertar análisis"})

        if isinstance(analisis_row, dict) and ("code" in analisis_row or "error" in analisis_row):
            msg = analisis_row.get("message") or analisis_row.get("error") or str(analisis_row)
            print(f"[KIREI] Error Supabase: {msg}")
            return JSONResponse(status_code=500,
                content={"error": f"Error en base de datos: {msg}"})

        # Extraer ID
        try:
            analisis_id = analisis_row[0]["id"] if isinstance(analisis_row, list) else analisis_row["id"]
        except Exception as e:
            print(f"[KIREI] No se pudo extraer analisis_id de: {analisis_row}")
            return JSONResponse(status_code=500,
                content={"error": f"Error extrayendo ID: {str(e)}"})

        print(f"[KIREI] analisis_id: {analisis_id}")

        # 5. Guardar respuestas cuestionario (no bloquea si falla)
        try:
            insertar("respuestas_cuestionario", {
                "analisis_id": analisis_id,
                "skin":        datos.get("skin"),
                "eye":         datos.get("eye"),
                "hair":        datos.get("hair"),
                "vein":        datos.get("vein"),
                "sun":         datos.get("sun"),
                "freckles":    datos.get("freckles"),
                "base":        datos.get("base"),
            })
        except Exception as e:
            print(f"[KIREI] ADVERTENCIA respuestas_cuestionario: {e}")

        # 6. Guardar recomendaciones (no bloquea si falla)
        try:
            for color in paleta["favorables"]:
                insertar("recomendaciones", {
                    "analisis_id": analisis_id, "tipo": "color",
                    "valor": color, "es_favorable": True,
                })
            for color in paleta["desfavorables"]:
                insertar("recomendaciones", {
                    "analisis_id": analisis_id, "tipo": "color",
                    "valor": color, "es_favorable": False,
                })
        except Exception as e:
            print(f"[KIREI] ADVERTENCIA recomendaciones: {e}")

        # 7. Consultar productos (no bloquea si falla)
        productos = []
        try:
            productos = consultar("productos", {"subtono": subtono}) or []
        except Exception as e:
            print(f"[KIREI] ADVERTENCIA productos: {e}")

        print(f"[KIREI] OK — productos: {len(productos)}")

        return {
            "analisis_id":           analisis_id,
            "fototipo":              fototipo,
            "subtono":               subtono,
            "confianza":             confianza,
            "colores_favorables":    paleta["favorables"],
            "colores_desfavorables": paleta["desfavorables"],
            "productos":             productos,
        }

    except Exception as e:
        print(f"[KIREI] ERROR no capturado: {traceback.format_exc()}")
        return JSONResponse(status_code=500,
            content={"error": f"Error interno: {str(e)}"})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)