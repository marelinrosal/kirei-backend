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
    return {"message": "Kirei backend — Fitzpatrick + Colorimetría ✅"}


# ── Paletas por temporada ─────────────────────────────────────────────────────
PALETAS = {
    "primavera_calida": {
        "favorables":    ["coral", "durazno", "dorado cálido", "verde manzana",
                          "turquesa claro", "amarillo miel", "salmón"],
        "desfavorables": ["negro puro", "gris frío", "borgoña oscuro", "azul marino intenso"],
    },
    "primavera_brillante": {
        "favorables":    ["coral brillante", "fucsia cálido", "amarillo limón",
                          "verde esmeralda", "turquesa vivo", "naranja cálido"],
        "desfavorables": ["beige apagado", "gris ratón", "marrón terroso"],
    },
    "primavera_clara": {
        "favorables":    ["melocotón", "rosa pálido", "lavanda suave", "menta",
                          "amarillo vainilla", "azul cielo", "blanco cálido"],
        "desfavorables": ["negro", "gris oscuro", "borgoña", "mostaza intensa"],
    },
    "verano_claro": {
        "favorables":    ["rosa empolvado", "lavanda", "azul periwinkle",
                          "verde salvia", "gris rosado", "blanco roto"],
        "desfavorables": ["naranja", "mostaza", "café cálido", "dorado", "verde oliva"],
    },
    "verano_suave": {
        "favorables":    ["malva", "lila", "azul acero", "verde grisáceo",
                          "rosa palo", "nude rosado", "borgoña suave"],
        "desfavorables": ["amarillo brillante", "naranja neón", "dorado intenso"],
    },
    "verano_frio": {
        "favorables":    ["fucsia frío", "rosa chicle", "azul eléctrico",
                          "morado pizarra", "plateado", "blanco puro"],
        "desfavorables": ["mostaza", "terracota", "verde oliva", "camel"],
    },
    "otono_calido": {
        "favorables":    ["terracota", "mostaza", "verde oliva", "camel",
                          "naranja tostado", "café dorado", "borgoña cálido"],
        "desfavorables": ["rosa frío", "azul eléctrico", "lila", "plateado", "gris frío"],
    },
    "otono_oscuro": {
        "favorables":    ["burdeos", "verde bosque", "marrón oscuro", "ocre",
                          "dorado antiguo", "cobre", "naranja quemado"],
        "desfavorables": ["rosa pastel", "azul bebé", "lavanda", "blanco puro"],
    },
    "otono_suave": {
        "favorables":    ["durazno apagado", "verde musgo", "tostado suave",
                          "terracota claro", "camel claro", "nude cálido"],
        "desfavorables": ["negro puro", "blanco brillante", "fucsia", "azul cobalto"],
    },
    "invierno_frio": {
        "favorables":    ["rosa frío", "lila", "azul cobalto", "gris perla",
                          "borgoña profundo", "azul pizarra", "lavanda"],
        "desfavorables": ["naranja", "mostaza", "verde oliva", "dorado", "camel"],
    },
    "invierno_oscuro": {
        "favorables":    ["negro", "blanco puro", "rojo intenso", "azul marino",
                          "morado real", "esmeralda oscuro", "plateado"],
        "desfavorables": ["beige", "camel", "durazno", "verde musgo", "dorado"],
    },
    "invierno_brillante": {
        "favorables":    ["fucsia eléctrico", "azul turquesa", "rojo cereza",
                          "verde esmeralda", "blanco brillante", "negro", "morado intenso"],
        "desfavorables": ["nude apagado", "mostaza", "naranja terroso", "camel"],
    },
}

PALETA_DEFAULT = {
    "favorables":    ["nude", "blush rosado", "taupe", "vino", "verde salvia", "azul marino"],
    "desfavorables": ["naranja neón", "amarillo limón", "fucsia intenso"],
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

        fototipo     = resultado["fototipo"]
        temporada    = resultado["temporada"]
        subtemporada = resultado["subtemporada"]
        confianza    = resultado["confianza"]
        paleta       = PALETAS.get(subtemporada, PALETA_DEFAULT)

        print(f"[KIREI] fototipo={fototipo}, temporada={temporada}, "
              f"subtemporada={subtemporada}, confianza={confianza}")

        # 4. Insertar en tabla analisis
        try:
            analisis_row = insertar("analisis", {
                "usuario_id":   usuario_id,
                "fototipo":     fototipo,
                "temporada":    temporada,
                "subtemporada": subtemporada,
                "confianza":    confianza,
            })
            print(f"[KIREI] analisis insertado: {analisis_row}")
        except Exception as e:
            print(f"[KIREI] ERROR insertando analisis: {traceback.format_exc()}")
            return JSONResponse(status_code=500,
                content={"error": f"Error guardando análisis: {str(e)}"})

        if analisis_row is None:
            print("[KIREI] Supabase devolvió None al insertar análisis")
            return JSONResponse(status_code=500,
                content={"error": "Supabase no devolvió datos al insertar análisis"})

        if isinstance(analisis_row, dict) and ("code" in analisis_row or "error" in analisis_row):
            msg = analisis_row.get("message") or analisis_row.get("error") or str(analisis_row)
            print(f"[KIREI] Error Supabase: {msg}")
            return JSONResponse(status_code=500,
                content={"error": f"Error en base de datos: {msg}"})

        try:
            analisis_id = analisis_row[0]["id"] if isinstance(analisis_row, list) else analisis_row["id"]
        except Exception as e:
            print(f"[KIREI] No se pudo extraer analisis_id de: {analisis_row}")
            return JSONResponse(status_code=500,
                content={"error": f"Error extrayendo ID: {str(e)}"})

        print(f"[KIREI] analisis_id: {analisis_id}")

        # 5. Guardar respuestas cuestionario
        try:
            insertar("respuestas_cuestionario", {
                "analisis_id": analisis_id,
                "skin":        datos.get("skin"),
                "eye":         datos.get("eye"),
                "hair":        datos.get("hair"),
                "sun":         datos.get("sun"),
                "freckles":    datos.get("freckles"),
                "tipo_piel":   datos.get("tipo_piel"),
                "forearm":     datos.get("forearm"),
                "hair_shine":  datos.get("hair_shine"),
            })
        except Exception as e:
            print(f"[KIREI] ADVERTENCIA respuestas_cuestionario: {e}")

        # 6. Guardar recomendaciones de colores
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

        # ── 7. Consultar productos por temporada ──────────────────────────────
        # FIX: filtramos por temporada (no fototipo, que no existe en la tabla).
        # Validamos que Supabase devuelva una lista y no un objeto de error.
        productos = []
        try:
            resultado_productos = consultar("productos", {"temporada": temporada})
            if isinstance(resultado_productos, list):
                productos = resultado_productos
            else:
                # Supabase devolvió un dict de error — lo logueamos y seguimos
                print(f"[KIREI] productos no es lista: {resultado_productos}")
        except Exception as e:
            print(f"[KIREI] ADVERTENCIA productos: {e}")

        print(f"[KIREI] OK — productos encontrados: {len(productos)}")

        return {
            "analisis_id":           analisis_id,
            "fototipo":              fototipo,
            "temporada":             temporada,
            "subtemporada":          subtemporada,
            "confianza":             confianza,
            "colores_favorables":    paleta["favorables"],
            "colores_desfavorables": paleta["desfavorables"],
            "productos":             productos,   # siempre es lista []
        }

    except Exception as e:
        print(f"[KIREI] ERROR no capturado: {traceback.format_exc()}")
        return JSONResponse(status_code=500,
            content={"error": f"Error interno: {str(e)}"})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)