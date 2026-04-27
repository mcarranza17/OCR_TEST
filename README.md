# OCR_TEST

MVP local para verificacion de identidad con DNI de Honduras.

El objetivo de esta demo es combinar:

- OCR documental para leer texto de la tarjeta de identidad hondurena.
- Deteccion facial en la foto del documento.
- Comparacion facial entre la foto del documento y una selfie.
- Una decision final simple: `verified`, `manual_review` o `rejected`.

> Nota: esto es una demo tecnica. No sustituye una validacion oficial del RNP ni debe usarse en produccion sin consentimiento explicito, controles de privacidad, auditoria, cifrado y calibracion con datos reales.

## Modelo elegido

### OCR: PaddleOCR

Se usa PaddleOCR porque es robusto para texto en imagenes reales, soporta multiples idiomas y funciona bien en fotografias con ruido, perspectiva o iluminacion imperfecta.

### Rostros: InsightFace `buffalo_l`

Se usa InsightFace porque `buffalo_l` empaqueta deteccion de rostro, landmarks/alineacion y embeddings faciales. El flujo compara embeddings con similitud coseno.

Importante: los modelos de InsightFace pueden tener restricciones de licencia para uso comercial. Revisar la licencia antes de cualquier despliegue fuera de demo/investigacion.

## Pipeline

1. Subir imagen frontal del DNI hondureno.
2. Ejecutar OCR y conservar el texto crudo con confianza promedio.
3. Parsear campos esperados:
   - numero de identidad
   - nombres
   - apellidos
   - fecha de nacimiento
   - fecha de expiracion
   - lugar de nacimiento
4. Detectar el rostro visible en el documento.
5. Subir selfie o foto de prueba de la persona.
6. Detectar el rostro en la selfie.
7. Generar embeddings faciales para ambos rostros.
8. Comparar embeddings.
9. Emitir decision:
   - `verified`: OCR valido y rostro coincide.
   - `manual_review`: OCR o rostro quedan cerca del umbral.
   - `rejected`: no se pudo validar o el rostro no coincide.

## Estructura

```text
OCR_TEST/
  app.py
  requirements.txt
  src/
    dni_parser.py
    face_matcher.py
    image_io.py
    ocr.py
    settings.py
  tests/
    test_dni_parser.py
  data/
    enrolled/
    probes/
    db/
```

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar demo

```bash
streamlit run app.py
```

La primera ejecucion puede descargar pesos de modelos de OCR/rostros.
Los modelos y caches se guardan localmente en `.models/` para no escribir en el home del usuario.

## Calibracion requerida

Los umbrales por defecto son solo una base para demo:

- `FACE_MATCH_THRESHOLD=0.45`
- `FACE_REVIEW_THRESHOLD=0.35`
- `MODEL_CACHE_DIR=.models`

La app configura estas caches locales al arrancar:

- `PADDLE_PDX_CACHE_HOME`
- `MPLCONFIGDIR`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `MODELSCOPE_CACHE`

Para usarlo seriamente hay que calibrar con imagenes reales del flujo:

- fotos buenas y malas del DNI
- selfies con cambios de luz
- lentes, angulos y expresiones
- pares correctos e incorrectos

## Fuentes de referencia

- RNP Honduras DNI: https://www.rnp.hn/sites/rnpweb/DNI
- Identidad digital BIEN/RNP: https://app.rnp.hn/bien/
- InsightFace model zoo: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- InsightFace ArcFace: https://www.insightface.ai/research/arcface
- InsightFace RetinaFace: https://insightface.ai/retinaface
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
