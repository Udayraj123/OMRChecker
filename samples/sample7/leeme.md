# Sample 7: Alineación por logo (CropOnLogo) — Léeme

Esta muestra explica cómo usar OMRChecker cuando las hojas se alinean mediante un **logo fijo** en lugar de marcadores en las esquinas. El preprocesador localiza el logo en cada página, alinea la imagen y lee la cuadrícula de burbujas con una referencia estable.

(English: This sample shows how to run OMRChecker with logo-based alignment using the CropOnLogo pre-processor.)

**Repositorio (fork ilizaran):** [https://github.com/ilizaran/OMRChecker](https://github.com/ilizaran/OMRChecker)

---

## Objetivo

- **Meta:** Procesar imágenes de hojas OMR que comparten un logo (cabecera, marca) y no tienen marcadores en las esquinas.
- **Método:** Usar el preprocesador **CropOnLogo** para detectar el logo, trasladar la imagen hasta una posición fija y leer la cuadrícula de burbujas según el template.
- **Resultado:** Alineación uniforme y detección correcta de burbujas en todas las páginas.

---

## Contenido de esta carpeta

| Archivo        | Descripción |
|----------------|-------------|
| `template.json` | Layout OMR (tamaño de página, burbujas, bloques) y preprocesadores (CropOnLogo + GaussianBlur). |
| `config.json`   | Ajustes de salida, alineación y umbrales. |
| `logo.jpg`      | Imagen de referencia del logo: recorte exacto del logo tal como sale en las hojas escaneadas. Obligatorio para CropOnLogo. |
| `pagina_000.jpg` … `pagina_004.jpg` | Imágenes de ejemplo de hojas OMR a procesar. |

El logo debe estar en la **misma carpeta** que `template.json`. No se usa como imagen OMR.

---

## Requisitos

- **Python:** 3.x (3.5 o superior).
- **Sistema:** Linux o macOS recomendado; Windows compatible.
- **Librerías:** OpenCV (opencv-python) y el resto de dependencias del proyecto.

---

## 1. Instalar dependencias globales

### Python y pip

Comprobar que hay Python 3 y pip:

```bash
python3 --version
python3 -m pip --version
```

Actualizar pip si hace falta:

```bash
python3 -m pip install --user --upgrade pip
```

### OpenCV

```bash
python3 -m pip install --user opencv-python
python3 -m pip install --user opencv-contrib-python
```

En Linux a veces hacen falta librerías del sistema:

```bash
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran
```

### Windows

En **Símbolo del sistema** (cmd) o **PowerShell**:

1. **Instalar Python 3** desde [python.org](https://www.python.org/downloads/). En la instalación, activar **“Añadir Python al PATH”**. Comprobar:
   ```cmd
   py -3 --version
   py -3 -m pip --version
   ```

2. **Instalar OpenCV y dependencias** desde la raíz del proyecto (carpeta donde está `main.py`):
   ```cmd
   cd C:\ruta\al\OMRChecker
   py -3 -m pip install --user --upgrade pip
   py -3 -m pip install --user opencv-python
   py -3 -m pip install --user opencv-contrib-python
   py -3 -m pip install --user -r requirements.txt
   ```
   Si sale “Could not open requirements file”, ejecutar desde la carpeta donde está `requirements.txt`. Si hay errores de distutils, añadir `--ignore-installed` al último comando.

---

## 2. Instalar el proyecto (OMRChecker)

Desde la **raíz del repositorio** (carpeta que contiene `main.py` y `samples/`):

**Linux / macOS:**
```bash
cd /ruta/al/OMRChecker
git clone https://github.com/ilizaran/OMRChecker
cd OMRChecker
python3 -m pip install --user -r requirements.txt
```

**Windows:**
```cmd
cd C:\ruta\al\OMRChecker
git clone https://github.com/ilizaran/OMRChecker
cd OMRChecker
py -3 -m pip install --user -r requirements.txt
```

Si hay errores de distutils, añadir `--ignore-installed` al `pip install`.

---

## 3. Configuración

### 3.1 Template (`template.json`)

- **`pageDimensions`:** `[1241, 1754]` — tamaño de la hoja en píxeles (tras alinear) para la cuadrícula de burbujas.
- **`bubbleDimensions`:** `[30, 20]` — ancho y alto de cada burbuja.
- **`fieldBlocks`:** Define el bloque de preguntas (origen, etiquetas, huecos entre burbujas y entre preguntas).
- **`preProcessors`:** Debe incluir **CropOnLogo** y normalmente un desenfoque:
  - **CropOnLogo:** `relativePath` (ej. `"logo.jpg"`), `min_matching_threshold` (ej. `0.3`). Opcionales: `expected_origin`, `sheetToLogoWidthRatio`.

Guía completa: [CropOnLogo guide](../../docs/CropOnLogo-guide.md).

### 3.2 Config (`config.json`)

- **`outputs`:** Nivel de imágenes mostradas/guardadas.
- **`alignment_params`:** `auto_align` normalmente `false` con CropOnLogo.
- **`threshold_params`:** Detección de burbujas; ajustar si se pierden o malinterpretan marcas.

### 3.3 Imagen del logo (`logo.jpg`)

- **Recorte exacto** del logo como en las hojas escaneadas (mismo diseño, sin bordes extra).
- **Misma orientación** que en los escaneados.
- **Escala:** La hoja se redimensiona antes de buscar el logo; el archivo del logo debe tener una escala similar o usar `sheetToLogoWidthRatio`.
- Formato: cualquiera que OpenCV lea (PNG, JPG, etc.).

---

## 4. Cómo ejecutar el ejemplo

Todos los comandos desde la **raíz del repositorio** (donde está `main.py`).

### Opción A: Usar esta carpeta como entrada

```bash
python3 main.py -i ./samples/sample7
```

La salida se escribe en `outputs/samples/sample7/` (o la ruta indicada con `-o`).

### Opción B: Copiar a `inputs` y ejecutar

```bash
cp -r ./samples/sample7 inputs/
python3 main.py
```

**Windows:** crear la carpeta `inputs`, copiar el contenido de `samples/sample7` dentro y ejecutar `py -3 main.py`.

### Ajustar el layout

Si necesitas cambiar posiciones de burbujas o layout:

```bash
python3 main.py -i ./samples/sample7 --setLayout
```

Editar `template.json` y repetir hasta que el overlay coincida con la hoja.

---

## 5. Uso por línea de comandos

```text
python3 main.py [--setLayout] [--inputDir dir1 [dir2 ...]] [--outputDir dir]
```

- **`-i` / `--inputDir`:** Carpeta de entrada (por defecto `inputs`). Se pueden pasar varias.
- **`-o` / `--outputDir`:** Carpeta de salida (por defecto `outputs`).
- **`-l` / `--setLayout`:** Modo layout para ajustar la plantilla en pantalla.

En Windows usar `py -3` en lugar de `python3` si hace falta.

---

## 6. Resolución de problemas

| Problema | Qué hacer |
|----------|-----------|
| **"Logo not found"** | Comprobar `relativePath` en `template.json` y que `logo.jpg` exista en la misma carpeta que `template.json`. |
| **Puntuación de coincidencia baja** | Asegurar que `logo.jpg` sea un recorte exacto a la misma escala y orientación. Bajar un poco `min_matching_threshold` o usar `sheetToLogoWidthRatio`. |
| **Alineación incorrecta** | Ajustar `expected_origin` en las opciones de CropOnLogo. Usar `--setLayout` para revisar la imagen alineada. |
| **Burbujas mal leídas** | Ajustar `threshold_params` en `config.json` y/o posiciones y `bubbleDimensions` en `template.json`. Usar `--setLayout`. |

Más detalles: [CropOnLogo guide](../../docs/CropOnLogo-guide.md).

---

## 7. Resumen

1. Instalar Python 3, OpenCV y dependencias del proyecto desde la raíz del repo.
2. Tener en esta carpeta `template.json`, `config.json`, `logo.jpg` e imágenes de hojas (ej. `pagina_000.jpg` …).
3. Ejecutar desde la raíz: `python3 main.py -i ./samples/sample7` (o copiar a `inputs` y ejecutar `python3 main.py`).
4. Usar `--setLayout` para ajustar el layout; consultar la guía CropOnLogo para logo y alineación.

---

## 8. Resumen de resultados con `summarize_results.py`

Tras ejecutar OMRChecker se genera un **CSV de resultados** (p. ej. en `outputs/samples/sample7/Results/`) con una fila por hoja y columnas por pregunta (`q1`, `q2`, …). El script **summarize_results.py** convierte ese CSV en una hoja de resumen: conteos por puntuación (0–10) por pregunta, número de encuestas y medias.

### Qué hace el script

- **Entrada:** CSV de resultados de OMRChecker (columnas `q1`, `q2`, … con valores 0–10 o dígitos concatenados).
- **Salida:** CSV de resumen con tabla (pregunta × puntuación 0–10), fila TOTAL, número de encuestas y media global y por pregunta.

Si se marca un solo valor se usa ese; si hay varios, se toma el mayor; si se marcan las 11 opciones se considera inválido.

### Cómo ejecutarlo

Desde la **raíz del repositorio**:

```bash
# Por defecto: lee outputs/Results/Results_01PM.csv, escribe resultado.csv
python3 scripts/summarize_results.py

# Entrada y salida personalizadas
python3 scripts/summarize_results.py ruta/al/resultados.csv ruta/resultado.csv
```

**Windows:**

```cmd
py -3 scripts\summarize_results.py
py -3 scripts\summarize_results.py ruta\resultados.csv ruta\resultado.csv
```

Ejemplo tras ejecutar sample7:

```bash
python3 scripts/summarize_results.py outputs/samples/sample7/Results/Results_01PM.csv resultado.csv
```

### Opciones

- **`-h` / `--help`** — Muestra uso y rutas por defecto.

Las etiquetas de las preguntas (`q1`–`q6`) están definidas en el script; para cambiarlas, editar el diccionario `question_labels` en `scripts/summarize_results.py`.
