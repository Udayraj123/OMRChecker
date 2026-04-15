# Sample 7: Logo-based alignment (CropOnLogo)

This sample shows how to run OMRChecker when sheets are aligned using a **fixed logo** instead of corner markers. The pre-processor finds the logo on each page with template matching and aligns the image so the logo sits at a known position, giving a stable reference for the bubble grid.

(En español: Esta muestra explica cómo usar OMRChecker cuando las hojas se alinean mediante un **logo fijo** en lugar de marcadores en las esquinas.)

**Repository (ilizaran fork):** [https://github.com/ilizaran/OMRChecker](https://github.com/ilizaran/OMRChecker)

---

## Objective

- **Goal:** Process OMR sheet images that share a common logo (e.g. header or branding) and have no corner/edge markers.
- **Method:** Use the **CropOnLogo** pre-processor to detect the logo, translate the image so the logo is at a fixed position, then read the bubble grid from the template.
- **Outcome:** You get consistent alignment and correct bubble detection across all pages.

(Objetivo: procesar imágenes de hojas OMR con un logo común y sin marcadores en las esquinas, usando CropOnLogo para alinear y leer las burbujas.)

---

## Contents of this folder

| File / asset      | Description |
|-------------------|-------------|
| `template.json`  | OMR layout (page size, bubble size, block positions) and pre-processors (CropOnLogo + GaussianBlur). |
| `config.json`    | Tuning (output verbosity, alignment, threshold). |
| `logo.jpg`       | Reference logo image: exact crop of the logo as it appears on the scanned sheets. Required by CropOnLogo. |
| `pagina_000.jpg` … `pagina_004.jpg` | Sample OMR sheet images to process. |

The logo file must be in the **same directory** as `template.json`. It is excluded automatically from being treated as an OMR image.

---

## Prerequisites

- **Python:** 3.x (3.5+).
- **System:** Linux or macOS recommended; Windows supported.
- **Libraries:** OpenCV (opencv-python) and the rest of the project dependencies.

(Requisitos: Python 3, OpenCV y dependencias del proyecto.)

---

## 1. Install global dependencies

### Python and pip

Check that Python 3 and pip are available:

```bash
python3 --version
python3 -m pip --version
```

Install or upgrade pip if needed:

```bash
python3 -m pip install --user --upgrade pip
```

### OpenCV

Install OpenCV (required for image processing and template matching):

```bash
python3 -m pip install --user opencv-python
python3 -m pip install --user opencv-contrib-python
```

On some Linux systems you may need system libraries first:

```bash
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran
```

### Windows

On Windows, use **Command Prompt** (cmd) or **PowerShell**:

1. **Install Python 3**  
   Download the installer from [python.org](https://www.python.org/downloads/). During setup, enable **“Add Python to PATH”**. Verify:
   ```cmd
   py -3 --version
   py -3 -m pip --version
   ```
   If you use `python` instead of `py -3`, ensure it points to Python 3.

2. **Install OpenCV and dependencies**  
   From the project root (folder containing `main.py`):
   ```cmd
   cd C:\path\to\OMRChecker
   py -3 -m pip install --user --upgrade pip
   py -3 -m pip install --user opencv-python
   py -3 -m pip install --user opencv-contrib-python
   py -3 -m pip install --user -r requirements.txt
   ```
   If you get “Could not open requirements file”, run the command from the directory where `requirements.txt` is located. If you see distutils-related errors, add `--ignore-installed` to the last command.

(En Windows: instalar Python 3 desde python.org con “Add to PATH”, luego desde la raíz del proyecto ejecutar pip para opencv y requirements.txt.)

---

## 2. Install the project (OMRChecker)

From the **repository root** (the folder that contains `main.py` and `samples/`):

**Linux / macOS:**
```bash
cd /path/to/OMRChecker
git clone https://github.com/ilizaran/OMRChecker   # or Udayraj123/OMRChecker
cd OMRChecker
python3 -m pip install --user -r requirements.txt
```

**Windows (cmd or PowerShell):**
```cmd
cd C:\path\to\OMRChecker
git clone https://github.com/ilizaran/OMRChecker
cd OMRChecker
py -3 -m pip install --user -r requirements.txt
```

If you see distutils-related errors, try adding `--ignore-installed` to the pip install command.

(Instalación: clonar el repo, entrar en OMRChecker e instalar dependencias con pip.)

---

## 3. Configuration

### 3.1 Template (`template.json`)

- **`pageDimensions`:** `[1241, 1754]` — size of the sheet in pixels (after alignment) used for the bubble grid.
- **`bubbleDimensions`:** `[30, 20]` — width and height of each bubble.
- **`fieldBlocks`:** Defines the block of questions (e.g. `SatisfactionBlock`) with:
  - `origin`: top-left of the block in sheet coordinates.
  - `bubbleValues`, `direction`, `fieldLabels`, `bubblesGap`, `labelsGap`: layout of bubbles and labels.
- **`preProcessors`:** Must include **CropOnLogo** and typically a blur step:
  - **CropOnLogo** `options`:
    - `relativePath`: `"logo.jpg"` — path to the logo image relative to the template directory.
    - `min_matching_threshold`: `0.3` — minimum match score (0–1). Lower = more tolerant; raise if you get false matches.
  - Optional: `expected_origin` `[x, y]` where the logo’s top-left should end up (default `[0, 0]`).
  - Optional: `sheetToLogoWidthRatio` if the logo file’s size does not match its size on the resized sheet.

See the [CropOnLogo guide](../../docs/CropOnLogo-guide.md) for all options and troubleshooting.

### 3.2 Config (`config.json`)

- **`outputs`:** `show_image_level`, `save_image_level` — control how many debug images are shown or saved.
- **`alignment_params`:** `auto_align` — typically `false` when using CropOnLogo.
- **`threshold_params`:** Bubble detection (e.g. `BUBBLE_VALUE_METHOD`, `MARK_SENSITIVITY_OFFSET`). Adjust if marks are missed or misread.

You can keep the provided `config.json` as-is for a first run.

### 3.3 Logo image (`logo.jpg`)

- Must be an **exact crop** of the logo as it appears on the scanned sheets (same design, no extra borders).
- Same **orientation** as in the scans (no rotation).
- **Scale:** Sheets are resized to `processing_width` (default 666 px) before matching. The logo file should have a similar relative size; if not, use `sheetToLogoWidthRatio` in CropOnLogo options.
- Format: any format OpenCV can read (e.g. PNG, JPG).

(Configuración: template.json define layout y CropOnLogo; config.json ajusta salidas y umbrales; logo.jpg debe ser un recorte exacto del logo en las hojas.)

---

## 4. How to run the example

All commands are run from the **OMRChecker repository root** (where `main.py` is).

### Option A: Use this sample folder as input

```bash
python3 main.py -i ./samples/sample7
```

Output is written to `outputs/samples/sample7/` (or the path implied by your `-o` / `--outputDir`).

### Option B: Copy sample to `inputs` and run

```bash
cp -r ./samples/sample7 inputs/
python3 main.py
```

Output will be under `outputs/sample7/` (or your configured output directory).

### Optional: Set or adjust the layout

If you need to adjust bubble positions or layout:

```bash
python3 main.py -i ./samples/sample7 --setLayout
```

Then edit `template.json` (e.g. origins, gaps) and run again until the overlay matches the sheet.

(Ejecución: desde la raíz del repo, `python3 main.py -i ./samples/sample7` o copiar sample7 a inputs y ejecutar `python3 main.py`.)

---

## 5. Full command-line usage

```text
python3 main.py [--setLayout] [--inputDir dir1 [dir2 ...]] [--outputDir dir]
```

- **`-i` / `--inputDir`:** Input directory (default: `inputs`). Can pass multiple (e.g. `-i ./samples/sample7 ./samples/sample1`).
- **`-o` / `--outputDir`:** Output directory (default: `outputs`).
- **`-l` / `--setLayout`:** Enable layout mode to adjust the template visually.

---

## 6. Troubleshooting

| Issue | What to do |
|-------|-------------|
| **"Logo not found"** | Check that `relativePath` in `template.json` is correct and that `logo.jpg` exists in the same folder as `template.json`. |
| **Match score too low** | Ensure `logo.jpg` is an exact crop at the same scale and orientation as on the sheet. Try lowering `min_matching_threshold` slightly (e.g. 0.25) or set `sheetToLogoWidthRatio` if the logo size on the resized sheet differs. |
| **Wrong alignment** | Adjust `expected_origin` in CropOnLogo options so the logo’s top-left ends where your template expects. Use `--setLayout` to inspect the aligned image. |
| **Bubbles not read correctly** | Tune `threshold_params` in `config.json` and/or bubble positions and `bubbleDimensions` in `template.json`. Use `--setLayout` to verify the grid. |

More details: [CropOnLogo guide](../../docs/CropOnLogo-guide.md).

(Problemas: logo no encontrado → comprobar ruta y archivo; puntuación baja → recorte exacto y escala; alineación incorrecta → expected_origin y --setLayout.)

---

## 7. Summary

1. Install Python 3, OpenCV, and project dependencies from the repo root.
2. Ensure `template.json`, `config.json`, and `logo.jpg` are in this folder, with sheet images (e.g. `pagina_000.jpg` …).
3. Run from repo root: `python3 main.py -i ./samples/sample7` (or copy to `inputs` and run `python3 main.py`).
4. Use `--setLayout` to adjust layout; use the CropOnLogo guide for logo and alignment issues.

(Resumen: instalar dependencias, tener template, config y logo en la carpeta, ejecutar main.py con esta carpeta como entrada.)

---

## 8. Summarizing results with `summarize_results.py`

After OMRChecker runs, it writes a **Results CSV** (e.g. under `outputs/samples/sample7/Results/` or `outputs/Results/`) with one row per sheet and columns for each question (e.g. `q1`, `q2`, …). The **summarize** script turns that CSV into a single summary spreadsheet: counts per score (0–10) per question, total surveys, and mean scores.

(Tras ejecutar OMRChecker se genera un CSV de resultados; el script de resumen genera una hoja con conteos por puntuación y medias.)

### What the script does

- **Input:** A Results CSV produced by OMRChecker (columns like `q1`, `q2`, … with values 0–10 or concatenated digits).
- **Output:** A summary CSV that includes:
  - A table: each row = one question, each column = score 0–10 (count of responses).
  - A “TOTAL” row with column totals.
  - Number of surveys, overall mean, and per-question mean.

Scores are parsed so that a single marked value (e.g. `7`) or `10` is taken as that score; if several are marked, the highest is used; if all 11 options are marked, the response is treated as invalid and excluded.

### How to run it

From the **repository root** (so that default paths resolve correctly, or pass paths explicitly):

```bash
# Default: reads outputs/Results/Results_01PM.csv, writes resultado.csv
python3 scripts/summarize_results.py

# Custom input and output
python3 scripts/summarize_results.py path/to/your_results.csv path/to/resultado.csv
```

**Windows (cmd or PowerShell):**

```cmd
py -3 scripts\summarize_results.py
py -3 scripts\summarize_results.py path\to\your_results.csv path\to\resultado.csv
```

After running sample7, the Results CSV is typically at `outputs/samples/sample7/Results/` (exact filename may vary). Example:

```bash
python3 scripts/summarize_results.py outputs/samples/sample7/Results/Results_01PM.csv resultado.csv
```

### Options

- **`-h` / `--help`** — Show usage and default paths.

The script uses fixed labels for `q1`–`q6` (e.g. “1 (global)”, “2 (persona)”). To change them, edit the `question_labels` dictionary in `scripts/summarize_results.py`.
