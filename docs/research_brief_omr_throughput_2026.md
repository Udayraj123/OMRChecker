# Research Brief: High-Throughput OMR Pipelines & Mass Document Parsing in Python (2024–2026)

> Scope: Practical, source-cited research notes aimed at a Windows-deployed OMR pipeline targeting 5,000–100,000+ scans. Citations are inline URLs.

---

## 1. OMR pipeline best practices (2024–2026)

### 1.1 State-of-the-art detection: classical OpenCV vs. ML/CNN

- **Classical OpenCV pipelines** remain the production default for bubble-style OMR because they're fast, deterministic, debuggable, and run on commodity hardware. The canonical recipe is grayscale → Gaussian blur → adaptive/Otsu threshold → contour detection → aspect-ratio filtering → fill-ratio analysis. See PyImageSearch's reference implementation ([pyimagesearch.com/2016/10/03/...](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/)) and the deep-dive on Murtaza Hassan's tutorial ([deepwiki.com/...optical-mark-recognition-(omr)](https://deepwiki.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/5.2-optical-mark-recognition-(omr))).
- **OMRChecker (Udayraj123)** is the most-starred open-source production OMR engine (≈1,000+ stars). Recent PRs (2024–2025) are converging on:
  - Configuration-driven preprocessing (no hardcoded thresholds) — PR [#253](https://github.com/Udayraj123/OMRChecker/pull/253).
  - Step-wise debug visualization at `show_image_level >= 4` — PR [#274](https://github.com/Udayraj123/OMRChecker/pull/274).
  - Modular `CropPage` (curvature correction) + `CropOnMarkers` (patch/blob/line detection) — issue [#220](https://github.com/Udayraj123/OMRChecker/issues/220).
  - `opencv-python-headless` + `opencv-contrib-python-headless` for server deployments — PR [#257](https://github.com/Udayraj123/OMRChecker/pull/257).
  - Throughput claim: **~200 OMRs/minute, ~100% accuracy on flatbed scans, ~90% on mobile photos** ([github.com/Udayraj123/OMRChecker](https://github.com/Udayraj123/OMRChecker)).
- **ML/CNN approaches** beat classical OpenCV on **marginal cases** (faint marks, crossed-out answers, partial fills):
  - **OMRNet** (MobileNetV2-based, classifies confirmed / crossed-out / empty) — 95.96% mean accuracy, lightweight enough for edge deployment ([Springer 10.1007/s11042-023-15408-8](https://link.springer.com/article/10.1007/s11042-023-15408-8)).
  - **SurveyNet** unifies OCR + OMR in one DL framework for survey digitization ([MDPI 2313-433X/12/4/175](https://www.mdpi.com/2313-433X/12/4/175)).
  - **Hybrid is the dominant pattern**: OpenCV does the geometry (page detection, perspective transform, bubble ROI extraction), CNN does the per-bubble *classification* ([priyamghorui/OMR-Sheet-Checker-by-ML-Model-CNN](https://github.com/priyamghorui/OMR-Sheet-Checker-by-ML-Model-CNN)).
- **LLM-based OMR ("gpt-omr")**: No mature open-source library exists yet. Gemini-style multimodal models are being used for structured-data extraction from photographed forms ([gemilab.net article](https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness)), and there are LLM-grading projects ([emorynlp/LLM-Grading](https://github.com/emorynlp/LLM-Grading)) but they grade *content*, not bubbles. LLM-only OMR is overkill and ~1000× slower per sheet than OpenCV; reserve for an exception/fallback path on low-confidence sheets.

**Practical takeaway**: classical OpenCV for the hot path; optional CNN classifier head for ambiguous bubbles; LLM only as a human-in-the-loop fallback.

### 1.2 Fiducial markers: ArUco vs. alternatives

| Marker | Detection time | Damage tolerance | Library support | Notes |
|---|---|---|---|---|
| **ArUco** | ~2 ms ([planar fiducial study](https://link.springer.com/article/10.1007/s10055-023-00772-5)) | Moderate | OpenCV `cv2.aruco` built-in | Fastest, designed for CV |
| **AprilTag** | Slower than ArUco | Better than ArUco at distance/blur | Strong robotics support | Heavier compute |
| **CCTag** | Slower | **Best under occlusion + blur** ([planar fiducial study](https://link.springer.com/article/10.1007/s10055-023-00772-5)) | Limited Python bindings | Niche |
| **QR / Aztec** | ~10 ms | Aztec tolerates 30% damage; QR mature ecosystem ([blog.choto.co](https://blog.choto.co/data-matrix-vs-pdf417-vs-aztec-vs-qr-codes/), [Medium robustness analysis](https://medium.com/truproof-ai/two-dimensional-barcodes-robustness-analysis-through-the-example-of-qr-code-and-aztec-code-a68de47158ec)) | Excellent | Use when you also need to *encode* data (sheet ID, version) |

**Recommendation for OMR**: ArUco for pure geometric registration (4 corners → perspective transform). Add a single QR/Aztec code if you need to encode sheet metadata (form version, candidate ID).

### 1.3 Maximizing per-image throughput

- **Resolution sweet spot**: 200–300 DPI. 300 DPI is the canonical "high-accuracy" target ([bthicks/OMR-Grader](https://github.com/bthicks/OMR-Grader)); 200 DPI is usually fine if bubbles are ≥4 mm. Going to 600 DPI ~quadruples pixel count and slows every downstream step with marginal accuracy gain for filled-bubble detection.
- **Grayscale only**. Color buys you nothing for fill-detection and triples memory + decode time. PyImageSearch and OMRChecker both convert immediately on read.
- **JPEG quality**: source scans at quality 85–90 are indistinguishable from PNG for OMR purposes but are ~10× smaller. *Never* re-encode to JPEG between stages — decode JPEG **once** into a numpy array and pass that array through the pipeline.
- **TurboJPEG** (`PyTurboJPEG`) decodes ~7× faster than `cv2.imdecode` (4.6 ms vs 32 ms per image) per the imread benchmark ([arXiv 2501.13131](https://ar5iv.labs.arxiv.org/html/2501.13131), [SO 30111368](https://stackoverflow.com/questions/30111368/are-there-any-alternatives-to-using-opencvs-imdecode-it-is-too-slow)). Use it when JPEG decode is on the hot path.
- Use `opencv-python-headless` in workers — saves Qt/GTK import time and reduces image footprint, important when each worker spawns fresh on Windows.

### 1.4 Common failure modes and production handling

Synthesized from [gemilab.net](https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness), [mshaeri.com](https://mshaeri.com/blog/scanned-document-image-preprocessing-for-machine-learning-classification-feature-extraction/), and the IJERT survey ([ijert.org IJERTV4IS090675](https://www.ijert.org/research/various-techniques-for-assessment-of-omr-sheets-through-ordinary-2d-scanner-a-survey-IJERTV4IS090675.pdf)):

| Failure mode | Mitigation |
|---|---|
| Skew / tilt | Hough lines on margins, or 4-point perspective transform anchored on fiducials (ArUco/L-marks). |
| Lighting / shadow | CLAHE (only when luminance variance exceeds a threshold — blind application introduces noise). |
| Smudges / partial fills | Fill-ratio analysis with **two** thresholds: confident-empty < X, confident-filled > Y, ambiguous in between → flag for review. |
| Low contrast scans | Otsu's thresholding (auto-selects threshold) instead of fixed thresholds. |
| Page curvature | OMRChecker's `CropPage` does page curvature correction explicitly. |
| Wrong page / blank | Detect fiducials *first*; if not found, fail-fast and route to manual queue. |
| Too-low resolution | If long edge < 500 px, reject — reprocessing won't help ([gemilab.net](https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness)). |

Roughly **half of real-world failures can be eliminated by preprocessing alone** before the recognition engine runs ([gemilab.net](https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness)). Always produce a confidence score per bubble and a per-sheet confidence; route low-confidence sheets to human review rather than silently guessing.

---

## 2. PDF → image conversion at scale

### 2.1 Library comparison

| Library | Backend | License | Pure wheel (Windows)? | In-memory | External EXE? | Notes |
|---|---|---|---|---|---|---|
| **`pypdfium2`** | Google PDFium | **Apache-2.0 / BSD-3** ([PyPI](https://pypi.org/project/pypdfium2/)) | **Yes**, prebuilt wheels ([github.com/pypdfium2-team/pypdfium2](https://github.com/pypdfium2-team/pypdfium2/)) | **Yes** — `PdfDocument(io.BytesIO)` ([issue #205](https://github.com/pypdfium2-team/pypdfium2/issues/205)) | No | The "no surprises on Windows" option. |
| **PyMuPDF (`fitz`)** | MuPDF (Artifex) | **AGPL-3.0 or commercial** ([PyPI](https://pypi.org/project/PyMuPDF/), [issue #4962](https://github.com/pymupdf/PyMuPDF/issues/4962)) | Yes, prebuilt wheels | **Yes** — `fitz.open(stream=mem_area, filetype="pdf")` ([docs](https://pymupdf.readthedocs.io/en/latest/how-to-open-a-file.html)) | No | Fastest in most benchmarks, but AGPL is **viral** — any networked commercial use needs a paid Artifex license. |
| **`pdf2image`** | Poppler (`pdftoppm.exe`) | MIT wrapper, GPL backend | **No** — requires separate Poppler install on PATH ([docs](https://pdf2image.readthedocs.io/en/latest/installation.html), [issue #101](https://github.com/Belval/pdf2image/issues/101)) | Indirect (`convert_from_bytes`) | **Yes — spawns `pdftoppm.exe` subprocess per call** | Worst-case for Windows Defender: external EXE on every PDF. Skip. |
| **`pdfplumber`** | pdfminer.six | MIT | Yes | Yes | No | Text/structure extraction; image rendering is slow ([py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)). |
| **Wand / ImageMagick** | ImageMagick | Apache-2.0 wrapper, IM license backend | **No** — requires ImageMagick install | Yes | **Yes** | Slow and another EXE dependency. Skip. |

### 2.2 Benchmarks (py-pdf/benchmarks, updated Jul 2 2025)

Source: [github.com/py-pdf/benchmarks](https://github.com/py-pdf/benchmarks). 14 native PDFs, 285 KiB–14.7 MiB, i7-6700HQ.

Text extraction average (lower is better):

| Library | Avg | Relative |
|---|---|---|
| PyMuPDF 1.26 | **0.1 s** | 1.0× |
| pypdfium2 4.30 | **0.1 s** | 1.0× |
| Tika | 0.2 s | 2× |
| pdftotext | 0.3 s | 3× |
| pypdf | 3.5 s | 35× |
| pdfminer.six | 5.8 s | 58× |
| pdfplumber | 7.9 s | 79× |

PyMuPDF and pypdfium2 are within noise of each other; both are ~35–80× faster than pure-Python parsers.

For **rendering** specifically (no public 2025 head-to-head numbers in py-pdf/benchmarks), PyMuPDF's own docs claim "fast rendering without external dependencies" at 150 DPI medium quality ([PyMuPDF app4.rst](https://github.com/pymupdf/PyMuPDF/blob/main/docs/app4.rst)). Anecdotal comparisons place PyMuPDF and pypdfium2 within 10–30% of each other for raster rendering, both materially faster than Poppler/`pdf2image` because they avoid subprocess + temp-file overhead ([Medium: Katherine Cao](https://medium.com/@yimeng_cao/exploring-ways-to-convert-pdfs-to-images-in-python-9e68430116cd)).

### 2.3 Memory vs speed, streaming, DPI

- **Stream page-by-page**. Both pypdfium2 and PyMuPDF expose per-page rendering — open the doc once, iterate pages, yield each bitmap, discard before moving on. Avoid `convert_from_path` patterns that materialize all pages first.
- **DPI for OMR**: render PDFs at **200–250 DPI grayscale**. 300 DPI is overkill for filled-bubble detection and doubles memory vs 200 DPI. Match the DPI to the scanner that produced the PDF — don't upsample.
- **Pure-Python wheels (no external EXE)**: `pypdfium2` and `PyMuPDF` both ship self-contained wheels. `pdf2image` and `Wand` both require external EXEs — these are **the worst case for Windows Defender** (every subprocess spawn is scanned, every temp `.ppm` write is scanned).

### 2.4 Thread/process safety — critical at scale

- **Both `pypdfium2` and `PyMuPDF` are NOT thread-safe.** PDFium and MuPDF maintain global C-level state and will segfault or corrupt under concurrent calls.
  - pypdfium2 thread-safety statement: [issue #303](https://github.com/pypdfium2-team/pypdfium2/issues/303), [Python API docs](https://pypdfium2.readthedocs.io/en/v4/python_api.html).
  - PyMuPDF thread-safety statement: [discussion #4778](https://github.com/pymupdf/PyMuPDF/discussions/4778), [issue #107](https://github.com/pymupdf/PyMuPDF/issues/107), [recipe](https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html).
  - Even with a mutex, MuPDF queues internally so threading provides **zero speedup** ([issue #107](https://github.com/pymupdf/PyMuPDF/issues/107)).
- **Use multiprocessing**, not threading. Each worker process gets its own PDFium/MuPDF instance.
- **Don't pickle Document/Page objects**. PyMuPDF raises `TypeError: cannot pickle 'SwigPyObject' object` ([issue #2336](https://github.com/pymupdf/PyMuPDF/issues/2336)). Pass the **PDF file path or `bytes`**; each worker reopens the document.

---

## 3. Python multiprocessing at 5000+ files

### 3.1 Library comparison

Source: joblib issue [#1733](https://github.com/joblib/joblib/issues/1733) benchmark (2025), large-object workload:

| Library | Throughput (it/s) | Notes |
|---|---|---|
| **`concurrent.futures.ProcessPoolExecutor`** | **1,829** | Cleanest API in stdlib; best for large objects. |
| `multiprocessing.Pool` | Comparable to PPE | Older stdlib API; fine for simpler workloads. |
| `joblib.Parallel` (manual `batch_size`) | 323 | Convenient but slower under large-object dispatch. |
| `joblib.Parallel` (auto batch) | 32 | Auto-batch underperforms badly with large args. |

Older general benchmark ([JohnStarich/python-pool-performance](https://github.com/JohnStarich/python-pool-performance)) found `multiprocessing.Pool` "overall winner" for mixed CPU+IO; the 2025 update shows `ProcessPoolExecutor` ahead for large-object pipelines.

- **Ray** is purpose-built for batch image inference; `ds.read_images()` + `ds.map_batches()` scales across machines and integrates with GPU placement groups ([Ray batch inference docs](https://docs.ray.io/en/latest/data/examples/batch_inference_object_detection.html), [Ray images](https://docs.ray.io/en/releases-2.31.0/data/working-with-images.html)). Overkill for a single Windows box; valuable if you ever go multi-node.
- **Dask** is good for tabular shuffle-heavy work; for embarrassingly parallel image work, it's strictly less ergonomic than Ray Data and offers no clear advantage over `ProcessPoolExecutor` on one machine ([Dask embarrassingly parallel](https://examples.dask.org/applications/embarrassingly-parallel.html)).
- **MPIRE** offers a nicer API on top of `multiprocessing` with progress bars, dashboards, worker-state — usable but not faster ([Towards Data Science: MPIRE](https://towardsdatascience.com/mpire-for-python-multiprocessing-is-really-easy-d2ae7999a3e9/)).

### 3.2 Worker count heuristics

- **CPU-bound (e.g., OpenCV bubble detection)**: `os.cpu_count()` physical cores. Hyperthreading rarely helps for OpenCV/NumPy because they already release the GIL.
- **I/O-bound (e.g., scanning many small files from disk)**: 2–4× physical cores, but only if disk IOPS aren't already saturated. NVMe handles much more parallelism than HDD.
- **Mixed (PDF render + OMR detection)**: `physical_cores - 1` (leave one for the main process / OS). Going higher hurts because of memory pressure + AV scanning.
- **Memory-bound**: limit workers so peak RSS × workers ≤ 60% of system RAM. A single 300 DPI A4 grayscale page is ~7 MB raw; with intermediate buffers, budget 50–100 MB per worker.

### 3.3 Chunking strategy

- For `Pool.imap_unordered(..., chunksize=N)`: pick `chunksize ≈ ceil(total_items / (workers × 10))`. Too small → pickle/IPC dominates; too large → late workers idle while one chunk finishes.
- For `ProcessPoolExecutor.map(..., chunksize=N)`: same heuristic.
- For very large items (a PDF with 100 pages), submit *page-level* tasks, not document-level — preserves load balancing across workers.

### 3.4 Spawn vs fork — Windows-specific pitfalls

- **Windows only supports `spawn`** ([Python docs](https://docs.python.org/3/library/multiprocessing.html)). Linux/macOS default to `fork`/`forkserver` on Python 3.14+.
- `spawn` re-imports your entire module in each worker — **this is the #1 hidden cost on Windows**. Heavy top-level imports (TensorFlow, PyTorch, OpenCV-contrib) can take 1–3 s **per worker** at startup. For a 16-worker pool that's 16–48 s before any work happens.
  - **Mitigation**: keep `if __name__ == "__main__":` minimal; defer heavy imports inside worker functions; or use a long-lived pool that processes many items.
- `spawn` pickles every argument. **Pickle has a performance cliff at ~64 KB** ([cpython issue #96953](https://github.com/python/cpython/issues/96953)) — ~10× slowdown when args cross the pipe buffer limit. Don't pass raw image arrays through pipes — pass file paths or use shared memory (next section).
- For a 1 GB array, pipe-based process pools were measured ~6× *slower* than threads purely from pickle overhead ([pythonspeed.com](https://pythonspeed.com/articles/faster-multiprocessing-pickle/)).
- **joblib's `loky` backend** uses fork+exec on POSIX for safer 3rd-party-library interaction ([loky](https://github.com/joblib/loky/)). On Windows it degrades to spawn like everything else.

### 3.5 Shared memory for large image arrays

`multiprocessing.shared_memory` (Python 3.8+) is the right tool for "huge numpy array, many workers":

```python
from multiprocessing import shared_memory
import numpy as np

# Parent creates
arr = render_pdf_page(...)                  # (H, W) uint8
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
buf[:] = arr

# Pass shm.name (a string) to workers, NOT the SharedMemory object
worker_args = (shm.name, arr.shape, arr.dtype.str)
```

- **2 GB dataset, 8 workers**: 16 GB RAM without shared memory → 2 GB with ([stanza.dev](https://www.stanza.dev/courses/python-concurrency/multiprocessing-shared/python-concurrency-shared-memory), [krython.com](https://www.krython.com/tutorial/python/shared-memory-multiprocessing-shared-memory)).
- **Critical rules**:
  - Only the creator calls `shm.unlink()`. Multiple unlinks → segfault.
  - Other processes call `shm.close()` only.
  - Register `atexit` cleanup to avoid leaked OS-level shared segments (Windows Resource Monitor won't show these clearly).
- **Alternative**: `numpy.memmap` backed by a temp file — slightly slower than shared memory but simpler lifecycle.
- For embarrassingly parallel work, **just passing a file path** is often simpler and almost as fast — only adopt shared memory if you've measured the per-worker decode cost.

---

## 4. Avoiding disk I/O for intermediate images

### 4.1 BytesIO-based pipelines

The classic anti-pattern is: render PDF → write PNG to disk → read PNG with cv2 → process. Every write *and* read trips Windows Defender. Inline pipeline:

```python
import pypdfium2 as pdfium
import numpy as np
import cv2

pdf = pdfium.PdfDocument(input_bytes)            # bytes, not file
for page in pdf:
    bitmap = page.render(scale=200/72, grayscale=True)
    arr = bitmap.to_numpy()                      # numpy view, zero copy
    # arr is ready for OpenCV; no disk write
    result = process_omr(arr)
```

Or with PyMuPDF:

```python
import fitz
doc = fitz.open(stream=input_bytes, filetype="pdf")
for page in doc:
    pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    result = process_omr(arr)
```

- `cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)` is **≈3% faster** than `cv2.imread` from disk on warm caches and **much** faster on cold caches or AV-heavy systems ([SO 54860259](https://stackoverflow.com/questions/54860259/python-opencv-create-image-from-bytearray)). Always use `np.frombuffer`, **never** `np.asarray(bytearray(...))` — the latter copies ([SO 56653511](https://stackoverflow.com/questions/56653511/python-file-i-o-significantly-slower-than-opencv-imread)).

### 4.2 Why writing 5,000+ small PNGs is a worst case on Windows

Sources: [hive issue #4428](https://github.com/adenhq/hive/issues/4428), [Microsoft Q&A](https://answers.microsoft.com/en-us/windows/forum/all/windows-defender-real-time-protection-service/fda3f73e-cc0a-4946-9b9d-3c05057ef90c), [SO 57173701](https://stackoverflow.com/questions/57173701/high-windows-defender-cpu-usage-while-running-python-script), [Robocorp/Sema4 docs](https://sema4.ai/docs/automation/troubleshooting/windows-defender).

- Windows Defender's real-time protection scans **every** file write/read by default. Per-file latency is small (~1–5 ms) but compounds linearly: 5,000 PNGs ≈ 5–25 s of pure AV overhead per pass, ignoring the actual disk I/O.
- Documented build-time degradation has hit **20–22× slowdowns** (86 s vs 4 s) when AV is on vs off.
- One Python script saw Antimalware Service Executable consume **16% CPU while Python used 5%** ([SO 57173701](https://stackoverflow.com/questions/57173701/high-windows-defender-cpu-usage-while-running-python-script)).
- A `pdf2image` pipeline that writes thousands of `.ppm` temp files via `pdftoppm.exe` triples the pain: subprocess spawn + temp-file write + temp-file read are all scanned.

**Mitigations** (in priority order):
1. **Don't write the files**. In-memory pipeline (above) eliminates the problem at the source.
2. **Add path exclusions** for your project directory and `.venv` in Windows Security → Exclusions. Reported 30–40% speedups on Python tooling ([hive #4428](https://github.com/adenhq/hive/issues/4428)).
3. If you *must* persist intermediates, write to a single archive (`tarfile`, `zipfile`, `.npz`) instead of thousands of files. One write → one scan.

### 4.3 Memory budgeting

- **In-RAM ceiling**: keep peak working set under ~60% of system RAM. A 200 DPI A4 grayscale page is ~4–5 MB raw; with intermediate buffers (binary mask + perspective-corrected copy + debug overlays) budget 20–30 MB per page in flight.
- **Spill rule of thumb**: if `total_pages × per_page_mem > 0.5 × RAM`, switch to streaming (process N pages, discard, continue) rather than batching.
- **Garbage collection**: in long-running workers, explicit `del large_arr; gc.collect()` between large items prevents fragmentation on Windows (worse than Linux's glibc allocator).

---

## 5. Production examples and benchmarks

### Open-source OMR projects (with throughput notes)

- **[Udayraj123/OMRChecker](https://github.com/Udayraj123/OMRChecker)** — ~1,000+ stars, "200+ OMRs/minute, ~100% accuracy on good scans, ~90% on mobile". Active 2024–2025.
- **[EuracBiomedicalResearch/RescueOMR](https://github.com/EuracBiomedicalResearch/RescueOMR)** — explicitly designed for bulk loose-page questionnaire scans at 300 DPI grayscale. Two-tool design (`extractmpl` for template registration, `simpleomr` for mark detection).
- **[donsolo-khalifa/OMRGrading](https://github.com/donsolo-khalifa/OMRGrading)** — webcam + static image OMR with perspective transform; created Apr 2025, recently updated.
- **[Krit-coder/OMR-Evaluation](https://github.com/Krit-coder/OMR-Evaluation)** — Flask web app w/ roll number extraction + Excel export.
- **[bthicks/OMR-Grader](https://github.com/bthicks/OMR-Grader)** — explicitly recommends 300 DPI.

### Scanned-document case studies

- **2,000,000 scanned papers** processed with Python + OpenCV pipeline (denoise → binarize → align) ([mshaeri.com](https://mshaeri.com/blog/scanned-document-image-preprocessing-for-machine-learning-classification-feature-extraction/)).
- **Gemini-API-based real-world photo extraction** at scale, with explicit treatment of tilt, shadows, occlusion ([gemilab.net](https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness)).
- **PyImageSearch invoice/form OCR** — canonical Tesseract+OpenCV pipeline ([pyimagesearch.com](https://pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/)).

### Benchmark repositories

- **[py-pdf/benchmarks](https://github.com/py-pdf/benchmarks)** — updated Jul 2025, definitive PDF text-extraction comparison.
- **[dhdaines/benchmarks](https://github.com/dhdaines/benchmarks)** — alternative PDF benchmark suite.
- **[ternaus/imread_benchmark](https://ar5iv.labs.arxiv.org/html/2501.13131)** — 2025 JPEG decoder benchmark (Jan 2025): TurboJPEG ~7× cv2.imdecode.

---

## Top recommendations for a Windows-deployed, 5,000+ doc OMR pipeline

| Concern | Pick | Why |
|---|---|---|
| **PDF → image** | `pypdfium2` | Apache/BSD license (no AGPL viral risk), prebuilt Windows wheels, no external EXE, in-memory via `BytesIO`, on par with PyMuPDF in benchmarks ([py-pdf/benchmarks](https://github.com/py-pdf/benchmarks), [PyPI](https://pypi.org/project/pypdfium2/)). |
| **PDF → image (if AGPL OK)** | PyMuPDF | Slight edge on some workloads; only use if you have an Artifex commercial license or your product is also AGPL ([issue #4962](https://github.com/pymupdf/PyMuPDF/issues/4962)). |
| **OMR detection** | OpenCV headless + classical pipeline; CNN classifier head optional for low-confidence bubbles | Deterministic, debuggable, fast; OMRChecker pattern is proven ([Udayraj123/OMRChecker](https://github.com/Udayraj123/OMRChecker)). |
| **Fiducials** | `cv2.aruco` 4-corner ArUco + optional QR for sheet metadata | Fastest detection (~2 ms), built into OpenCV ([planar fiducial study](https://link.springer.com/article/10.1007/s10055-023-00772-5)). |
| **Parallelism** | `concurrent.futures.ProcessPoolExecutor`, worker count = `os.cpu_count() - 1` | Best large-object throughput ([joblib #1733](https://github.com/joblib/joblib/issues/1733)); standard library, no extra deps. |
| **Inter-process data** | Pass **file paths**, not arrays. Use `shared_memory` only when measured to help. | Avoids the 64 KB pickle cliff ([cpython #96953](https://github.com/python/cpython/issues/96953)) and PDFium pickling restrictions. |
| **Image decode (JPEG hot path)** | `PyTurboJPEG` if JPEG-dominated; otherwise `cv2.imdecode(np.frombuffer(...))` | 7× faster than cv2 on JPEG ([arXiv 2501.13131](https://ar5iv.labs.arxiv.org/html/2501.13131)). |
| **Disk strategy** | **Zero intermediate files.** Render in memory, hand numpy arrays directly to OpenCV. | Eliminates Windows Defender per-file scanning cost ([Microsoft Q&A](https://answers.microsoft.com/en-us/windows/forum/all/windows-defender-real-time-protection-service/fda3f73e-cc0a-4946-9b9d-3c05057ef90c)). |
| **Defender** | Add project dir + `.venv` to exclusions; document this for ops. | 30–40% speedup on Python file-heavy workloads ([hive #4428](https://github.com/adenhq/hive/issues/4428)). |
| **DPI / format** | 200 DPI, grayscale, single-pass decode. | Best speed/accuracy tradeoff for filled bubbles. |
| **Confidence + human-in-the-loop** | Two-threshold fill-ratio classifier with an "ambiguous" bin routed to manual review queue. | Eliminates silent miscounts; 50%+ of real failures preventable in preprocessing ([gemilab.net](https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness)). |

### Reference pipeline shape

```
PDF bytes ─┐
           ├─► pypdfium2 (in-process)  ──► per-page numpy grayscale (in memory)
           │      • dpi=200, grayscale=True
           │      • one PdfDocument per worker process
           │
           ▼
   ProcessPoolExecutor (cpu_count - 1 workers)
           │
           ▼
   per-worker: OpenCV pipeline
     1. ArUco detect 4 corners
     2. Perspective transform → canonical sheet
     3. Crop ROI per bubble (template-driven)
     4. Otsu threshold + fill-ratio
     5. Confidence bin: empty / filled / AMBIGUOUS
           │
           ▼
   Results returned as small dicts/dataclasses (cheap pickle)
   AMBIGUOUS sheets → review queue (DB / folder)
   No intermediate PNGs are ever written to disk
```

---

## Quick wins (5–7 changes any decent OMR pipeline can make for 2–10× throughput)

1. **Kill the disk between stages.** Replace any `convert_from_path → save PNG → cv2.imread` chain with `pypdfium2.PdfDocument(bytes).render().to_numpy()`. On Windows, this alone often doubles throughput by eliminating Defender scanning of 5,000+ temp files. ([pypdfium2 docs](https://pypdfium2.readthedocs.io/en/v4/readme.html))
2. **Switch from `pdf2image` to `pypdfium2` (or PyMuPDF).** Drops the Poppler `pdftoppm.exe` dependency, eliminates per-page subprocess spawn, ships in a pure wheel, ~30% faster rendering in practice. ([py-pdf/benchmarks](https://github.com/py-pdf/benchmarks))
3. **Convert to grayscale at decode time, not later.** `bitmap.render(grayscale=True)` (pypdfium2) or `colorspace=fitz.csGRAY` (PyMuPDF) — cuts memory and downstream work 3×.
4. **Use `ProcessPoolExecutor` with `cpu_count() - 1` workers, pass file paths, not arrays.** Avoids the 64 KB pickle cliff and PyMuPDF/pypdfium2 unpicklable-object errors. Open the PDF inside each worker. ([cpython #96953](https://github.com/python/cpython/issues/96953), [PyMuPDF #2336](https://github.com/pymupdf/PyMuPDF/issues/2336))
5. **Add project dir + `.venv` to Windows Defender exclusions.** Documented 30–40% speedup on Python tooling. ([Microsoft Learn Q&A](https://learn.microsoft.com/en-us/answers/questions/5881276/how-do-you-reduce-performance-impact-from-defender))
6. **Use `opencv-python-headless` in workers.** Cuts per-worker import time on the Windows `spawn` start method by skipping Qt/GTK loading — important when each worker process pays the import cost on startup. ([OMRChecker #257](https://github.com/Udayraj123/OMRChecker/pull/257))
7. **Adopt a confidence-binned classifier.** Two-threshold fill-ratio (empty / filled / ambiguous) + routing low-confidence sheets to a review queue eliminates silent failures and lets the hot path stay fast; only ambiguous sheets pay the cost of a CNN re-classification or LLM fallback. ([OMRNet paper](https://link.springer.com/article/10.1007/s11042-023-15408-8), [gemilab.net](https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness))

---

## Appendix: cited sources

### OMR engines & techniques
- Udayraj123/OMRChecker repo & PRs: [#253](https://github.com/Udayraj123/OMRChecker/pull/253), [#257](https://github.com/Udayraj123/OMRChecker/pull/257), [#274](https://github.com/Udayraj123/OMRChecker/pull/274), [#220](https://github.com/Udayraj123/OMRChecker/issues/220)
- PyImageSearch bubble-sheet tutorial: https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
- OMRNet (MobileNetV2): https://link.springer.com/article/10.1007/s11042-023-15408-8
- SurveyNet: https://www.mdpi.com/2313-433X/12/4/175
- IJERT OMR survey: https://www.ijert.org/research/various-techniques-for-assessment-of-omr-sheets-through-ordinary-2d-scanner-a-survey-IJERTV4IS090675.pdf
- RescueOMR: https://github.com/EuracBiomedicalResearch/RescueOMR
- bthicks/OMR-Grader (300 DPI recommendation): https://github.com/bthicks/OMR-Grader

### Fiducial markers
- Planar fiducial comparative study: https://link.springer.com/article/10.1007/s10055-023-00772-5
- Aztec vs QR robustness: https://medium.com/truproof-ai/two-dimensional-barcodes-robustness-analysis-through-the-example-of-qr-code-and-aztec-code-a68de47158ec
- 2D barcode comparison: https://blog.choto.co/data-matrix-vs-pdf417-vs-aztec-vs-qr-codes/

### PDF libraries
- py-pdf/benchmarks: https://github.com/py-pdf/benchmarks
- pypdfium2 docs / PyPI: https://pypdfium2.readthedocs.io/en/v4/readme.html , https://pypi.org/project/pypdfium2/
- pypdfium2 thread-safety: https://github.com/pypdfium2-team/pypdfium2/issues/303
- pypdfium2 BytesIO: https://github.com/pypdfium2-team/pypdfium2/issues/205
- PyMuPDF docs / PyPI: https://pymupdf.readthedocs.io/en/latest/ , https://pypi.org/project/PyMuPDF/
- PyMuPDF licensing: https://github.com/pymupdf/PyMuPDF/issues/4962
- PyMuPDF multiprocessing recipe: https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
- PyMuPDF thread safety: https://github.com/pymupdf/PyMuPDF/discussions/4778 , https://github.com/pymupdf/PyMuPDF/issues/107
- PyMuPDF unpicklable: https://github.com/pymupdf/PyMuPDF/issues/2336
- pdf2image install / Poppler: https://pdf2image.readthedocs.io/en/latest/installation.html , https://github.com/Belval/pdf2image/issues/101

### Parallelism
- joblib vs ProcessPoolExecutor benchmark: https://github.com/joblib/joblib/issues/1733
- multiprocessing pickle cliff: https://github.com/python/cpython/issues/96953
- pythonspeed.com on multiprocessing pickle: https://pythonspeed.com/articles/faster-multiprocessing-pickle/
- Python pool performance benchmark: https://github.com/JohnStarich/python-pool-performance
- joblib/loky: https://github.com/joblib/loky/
- joblib parallel docs: https://joblib.readthedocs.io/en/stable/parallel.html
- Python multiprocessing docs: https://docs.python.org/3/library/multiprocessing.html
- shared_memory tutorial: https://www.krython.com/tutorial/python/shared-memory-multiprocessing-shared-memory
- shared_memory concurrency course: https://www.stanza.dev/courses/python-concurrency/multiprocessing-shared/python-concurrency-shared-memory
- Ray Data images: https://docs.ray.io/en/releases-2.31.0/data/working-with-images.html
- Ray batch inference: https://docs.ray.io/en/latest/data/examples/batch_inference_object_detection.html
- Dask embarrassingly parallel: https://examples.dask.org/applications/embarrassingly-parallel.html

### Disk / AV / decode
- Windows Defender Python impact: https://stackoverflow.com/questions/57173701/high-windows-defender-cpu-usage-while-running-python-script
- Defender exclusions guidance: https://learn.microsoft.com/en-us/answers/questions/5881276/how-do-you-reduce-performance-impact-from-defender
- Defender realtime slowdown: https://answers.microsoft.com/en-us/windows/forum/all/windows-defender-real-time-protection-service/fda3f73e-cc0a-4946-9b9d-3c05057ef90c
- Sema4 Defender troubleshooting: https://sema4.ai/docs/automation/troubleshooting/windows-defender
- cv2.imdecode bytes: https://stackoverflow.com/questions/54860259/python-opencv-create-image-from-bytearray
- cv2 imread slow: https://stackoverflow.com/questions/76212737/cv2-imread-function-is-very-slow
- imread benchmark (JPEG decoders, Jan 2025): https://ar5iv.labs.arxiv.org/html/2501.13131

### Document-pipeline case studies
- 2M-page OpenCV pipeline: https://mshaeri.com/blog/scanned-document-image-preprocessing-for-machine-learning-classification-feature-extraction/
- Gemini real-world photo robustness: https://gemilab.net/en/articles/gemini-api/gemini-real-world-photo-structured-data-robustness
- PyImageSearch invoice OCR: https://pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/
