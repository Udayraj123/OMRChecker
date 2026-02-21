# File Utils Flows

**Module**: Domain - Utils - File
**Python Reference**: `src/utils/file.py`
**Last Updated**: 2026-02-21

---

## Overview

File Utils provides three main components:

1. **JSON Loading**: Safe JSON file loading with error handling
2. **PathUtils**: Path normalization and output directory management
3. **SaveImageOps**: Debug image stack generation and saving

**Use Case**: Cross-platform file operations, directory structure creation, debug visualization output

---

## Component 1: JSON Loading

### load_json() Flow

```
START: load_json(path, **rest)
│
├─► STEP 1: Check File Existence
│   │
│   if not Path(path).exists():
│   │
│   └─► RAISE InputFileNotFoundError(Path(path), "JSON")
│       END (exception)
│
├─► STEP 2: Open and Parse JSON
│   │
│   with Path.open(path) as f:
│   │   loaded = json.load(f, **rest)
│   │
│   ├─ Success: Continue to Step 3
│   │
│   └─ On JSONDecodeError:
│       │
│       ├─► Log critical error: "Error when loading json file at: '{path}'"
│       └─► RAISE ConfigLoadError(Path(path), f"Invalid JSON format: {error}")
│           END (exception)
│
└─► STEP 3: Return Parsed Data
    │
    RETURN loaded  # dict[str, Any]

END
```

### Parameters

**path**:
- Type: `str` or `Path`
- Path to JSON file

**rest**:
- Type: `dict` (keyword arguments)
- Passed to `json.load()` (e.g., `object_hook`, `parse_float`)

### Return Value

**Type**: `dict[str, Any]`
**Content**: Parsed JSON object (must be dict at root)

### Example Usage

```python
# Basic usage
template_data = load_json("template.json")

# With custom parser
config_data = load_json(
    "config.json",
    parse_float=decimal.Decimal  # Custom float parsing
)
```

### Error Scenarios

**Scenario 1: File Not Found**
```
Input: path = "missing.json"

Flow:
├─► Path("missing.json").exists() → False
└─► RAISE InputFileNotFoundError(Path("missing.json"), "JSON")

Exception:
InputFileNotFoundError: JSON file not found at: missing.json
```

**Scenario 2: Invalid JSON**
```
Input: path = "malformed.json"
Content: { "key": value }  # Missing quotes around value

Flow:
├─► File exists → Continue
├─► json.load() → JSONDecodeError
├─► Log critical error
└─► RAISE ConfigLoadError(Path("malformed.json"), "Invalid JSON format: ...")

Exception:
ConfigLoadError: Invalid JSON format: Expecting value: line 1 column 10 (char 9)
```

**Scenario 3: Non-UTF8 Characters**
```
Input: File with non-UTF8 encoding

Current: No explicit handling (relies on default encoding)
TODO: See comment in code about non-utf character handling
May raise UnicodeDecodeError if encoding is not UTF-8
```

---

## Component 2: PathUtils - Static Methods

### Method 1: remove_non_utf_characters()

```
START: PathUtils.remove_non_utf_characters(path_string)
│
├─► STEP 1: Filter Characters
│   │
│   result = "".join(
│       x for x in path_string
│       if x in PathUtils.printable_chars
│   )
│   │
│   │ printable_chars = set(string.printable)
│   │ Includes: [a-zA-Z0-9] + punctuation + whitespace
│
└─► STEP 2: Return Cleaned String
    │
    RETURN result

END
```

**Purpose**: Remove non-printable/non-ASCII characters from paths

**Printable Characters**:
- Defined as `string.printable` (100 characters)
- Letters: `a-z`, `A-Z`
- Digits: `0-9`
- Punctuation: `!"#$%&'()*+,-./:;<=>?@[\]^_` `` `{|}~`
- Whitespace: space, tab, newline, etc.

**Example**:
```python
input = "test\x00file\u2019.txt"  # Contains null byte and fancy apostrophe
output = PathUtils.remove_non_utf_characters(input)
# output = "testfile.txt"
```

### Method 2: sep_based_posix_path()

```
START: PathUtils.sep_based_posix_path(path)
│
├─► STEP 1: Normalize Path
│   │
│   path = os.path.normpath(path)
│   │
│   │ Purpose: Collapse redundant separators, resolve ".."
│   │ Examples:
│   │   "a//b" → "a/b" (Unix)
│   │   "a\\b" → "a\b" (Windows)
│
├─► STEP 2: Check for Windows Separators
│   │
│   if os.path.sep == "\\" or "\\" in path:
│   │
│   ├─ TRUE: Convert to POSIX
│   │   │
│   │   path = PureWindowsPath(path).as_posix()
│   │   │
│   │   │ Examples:
│   │   │   "C:\\Users\\test.txt" → "C:/Users/test.txt"
│   │   │   "folder\\file.txt" → "folder/file.txt"
│   │
│   └─ FALSE: Path is already POSIX-style
│       Continue to Step 3
│
└─► STEP 3: Remove Non-UTF Characters
    │
    path = PathUtils.remove_non_utf_characters(path)
    │
    RETURN path

END
```

**Purpose**: Convert any path to POSIX-style (forward slashes) and clean

**Use Cases**:
1. Cross-platform path serialization (to JSON)
2. Consistent path storage
3. Path display in UI

**Examples**:

```python
# Windows → POSIX
input = "C:\\OMRChecker\\outputs\\file.jpg"
output = PathUtils.sep_based_posix_path(input)
# output = "C:/OMRChecker/outputs/file.jpg"

# Unix (no change)
input = "/home/user/omr/file.jpg"
output = PathUtils.sep_based_posix_path(input)
# output = "/home/user/omr/file.jpg"

# Mixed separators
input = "folder\\subfolder/file.txt"
output = PathUtils.sep_based_posix_path(input)
# output = "folder/subfolder/file.txt"

# With non-UTF characters
input = "folder\x00\\file\u2019.txt"
output = PathUtils.sep_based_posix_path(input)
# output = "folder/file.txt"
```

---

## Component 3: PathUtils - Directory Management

### Instance Initialization

```
START: PathUtils.__init__(output_dir: Path)
│
├─► STEP 1: Store Output Directory
│   │
│   self.output_dir = output_dir
│
├─► STEP 2: Define Standard Subdirectories
│   │
│   self.save_marked_dir = output_dir / "CheckedOMRs"
│   self.image_metrics_dir = output_dir / "ImageMetrics"
│   self.results_dir = output_dir / "Results"
│   self.manual_dir = output_dir / "Manual"
│   │
│   self.errors_dir = self.manual_dir / "ErrorFiles"
│   self.multi_marked_dir = self.manual_dir / "MultiMarkedFiles"
│   │
│   self.evaluations_dir = output_dir / "Evaluations"
│   self.debug_dir = output_dir / "Debug"
│
└─► STEP 3: Return Instance
    │
    RETURN PathUtils instance

END
```

**Directory Structure Created**:
```
output_dir/
├── CheckedOMRs/          (save_marked_dir)
│   ├── colored/
│   ├── stack/
│   │   └── colored/
│   └── _MULTI_/
│       └── colored/
├── ImageMetrics/         (image_metrics_dir)
├── Results/              (results_dir)
├── Manual/               (manual_dir)
│   ├── ErrorFiles/       (errors_dir)
│   │   └── colored/
│   └── MultiMarkedFiles/ (multi_marked_dir)
│       └── colored/
├── Evaluations/          (evaluations_dir)
└── Debug/                (debug_dir)
```

### create_output_directories() Flow

```
START: PathUtils.create_output_directories()
│
├─► STEP 1: Log Start
│   │
│   logger.info("Checking Directories...")
│
├─► STEP 2: Create Main Marked Directory
│   │
│   if not self.save_marked_dir.exists():
│   │
│   └─► Path.mkdir(self.save_marked_dir, parents=True)
│
├─► STEP 3: Create Marked Subdirectories
│   │
│   For each directory in [
│       save_marked_dir / "colored",
│       save_marked_dir / "stack",
│       save_marked_dir / "stack" / "colored",
│       save_marked_dir / "_MULTI_",
│       save_marked_dir / "_MULTI_" / "colored"
│   ]:
│   │
│   ├─► if not directory.exists():
│   │   │
│   │   ├─► Path.mkdir(directory)
│   │   └─► logger.info(f"Created : {directory}")
│
├─► STEP 4: Create Image Bucket Directories
│   │
│   For each directory in [
│       manual_dir,
│       multi_marked_dir,
│       errors_dir
│   ]:
│   │
│   ├─► if not directory.exists():
│   │   │
│   │   ├─► logger.info(f"Created : {directory}")
│   │   ├─► Path.mkdir(directory, parents=True)
│   │   └─► Path.mkdir(directory / "colored")
│
└─► STEP 5: Create Non-Image Directories
    │
    For each directory in [
        results_dir,
        image_metrics_dir,
        evaluations_dir
    ]:
    │
    ├─► if not directory.exists():
    │   │
    │   ├─► logger.info(f"Created : {directory}")
    │   └─► Path.mkdir(directory, parents=True)

END
```

**Purpose**: Idempotently create all required output directories

**Behavior**:
- Creates directories only if they don't exist
- Uses `parents=True` to create parent directories
- Logs each created directory
- Safe to call multiple times

**Example Log Output**:
```
Checking Directories...
Created : /output/CheckedOMRs
Created : /output/CheckedOMRs/colored
Created : /output/CheckedOMRs/stack
Created : /output/CheckedOMRs/stack/colored
Created : /output/Manual
Created : /output/Manual/ErrorFiles
Created : /output/Manual/ErrorFiles/colored
Created : /output/Results
```

---

## Component 4: SaveImageOps - Debug Image Management

### Initialization

```
START: SaveImageOps.__init__(tuning_config)
│
├─► STEP 1: Initialize Image Storage
│   │
│   self.gray_images = defaultdict(list)
│   self.colored_images = defaultdict(list)
│   │
│   │ Structure: dict[int, list[tuple[str, np.ndarray]]]
│   │ Keys: Save image levels (1-6)
│   │ Values: List of (title, image) tuples
│
├─► STEP 2: Store Config
│   │
│   self.tuning_config = tuning_config
│   self.save_image_level = tuning_config.outputs.save_image_level
│   │
│   │ save_image_level: Controls which debug images to save
│   │ Range: 0 (none) to 6 (all)
│
└─► STEP 3: Return Instance

END
```

### append_save_image() Flow

```
START: append_save_image(title, keys, gray_image=None, colored_image=None)
│
├─► STEP 1: Validate Title
│   │
│   if not isinstance(title, str):
│   │
│   └─► RAISE TypeError(f"title={title} is not a string")
│       END (exception)
│
├─► STEP 2: Normalize Keys
│   │
│   if isinstance(keys, int):
│   │   keys = [keys]
│   │
│   │ Now keys is always a list
│
├─► STEP 3: Copy Images
│   │
│   gray_image_copy = None
│   colored_image_copy = None
│   │
│   if gray_image is not None:
│   │   gray_image_copy = gray_image.copy()
│   │
│   if colored_image is not None:
│   │   colored_image_copy = colored_image.copy()
│   │
│   │ Purpose: Prevent mutations affecting stored images
│
└─► STEP 4: Append to Storage
    │
    For key in keys:
    │
    ├─► if int(key) > self.save_image_level:
    │   │   continue  # Skip this key (level too high)
    │
    ├─► if gray_image_copy is not None:
    │   │   self.gray_images[key].append([title, gray_image_copy])
    │
    └─► if colored_image_copy is not None:
        │   self.colored_images[key].append([title, colored_image_copy])

END
```

**Purpose**: Store debug images at specific save levels

**Parameters**:
- **title**: Human-readable description (e.g., "After Threshold")
- **keys**: Save level(s) - int or list[int] (1-6)
- **gray_image**: Optional grayscale numpy array
- **colored_image**: Optional BGR numpy array

**Save Level Guidelines**:
```
Level 0: No debug images
Level 1: Final outputs only
Level 2: Major pipeline stages
Level 3: Preprocessing steps
Level 4: Detection details
Level 5: Intermediate calculations
Level 6: Everything (verbose)
```

**Example Usage**:
```python
save_ops = SaveImageOps(config)

# Save at multiple levels
save_ops.append_save_image(
    title="Aligned Image",
    keys=[2, 3],  # Save at levels 2 and 3
    gray_image=aligned_gray,
    colored_image=aligned_color
)

# Save only if level >= 4
save_ops.append_save_image(
    title="Bubble Contours",
    keys=4,
    colored_image=contour_image
)
```

### save_image_stacks() Flow

```
START: save_image_stacks(file_path, save_marked_dir, key=None, images_per_row=4)
│
├─► STEP 1: Determine Save Key
│   │
│   key = self.save_image_level if key is None else key
│
├─► STEP 2: Check Save Level
│   │
│   if self.save_image_level < int(key):
│   │   RETURN  # Nothing to save at this level
│
├─► STEP 3: Get File Stem
│   │
│   stem = file_path.stem
│   │
│   │ Example: file_path = "sheet_01.jpg" → stem = "sheet_01"
│
├─► STEP 4: Save Gray Stack (if any)
│   │
│   if len(self.gray_images[key]) > 0:
│   │
│   ├─► Log titles of images in stack
│   │   logger.info(f"Gray Stack level: {key}", [title for title, _ in gray_images])
│   │
│   ├─► Create stacked image
│   │   result = self.get_result_hstack(self.gray_images[key], images_per_row)
│   │
│   ├─► Build save path
│   │   stack_path = f"{save_marked_dir}/stack/{stem}_{key}_stack.jpg"
│   │
│   ├─► Save image
│   │   ImageUtils.save_img(stack_path, result)
│   │
│   └─► Log save path
│       logger.info(f"Saved stack image to: {stack_path}")
│   │
│   else:
│   │
│   └─► logger.info(f"Note: Nothing to save for gray image. Stack level: {key}")
│
└─► STEP 5: Save Colored Stack (if any)
    │
    if len(self.colored_images[key]) > 0:
    │
    ├─► Log titles
    ├─► Create stacked image
    │   colored_result = self.get_result_hstack(colored_images[key], images_per_row)
    │
    ├─► Build save path
    │   colored_stack_path = f"{save_marked_dir}/stack/colored/{stem}_{key}_stack.jpg"
    │
    ├─► Save image
    │   ImageUtils.save_img(colored_stack_path, colored_result)
    │
    └─► Log save path
    │
    else:
    │
    └─► logger.info(f"Note: Nothing to save for colored image. Stack level: {key}")

END
```

**Purpose**: Save all collected debug images as grid stacks

**Output Files**:
```
save_marked_dir/
├── stack/
│   ├── sheet_01_2_stack.jpg       (gray, level 2)
│   ├── sheet_01_3_stack.jpg       (gray, level 3)
│   └── colored/
│       ├── sheet_01_2_stack.jpg   (colored, level 2)
│       └── sheet_01_3_stack.jpg   (colored, level 3)
```

### get_result_hstack() Flow

```
START: get_result_hstack(titles_and_images, images_per_row)
│
├─► STEP 1: Get Display Dimensions
│   │
│   config = self.tuning_config
│   _display_height, display_width = config.outputs.display_image_dimensions
│   │
│   │ Example: display_width = 400 (pixels)
│
├─► STEP 2: Resize All Images
│   │
│   images = ImageUtils.resize_multiple(
│       [image for _title, image in titles_and_images],
│       display_width
│   )
│   │
│   │ Purpose: Standardize width for grid layout
│
├─► STEP 3: Create Grid Layout
│   │
│   grid_images = MathUtils.chunks(images, images_per_row)
│   │
│   │ Example: 7 images, images_per_row=4
│   │ → [[img1, img2, img3, img4], [img5, img6, img7]]
│
├─► STEP 4: Stack Grid
│   │
│   result = ImageUtils.get_vstack_image_grid(grid_images)
│   │
│   │ Layout:
│   │ [img1] [img2] [img3] [img4]
│   │ [img5] [img6] [img7]
│
└─► STEP 5: Final Resize
    │
    final_width = min(
        len(titles_and_images) * display_width // 3,
        int(display_width * 2.5)
    )
    │
    result = ImageUtils.resize_single(result, final_width)
    │
    │ Purpose: Make final stack image reasonably sized
    │
    RETURN result

END
```

**Purpose**: Create a grid of debug images with standardized sizing

**Layout Strategy**:
1. Resize all images to same width
2. Arrange in grid (4 images per row by default)
3. Vertical stack rows
4. Resize final composite

**Example**:
```
Input: 6 images, images_per_row=3
Output:
┌─────┬─────┬─────┐
│ Img1│ Img2│ Img3│
├─────┼─────┼─────┤
│ Img4│ Img5│ Img6│
└─────┴─────┴─────┘
```

### reset_all_save_img() Flow

```
START: reset_all_save_img()
│
└─► Clear All Levels
    │
    For i in range(7):  # Levels 1-7
    │
    ├─► self.gray_images[i + 1] = []
    └─► self.colored_images[i + 1] = []

END
```

**Purpose**: Clear all stored debug images (called between processing files)

**Levels Cleared**: 1 through 7 (max save level is 6)

**When to Call**: After `save_image_stacks()` and before processing next file

---

## Integration Flow: Complete Debug Image Pipeline

```
START: Process Single OMR Sheet
│
├─► 1. Initialize SaveImageOps
│   save_ops = SaveImageOps(tuning_config)
│
├─► 2. Throughout Processing Pipeline
│   │
│   ├─► After preprocessing:
│   │   save_ops.append_save_image("Preprocessed", keys=3, gray_image=preprocessed)
│   │
│   ├─► After alignment:
│   │   save_ops.append_save_image("Aligned", keys=2, colored_image=aligned)
│   │
│   ├─► After detection:
│   │   save_ops.append_save_image("Detected Bubbles", keys=[2,4], colored_image=detected)
│   │
│   └─► After evaluation:
│       save_ops.append_save_image("Final Result", keys=1, colored_image=final)
│
├─► 3. Save All Stacks
│   │
│   path_utils = PathUtils(output_dir)
│   │
│   For level in [1, 2, 3, 4, 5, 6]:
│   │   if save_ops.save_image_level >= level:
│   │       save_ops.save_image_stacks(
│   │           file_path=sheet_path,
│   │           save_marked_dir=path_utils.save_marked_dir,
│   │           key=level
│   │       )
│
└─► 4. Reset for Next File
    save_ops.reset_all_save_img()

END
```

---

## Browser Migration

### JSON Loading

**Browser Implementation**:

```javascript
async function loadJson(path) {
  // Browser: path is File or URL
  let response;

  if (path instanceof File) {
    // File from input element
    const text = await path.text();
    try {
      return JSON.parse(text);
    } catch (error) {
      throw new ConfigLoadError(
        path.name,
        `Invalid JSON format: ${error.message}`
      );
    }
  } else {
    // URL (fetch from server)
    try {
      response = await fetch(path);
      if (!response.ok) {
        throw new InputFileNotFoundError(path, "JSON");
      }
      return await response.json();
    } catch (error) {
      if (error.name === 'SyntaxError') {
        throw new ConfigLoadError(path, `Invalid JSON format: ${error.message}`);
      }
      throw error;
    }
  }
}
```

**Key Differences**:
1. **File Access**: Use File API instead of filesystem
2. **Async**: All file operations are async
3. **Error Handling**: Different error types (fetch errors vs. file errors)

### Path Utilities

**Browser Implementation**:

```javascript
class PathUtils {
  static printableChars = new Set(
    // ASCII printable: 32-126
    Array.from({length: 95}, (_, i) => String.fromCharCode(i + 32))
  );

  static removeNonUtfCharacters(pathString) {
    return Array.from(pathString)
      .filter(char => PathUtils.printableChars.has(char))
      .join('');
  }

  static sepBasedPosixPath(path) {
    // Browser: always use forward slashes (no backslash conversion needed)
    // Just clean path and normalize slashes
    return PathUtils.removeNonUtfCharacters(
      path.replace(/\\/g, '/')  // Convert backslashes
          .replace(/\/+/g, '/')  // Collapse multiple slashes
    );
  }

  constructor(outputDir) {
    // Browser: outputDir is just a string prefix
    // Actual file operations use downloads or IndexedDB
    this.outputDir = outputDir;

    // Define paths (for display/organization only)
    this.saveMarkedDir = `${outputDir}/CheckedOMRs`;
    this.imageMetricsDir = `${outputDir}/ImageMetrics`;
    this.resultsDir = `${outputDir}/Results`;
    this.manualDir = `${outputDir}/Manual`;
    this.errorsDir = `${this.manualDir}/ErrorFiles`;
    this.multiMarkedDir = `${this.manualDir}/MultiMarkedFiles`;
    this.evaluationsDir = `${outputDir}/Evaluations`;
    this.debugDir = `${outputDir}/Debug`;
  }

  createOutputDirectories() {
    // Browser: No directory creation (download to user's download folder)
    // Or use IndexedDB for storage
    console.log("Output directories prepared (browser mode)");

    // Optional: Create IndexedDB structure
    // await this.initIndexedDB();
  }
}
```

**Browser Adaptations**:

1. **No Directory Creation**:
   - Files downloaded individually or stored in IndexedDB
   - Directory structure is virtual (in filename or IndexedDB keys)

2. **Path Handling**:
   - Always use forward slashes
   - No actual filesystem, just string manipulation

3. **Storage Options**:
   ```javascript
   // Option 1: Downloads with virtual paths
   function saveFile(virtualPath, blob) {
     const link = document.createElement('a');
     link.href = URL.createObjectURL(blob);
     link.download = virtualPath.replace(/\//g, '_');  // Flatten path
     link.click();
   }

   // Option 2: IndexedDB storage
   async function saveToIndexedDB(path, blob) {
     const db = await openDB('omr-outputs', 1);
     await db.put('files', {path, blob});
   }

   // Option 3: ZIP archive
   async function saveAsZip(files) {
     const zip = new JSZip();
     files.forEach(({path, blob}) => {
       zip.file(path, blob);  // Preserves directory structure
     });
     const zipBlob = await zip.generateAsync({type: 'blob'});
     saveFile('omr-output.zip', zipBlob);
   }
   ```

### Debug Image Management

**Browser Implementation**:

```javascript
class SaveImageOps {
  constructor(tuningConfig) {
    this.grayImages = new Map();    // Map<number, Array<[string, ImageData]>>
    this.coloredImages = new Map();
    this.tuningConfig = tuningConfig;
    this.saveImageLevel = tuningConfig.outputs.saveImageLevel;

    // Initialize levels 1-6
    for (let i = 1; i <= 6; i++) {
      this.grayImages.set(i, []);
      this.coloredImages.set(i, []);
    }
  }

  appendSaveImage(title, keys, grayImage = null, coloredImage = null) {
    if (typeof title !== 'string') {
      throw new TypeError(`title=${title} is not a string`);
    }

    // Normalize keys
    if (typeof keys === 'number') {
      keys = [keys];
    }

    // Copy images (clone ImageData or Canvas)
    const grayImageCopy = grayImage ? cloneImageData(grayImage) : null;
    const coloredImageCopy = coloredImage ? cloneImageData(coloredImage) : null;

    for (const key of keys) {
      if (key > this.saveImageLevel) continue;

      if (grayImageCopy) {
        this.grayImages.get(key).push([title, grayImageCopy]);
      }
      if (coloredImageCopy) {
        this.coloredImages.get(key).push([title, coloredImageCopy]);
      }
    }
  }

  async saveImageStacks(filePath, saveMarkedDir, key = null, imagesPerRow = 4) {
    key = key ?? this.saveImageLevel;

    if (this.saveImageLevel < key) return;

    const stem = filePath.split('.')[0];

    // Process gray images
    const grayList = this.grayImages.get(key);
    if (grayList.length > 0) {
      const result = await this.getResultHStack(grayList, imagesPerRow);
      const blob = await canvasToBlob(result);
      await downloadFile(`${saveMarkedDir}/stack/${stem}_${key}_stack.jpg`, blob);
    }

    // Process colored images
    const coloredList = this.coloredImages.get(key);
    if (coloredList.length > 0) {
      const result = await this.getResultHStack(coloredList, imagesPerRow);
      const blob = await canvasToBlob(result);
      await downloadFile(`${saveMarkedDir}/stack/colored/${stem}_${key}_stack.jpg`, blob);
    }
  }

  async getResultHStack(titlesAndImages, imagesPerRow) {
    const displayWidth = this.tuningConfig.outputs.displayImageDimensions[1];

    // Resize images
    const images = await Promise.all(
      titlesAndImages.map(([_, img]) => resizeImage(img, displayWidth))
    );

    // Create grid
    const grid = chunk(images, imagesPerRow);

    // Stack vertically
    const result = await stackImageGrid(grid);

    // Final resize
    const finalWidth = Math.min(
      Math.floor(titlesAndImages.length * displayWidth / 3),
      Math.floor(displayWidth * 2.5)
    );

    return await resizeImage(result, finalWidth);
  }

  resetAllSaveImg() {
    for (let i = 1; i <= 7; i++) {
      this.grayImages.set(i, []);
      this.coloredImages.set(i, []);
    }
  }
}

// Helper functions
function cloneImageData(imageData) {
  return new ImageData(
    new Uint8ClampedArray(imageData.data),
    imageData.width,
    imageData.height
  );
}

async function canvasToBlob(canvas) {
  return new Promise(resolve => {
    canvas.toBlob(resolve, 'image/jpeg', 0.9);
  });
}

function chunk(array, size) {
  const chunks = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}
```

**Browser Considerations**:

1. **Image Representation**: Use Canvas/ImageData instead of numpy arrays
2. **Async Operations**: Canvas operations are async
3. **Memory Management**: Explicitly release ImageData when done
4. **Download Strategy**: Use blob URLs or ZIP archives

---

## Related Documentation

- **Image Utils**: `../image/flows.md`
- **Math Utils**: `../math/flows.md`
- **File System Patterns**: `../../../technical/filesystem/filesystem-operations.md`
- **Error Handling**: `../../../foundation/error-handling.md`

---

## Summary

File Utils provides three key components:

1. **load_json()**: Safe JSON loading with rich error messages
2. **PathUtils**: Cross-platform path handling and directory structure
3. **SaveImageOps**: Debug image collection and grid visualization

**Python Features**:
- Path library for cross-platform paths
- defaultdict for automatic list initialization
- Logging integration for directory creation

**Browser Migration**:
- File API for JSON loading
- Virtual directory structure (no filesystem)
- Canvas for image operations
- Downloads or IndexedDB for output
- ZIP archives for batch downloads

**Key Pattern**: Separate path logic (virtual) from storage logic (actual I/O)
