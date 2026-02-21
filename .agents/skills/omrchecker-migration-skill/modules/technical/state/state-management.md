# State Management in OMRChecker

**Module**: modules/technical/state/
**Created**: 2026-02-20

## Python State

**ProcessingContext**: Carries state through pipeline
```python
class ProcessingContext:
    file_path: str
    gray_image: np.ndarray
    colored_image: Optional[np.ndarray]
    field_id_to_interpretation: Dict[str, FieldInterpretation]
```

## Browser State Management

### Option 1: Plain Objects

```javascript
class ProcessingContext {
  constructor(filePath) {
    this.filePath = filePath;
    this.grayImage = null;
    this.coloredImage = null;
    this.interpretations = new Map();
  }
}
```

### Option 2: Zustand (Recommended)

```bash
npm install zustand
```

```javascript
import create from 'zustand';

const useOMRStore = create((set) => ({
  template: null,
  currentImage: null,
  results: [],

  setTemplate: (template) => set({ template }),
  setCurrentImage: (image) => set({ currentImage: image }),
  addResult: (result) => set((state) => ({
    results: [...state.results, result]
  }))
}));
```

### Option 3: Redux (Large Apps)

```javascript
const omrSlice = createSlice({
  name: 'omr',
  initialState: {
    template: null,
    processing: false,
    results: []
  },
  reducers: {
    setTemplate: (state, action) => {
      state.template = action.payload;
    }
  }
});
```

**Recommendation**: Plain objects for simple state, Zustand for reactive UI updates.
