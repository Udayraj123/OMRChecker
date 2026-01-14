# CropOnPatchesCommon Refactoring Analysis

## 📊 Current State

**File**: `src/processors/image/CropOnPatchesCommon.py`
**Size**: 379 lines
**Role**: Base class for patch-based detection processors (CropOnDotLines, CropOnCustomMarkers)
**Extends**: WarpOnPointsCommon

---

## 🔍 Responsibility Analysis

### Current Responsibilities (9 areas)

1. **Zone Configuration & Validation** (Lines 68-135)
   - `parse_and_apply_scan_zone_presets_and_defaults()` - 23 lines
   - `validate_scan_zones()` - 15 lines
   - `validate_points_layouts()` - 28 lines

2. **Point Extraction Orchestration** (Lines 136-235)
   - `extract_control_destination_points()` - 100 lines ⚠️ **LARGEST METHOD**
   - Coordinates detection, visualization, validation

3. **Runtime Zone Management** (Lines 236-260)
   - `get_runtime_zone_description_with_defaults()` - 2 lines
   - `get_edge_contours_map_from_zone_points()` - 22 lines

4. **Visualization** (Lines 261-294)
   - `draw_zone_contours_and_anchor_shifts()` - 34 lines
   - `draw_scan_zone_util()` - 28 lines

5. **Point Selection** (Lines 295-342)
   - `find_and_select_point_from_dot()` - 29 lines
   - `select_point_from_rectangle()` - 17 lines ⚠️ **PURE UTILITY**

6. **Zone Computation** (Lines 343-351)
   - `compute_scan_zone_util()` - 9 lines ⚠️ **DELEGATES TO ShapeUtils**

7. **Abstract Methods** (Lines 35-45)
   - `find_and_select_points_from_line()` - NotImplementedError
   - `find_dot_corners_from_options()` - NotImplementedError

8. **Initialization** (Lines 47-66)
   - `__init__()` - 20 lines
   - Lifecycle methods

9. **Simple Utilities** (Lines 59-66)
   - `exclude_files()` - 2 lines
   - `__str__()` - 2 lines
   - `prepare_image_before_extraction()` - 2 lines

---

## 🎯 Refactoring Opportunities

### Option 1: Extract Zone Management Module ⭐ (RECOMMENDED)
**Create**: `patch_zone_management.py` or `scan_zone_utils.py`

**Extract Functions** (~68 lines):
```python
# Zone configuration
def parse_scan_zones_with_defaults(
    scan_zones: list,
    default_descriptions: dict,
) -> list:
    """Parse and apply defaults to scan zones"""
    pass

def validate_scan_zone_labels(scan_zones: list) -> None:
    """Validate no duplicate labels"""
    pass

def validate_points_layout(
    points_layout: str,
    scan_zone_presets: dict,
    provided_zones: list,
) -> None:
    """Validate layout matches expected zones"""
    pass

# Point selection
def select_point_from_rectangle(
    rectangle: np.ndarray,
    selector: str,
) -> Optional[tuple]:
    """Select a specific point from rectangle corners"""
    pass

# Edge mapping
def build_edge_contours_map(
    zone_preset_points: dict,
    target_endpoints: dict,
) -> dict:
    """Build edge contours map from zone points"""
    pass
```

**Benefits**:
- ✅ Reusable zone validation logic
- ✅ Easier to test independently
- ✅ ~68 lines extracted (18% reduction)
- ✅ Clean separation: orchestration vs. utilities

**Effort**: Low-Medium (2-3 hours)

---

### Option 2: Extract Point Selection Strategy ⭐
**Create**: `point_selection.py` or add to `point_utils.py`

**Extract** (~46 lines):
```python
class PointSelector:
    """Strategy for selecting points from shapes"""

    @staticmethod
    def select_from_rectangle(rectangle, selector):
        """Select point from rectangle (TL, TR, BR, BL, CENTER)"""
        pass

    @staticmethod
    def select_from_dot_zone(
        dot_corners, zone_description, selector
    ):
        """Find and select point from dot zone"""
        pass
```

**Benefits**:
- ✅ Strategy Pattern for point selection
- ✅ Easier to add new selection methods
- ✅ ~46 lines extracted (12% reduction)

**Effort**: Low (1-2 hours)

---

### Option 3: Leave As-Is (Base Class Coordination) ❌
**Reasoning**: This is already a well-organized base class

**Arguments Against Refactoring**:
- Most methods are orchestration (not algorithmic)
- Already uses extracted detection modules (via subclasses)
- Complexity is in coordination, not computation
- 379 lines is reasonable for a base class
- Further extraction may over-engineer

**Verdict**: **CropOnPatchesCommon is already well-factored**

The recent refactorings (CropOnDotLines, CropOnCustomMarkers) extracted the **algorithmic complexity** (detection logic) into modules. What remains in CropOnPatchesCommon is **orchestration logic**, which belongs in a base class.

---

## 💡 Recommendation

### **Option 3: Leave CropOnPatchesCommon As-Is** ⭐

**Reasoning**:
1. ✅ **Already refactored indirectly**: Detection logic extracted from subclasses
2. ✅ **Appropriate size**: 379 lines for a coordinator base class is acceptable
3. ✅ **Clear responsibilities**: Zone management, orchestration, visualization
4. ✅ **No algorithmic complexity**: Most complexity pushed to subclasses/modules
5. ✅ **Testable through subclasses**: CropOnDotLines, CropOnCustomMarkers tests cover this

**What Makes It Well-Designed**:
- Template Method Pattern: Defines workflow, delegates specifics to subclasses
- Single Responsibility: Coordinates patch-based detection, doesn't implement it
- Open/Closed: Easy to extend with new scanner types
- Dependency Inversion: Depends on abstractions (abstract methods for subclasses)

**Size Comparison**:
- WarpOnPointsCommon (after refactoring): 516 lines ✅
- CropOnPatchesCommon: 379 lines ✅
- Average base class size: ~400-500 lines ✅

---

## 🎯 Next Action

### Skip CropOnPatchesCommon Refactoring → Move to CropOnMarkers

**CropOnMarkers.py Analysis**:
- Size: 33 lines
- Role: Simple delegator between CropOnDotLines and CropOnCustomMarkers
- Expected effort: 15-30 minutes review
- Likely outcome: No refactoring needed (too simple)

**Then**: **Phase 3 Python Complete!** ✅

---

## 📊 Phase 3 Summary (If Skipping This File)

| File | Status | Lines | Reduction | Pattern |
|------|--------|-------|-----------|---------|
| CropPage | ✅ Done | 235 → 136 | -99 (-42%) | Extract page_detection |
| CropOnCustomMarkers | ✅ Done | 480 → ~340 | -140 (-29%) | Extract marker_detection |
| CropOnDotLines | ✅ Done | 528 → 357 | -171 (-32%) | Extract dot_line_detection |
| **CropOnPatchesCommon** | ✅ **Skip** | **379** | **N/A** | **Already well-factored** |
| CropOnMarkers | ⏳ Review | 33 | TBD | Simple delegator |

**Total Reduction**: 410 lines across 3 files
**Phase 3 Completion**: 80% (if skipping CropOnPatchesCommon)

---

## ✅ Decision

**Skip CropOnPatchesCommon refactoring** because:
1. It's already well-organized as a coordinator base class
2. Algorithmic complexity already extracted to subclass detection modules
3. Further extraction would be over-engineering
4. 379 lines is appropriate for its responsibility level
5. Better to spend time on TypeScript port

**Next**: Review CropOnMarkers.py (15-30 minutes), then **start TypeScript port**!

---

**Analysis Date**: January 14, 2026
**Recommendation**: Skip refactoring, move to TypeScript port
**Phase 3 Status**: 3/5 files refactored (60%), 4/5 reviewed (80%)

