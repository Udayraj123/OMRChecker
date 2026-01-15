# 🔍 Code Review: Unused Functions & Missing Deep Copies

**Date**: January 16, 2026
**Reviewer**: AI Assistant
**Scope**: TypeScript utils and evaluation modules

---

## ❌ Issues Found

### 1. **deepClone is NOT currently used** (but WILL BE NEEDED)

**Status**: ⚠️ **Keep it - Required for bug fix**

**Current Usage**:
- ✅ Exported from `utils/object.ts`
- ✅ Exported from `@omrchecker/core`
- ✅ Has comprehensive tests
- ❌ **NOT used anywhere in production code**

**BUT Python Uses `deepcopy` in:**
- `SectionMarkingScheme.deepcopy_with_questions()`
- `AnswerMatcher` (marking dict copy)
- Several other places

---

### 2. **Bug Found: Shallow Copy in SectionMarkingScheme** 🐛

**File**: `SectionMarkingScheme.ts` line 261-267

**Current Code (WRONG)**:
```typescript
deepcopyWithQuestions(questions: string[]): SectionMarkingScheme {
  const clone = Object.create(Object.getPrototypeOf(this));
  Object.assign(clone, this); // ⚠️ SHALLOW COPY!
  clone.questions = questions;
  clone.validateMarkingScheme();
  return clone;
}
```

**Python Code (CORRECT)**:
```python
def deepcopy_with_questions(self, questions):
    clone = deepcopy(self)  # ✅ DEEP COPY
    clone.update_questions(questions)
    return clone
```

**Problem**:
- Our TypeScript version does **shallow copy**
- Python does **deep copy**
- This means nested objects (like `marking`, `streaks`) are **shared references**
- Modifying the clone's nested properties affects the original! 🐛

**Impact**:
- ⚠️ High - Could cause subtle bugs with streak tracking
- If multiple schemes share references, they could interfere with each other

---

### 3. **Bug Found: Shallow Copy in AnswerMatcher** 🐛

**File**: `AnswerMatcher.ts` line 149

**Current Code (WRONG)**:
```typescript
private setLocalMarkingDefaults(sectionMarkingScheme: SectionMarkingScheme): void {
  this.emptyValue = sectionMarkingScheme.emptyValue;

  // Make a copy of section marking locally
  this.marking = { ...sectionMarkingScheme.marking }; // ⚠️ SHALLOW COPY!

  // ... more code
}
```

**Python Code (CORRECT)**:
```python
def set_local_marking_defaults(self, section_marking_scheme) -> None:
    self.empty_value = section_marking_scheme.empty_value
    # Make a copy of section marking locally
    self.marking = deepcopy(section_marking_scheme.marking)  # ✅ DEEP COPY
```

**Problem**:
- If `marking` values are objects/arrays, they're shared references
- For streak marking (arrays), this could cause issues

**Impact**:
- ⚠️ Medium - Only affects if marking values are complex objects
- Current simple number values are okay, but fragile

---

## ✅ Recommendation: Fix the Bugs

### **Action 1**: Update `SectionMarkingScheme.deepcopyWithQuestions()`

```typescript
import { deepClone } from '../../utils/object';

deepcopyWithQuestions(questions: string[]): SectionMarkingScheme {
  const clone = deepClone(this);
  clone.updateQuestions(questions);
  return clone;
}
```

### **Action 2**: Consider updating `AnswerMatcher.setLocalMarkingDefaults()`

For robustness, use deep copy:
```typescript
import { deepClone } from '../../utils/object';

private setLocalMarkingDefaults(sectionMarkingScheme: SectionMarkingScheme): void {
  this.emptyValue = sectionMarkingScheme.emptyValue;
  this.marking = deepClone(sectionMarkingScheme.marking);
  // ... rest of code
}
```

---

## 📊 Other Unused/Underused Functions Check

### Checking all utils for usage...

Let me check each utility function:

1. ✅ **isObject** - Used by `deepMerge`
2. ✅ **deepMerge** - Used by `EvaluationConfig`
3. ⚠️ **deepClone** - NOT used yet, but NEEDED for bug fixes

### Other potential issues:

Checking for other functions that might be unused...

---

## 🎯 Summary

| Issue | Severity | Status | Action Needed |
|-------|----------|--------|---------------|
| `deepClone` unused | ⚠️ Low | Keep it | Required for fixes |
| Shallow copy in `deepcopyWithQuestions` | 🐛 High | Fix needed | Use `deepClone` |
| Shallow copy in `setLocalMarkingDefaults` | ⚠️ Medium | Consider | Use `deepClone` |

---

## 💡 Recommendation

**DO NOT remove `deepClone`** - it's needed to fix the bugs above!

Instead:
1. ✅ **Fix `SectionMarkingScheme.deepcopyWithQuestions()`** - Use `deepClone` (matches Python)
2. ✅ **Fix `AnswerMatcher.setLocalMarkingDefaults()`** - Use `deepClone` (matches Python)
3. ✅ **Keep `deepClone`** - Now it will be used!

This will ensure TypeScript matches Python's behavior exactly.

---

**Would you like me to implement these fixes?**

