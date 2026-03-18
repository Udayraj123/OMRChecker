# Right-To-Left (RTL) Custom Fields

## Background

Some OMR sheets used in RTL languages have answer bubbles that read from right to left (e.g. D, C, B, A) instead of the more typical left to right (e.g. A, B, C, D).

## Using Built-in RTL Field Types
OMRChecker includes two built-in RTL field types to be used in your template.

`QTYPE_MCQ4_RTL` - Representing 4 choice questions with reversed bubble values ["D", "C", "B", "A"]

`QTYPE_MCQ5_RTL` - Representing 5 choice questions with reversed bubble values ["E", "D", "C", "B", "A"]

These can be used in your `template.json` similar to the other field type:
```
"MCQBlock_RTL": {
    "fieldType": "QTYPE_MCQ4_RTL",
    "fieldLabels": ["q1..10"],
    "bubblesGap": 40,
    "labelsGap": 50,
    "origin": [100, 100]
}
```
## Creating Custom RTL Field
For situations where you might need a different number of choices, you can create a custom reversed field type inline by specifying `bubbleValues` and `direction` in the field block.
```
"MCQBlock_CUSTOM_RTL": {
    "bubbleValues": ["G", "F", "E", "D", "C", "B", "A"],
    "direction": "horizontal",
    "fieldLabels": ["q1..7"],
    "bubblesGap": 40,
    "labelsGap": 50,
    "origin": [100, 100]
}
```
Alternatively, you can also just add the custom field type in `src/constants/common.py` under `FIELD_TYPES` for reuse.

Reference to Issue #234 for original issue discussion on RTL feature.
