# Sample Markers Reference

## Overview

All tests in `test_all_samples.py` are marked with `@pytest.mark.sample_<normalized_path>` to prevent future test collisions. The marker name is derived from the sample path by:
- Replacing `/` with `_`
- Replacing `-` with `_`
- Prefixing with `sample_`

## Sample Marker Mapping

| Test Function | Sample Path | Marker |
|--------------|-------------|--------|
| `test_run_omr_marker_mobile` | `1-mobile-camera` | `@pytest.mark.sample_1_mobile_camera` |
| `test_run_omr_marker` | `2-omr-marker` | `@pytest.mark.sample_2_omr_marker` |
| `test_run_bonus_marking` | `3-answer-key/bonus-marking` | `@pytest.mark.sample_3_answer_key_bonus_marking` |
| `test_run_bonus_marking_grouping` | `3-answer-key/bonus-marking-grouping` | `@pytest.mark.sample_3_answer_key_bonus_marking_grouping` |
| `test_run_answer_key_using_csv` | `3-answer-key/using-csv` | `@pytest.mark.sample_3_answer_key_using_csv` |
| `test_run_answer_key_using_image` | `3-answer-key/using-image` | `@pytest.mark.sample_3_answer_key_using_image` |
| `test_run_answer_key_using_image_grouping` | `3-answer-key/using-image-grouping` | `@pytest.mark.sample_3_answer_key_using_image_grouping` |
| `test_run_answer_key_weighted_answers` | `3-answer-key/weighted-answers` | `@pytest.mark.sample_3_answer_key_weighted_answers` |
| `test_run_crop_four_dots` | `experimental/1-timelines-and-dots/four-dots` | `@pytest.mark.sample_experimental_1_timelines_and_dots_four_dots` |
| `test_run_crop_two_dots_one_line` | `experimental/1-timelines-and-dots/four-dots` | `@pytest.mark.sample_experimental_1_timelines_and_dots_four_dots` ⚠️ |
| `test_run_two_lines` | `experimental/1-timelines-and-dots/two-lines` | `@pytest.mark.sample_experimental_1_timelines_and_dots_two_lines` |
| `test_run_template_shifts` | `experimental/2-template-shifts` | `@pytest.mark.sample_experimental_2_template_shifts` |
| `test_run_feature_based_alignment` | `experimental/3-feature-based-alignment` | `@pytest.mark.sample_experimental_3_feature_based_alignment` |
| `test_run_community_Antibodyy` | `community/Antibodyy` | `@pytest.mark.sample_community_Antibodyy` |
| `test_run_community_ibrahimkilic` | `community/ibrahimkilic` | `@pytest.mark.sample_community_ibrahimkilic` |
| `test_run_community_Sandeep_1507` | `community/Sandeep-1507` | `@pytest.mark.sample_community_Sandeep_1507` |
| `test_run_community_Shamanth` | `community/Shamanth` | `@pytest.mark.sample_community_Shamanth` |
| `test_run_community_UmarFarootAPS` | `community/UmarFarootAPS` | `@pytest.mark.sample_community_UmarFarootAPS` |
| `test_run_community_JoyChopra1298` | `community/JoyChopra1298` | `@pytest.mark.sample_community_JoyChopra1298` |

## ⚠️ Detected Collision

**Two tests use the same sample:**
- `test_run_crop_four_dots`
- `test_run_crop_two_dots_one_line`

Both use: `experimental/1-timelines-and-dots/four-dots`

**Status:** Both tests are marked with `@pytest.mark.sample_experimental_1_timelines_and_dots_four_dots`, which helps identify the collision. Since they're in the same file and use `--dist=loadscope`, they run on the same worker, preventing conflicts.

## Usage

### Check which tests use a specific sample

```bash
# Find all tests using a specific sample
uv run pytest src/tests/test_all_samples.py -m "sample_1_mobile_camera" --co

# Find all tests using experimental samples
uv run pytest src/tests/test_all_samples.py -m "sample_experimental" --co

# Find all tests using community samples
uv run pytest src/tests/test_all_samples.py -m "sample_community" --co
```

### Before adding a new test

1. **Check if sample path is already used:**
   ```bash
   # Normalize your sample path: "new/sample-path" -> "sample_new_sample_path"
   # Then check:
   uv run pytest src/tests/test_all_samples.py -m "sample_new_sample_path" --co
   ```

2. **If marker exists:** The sample is already in use. Either:
   - Use a different sample path
   - Ensure the existing test and new test can safely share the sample (same worker via loadscope)

3. **If marker doesn't exist:** Add the marker to your new test:
   ```python
   @pytest.mark.sample_new_sample_path
   def test_new_feature(run_sample, mocker, snapshot) -> None:
       """Test using sample: new/sample-path"""
       sample_outputs = run_sample(mocker, "new/sample-path")
       assert snapshot == sample_outputs
   ```

## Benefits

1. **Collision Detection:** Easy to identify if a sample path is already in use
2. **Documentation:** Clear mapping between tests and sample paths
3. **Filtering:** Can run tests for specific samples
4. **Future-Proof:** Prevents accidental reuse of sample paths

## Marker Normalization Rules

To create a marker from a sample path:
1. Prefix with `sample_`
2. Replace `/` with `_`
3. Replace `-` with `_`
4. Keep alphanumeric characters and underscores

**Examples:**
- `1-mobile-camera` → `sample_1_mobile_camera`
- `3-answer-key/bonus-marking` → `sample_3_answer_key_bonus_marking`
- `community/Sandeep-1507` → `sample_community_Sandeep_1507`

