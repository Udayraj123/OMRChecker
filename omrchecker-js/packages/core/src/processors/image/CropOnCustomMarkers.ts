/**
 * CropOnCustomMarkers — TypeScript port placeholder.
 *
 * Port of Python: src/processors/image/crop_on_patches/custom_markers.py
 * Issue: omr-c6r
 *
 * The browser-compatible implementation lives in CropOnMarkers.ts, which
 * implements the CropOnCustomMarkers logic for the FOUR_MARKERS layout
 * directly inside the CropOnMarkers class.
 *
 * Dynamic layout lookup (points_layout) replaces the hardcoded "FOUR_MARKERS"
 * key in scanZonePresetsForLayout — matching the Python refactor in omr-c6r.
 *
 * TODO: Expand this stub into a full port of CropOnCustomMarkers with
 *       subclass-extensible scanZonePresetsForLayout when additional layout
 *       types are added beyond FOUR_MARKERS.
 *
 * Note(omr-sun): Python equivalent now uses CropOnPatchesCommon._build_base_parsed_options
 *       for the validate_and_remap_options_schema base dict. CropOnMarkers.validateAndRemapOptionsSchema
 *       mirrors that logic inline.
 */

export { CropOnMarkers as CropOnCustomMarkers } from './CropOnMarkers';
