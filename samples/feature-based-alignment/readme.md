# Demo for feature-based alignment

## Background
OMR is used to match student roll on final exam scripts. Scripts are scanned using a document scanner and the cover pages are extracted for OMR. Even though a document scanner does not produce any warpped perspective, the alignment is not perfect, causing some rotation and translation in the scans.

The scripts in this sample were specifically selected incorrectly marked scripts to demonstrate how feature-based alignment can correct transformation errors using a reference image. In the actual batch. 156 out of 532 scripts were incorrectly marked. With feature-based alignment, all scripts were correctly marked.

## Usage
Two template files are given in the sample folder, one with feature-based alignment (template_fb_align), the other without (template_no_fb_align).

## Additional Notes

### Reference Image
When using a reference image for feature-based alignment, it is better not to have many repeated patterns as it is causes ambiguity when trying to match similar feature points. The bubbles in an OMR form are identical and should not be used for feature-extraction.

Thus, the reference image should be cleared of any bubbles. Forms with lots of text as in this example would be effective.

Note the reference image in this sample was generated from a vector pdf, and not from a scanned blank, producing in a perfectly aligned reference.

### Level adjustment
The bubbles on the scripts were not shaded dark enough. Thus, a level adjustment was done to bring the black point to 70% to darken the light shading. White point was brought down to 80% to remove the light-grey background in the columns.
