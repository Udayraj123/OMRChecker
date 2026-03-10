## Sample OMRs
# What is a Dot (index point)?
We can use any blob-like unique shape on the OMR sheet to be used as the four index points aka Dots.
These 4 points provided at the 4 corners of the OMR sheet are the most important components of an OMR sheet. The accuracy of evaluation depends on the clarity of these 4 points in the scanned OMR sheet image.

# What is a Timeline on an OMR sheet?

In the old pattern, machine read sheets, were additional black marks placed at equal increments running throughout the length of the sheet on either side or on both sides. This strip of black marks is called a timeline which helps the OMR machine to identify the next row of bubbles. ([source](https://www.addmengroup.com/downloads/Addmen-OMR-Sheet-Design-Guide.pdf))

OMR software can use the timeline to properly locate the bubbles. So any changes in the timeline markers can generate wrong result. It is usually not larger than the size of bubbles

<!-- TODO: image of a timeline -->

# When should I use it?

- If your OMR sheet has this timeline and if adding a custom marker(in four corner points) is not feasible, you can try evaluating your OMR sheet using this method.
- Since the preprocessor converts the dashed line into a filled rectangular blob and uses that for detection, you can use this preprocessor on any rectangular thick lines or thick dots that can be converted to distinguishable blobs.

# How to use?
We have provided support for 4 configurations: "ONE_LINE_TWO_DOTS", "TWO_DOTS_ONE_LINE", "TWO_LINES", "FOUR_DOTS".
Depending on the configuration of your OMR sheet, choose the one that is applicable.

Open the samples in this folder to know how to configure each type in detail.

Also, for further tuning - we support 5 configurations for selecting the points out of the thick blobs: "CENTERS", "INNER_WIDTHS", "INNER_HEIGHTS", "INNER_CORNERS", "OUTER_CORNERS"

### Drawbacks and Improvements
Currently OMRChecker requires the Dots and Lines to "stand out" from other marks in the sheet. If there's any noise nearby the marker including the print itself, the detection would fail.
<!-- TODO: insert obstructed OMR -->

Track improvements in this functionality [here](https://github.com/users/Udayraj123/projects/2?pane=issue&itemId=57863176).
