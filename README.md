# OMR-Scanner
A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. With support for a customizable marking scheme with section-wise marking, bonus questions, etc. 

## TODOs
0. Normalize the brightness: Run on 10-15 scans of empty OMRs for threshold tuning
	> Can you plot the histogram on single image?
1. For multi marks: Write algo to check threshold locally - variance values
	> Can you plot the threshold distribution?
2. For shifted templ: Find out methods to force align on rectangular strips
	> Prob- some given scans also have negligible bg strips

## Curr Tasks
### 1) Run the outputs into omr detector.
	1.1 [X] Rescale Thresholds into 0-255 range
	1.2 [X] Show clr vals at the boxes.
	1.3 [X] Readjust threshold values
	1.4 [X] Show b4_after comparision



## Design Choices

### Function Means chart
| Function | Means 	| | | 
-----------------------------------------------------------------------
| Choosing ROI | [Four Circles] | Four Dots | Sidebars |
-----------------------------------------------------------------------
| Adding boxes |		|			  |						|
| in template  | Individual boxes | QGroup-wise | [Q-wise] |
-----------------------------------------------------------------------
