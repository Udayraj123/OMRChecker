# OMR-Scanner
A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. With support for a customizable marking scheme with section-wise marking, bonus questions, etc. 

## TODOs
[X] Normalize the brightness: Run on 10-15 scans of empty OMRs for threshold tuning
	> Can you plot the histogram on single image? _/_/
[X] For multi marks: Write algo to check threshold locally - variance values
	> Can you plot the threshold distribution? _/_/
[ ] For shifted templ: Find out methods to force align on rectangular strips
	> Prob- some given scans also have negligible bg strips

## Curr Tasks
(28-30Aug 2018)
### 1) Run the outputs into omr detector.
	1.1 [X] Rescale Thresholds into 0-255 range
	1.2 [X] Show clr vals at the boxes.
	1.3 [X] Readjust threshold values
	1.4 [X] Show b4_after comparision

(11 Sept 2018)
### 2) Redesign the template making code.
	2.1 [X] Make Q and Pt class
	2.2 [X] Rethink and implement grid gen functions
	2.3 [X] Fit the template for ints
(12 Sept 2018)
	2.4 [X] JSONify the template
	2.5 [X] Fit the template for mcq and rolls

(12 Sept 2018)
### 3) Plot for each question
	3.1 [X] Change readResponse to adapt to Q class
	3.2 [X] Make hist subplots
	3.3 [X] Make boxplot subplots _/_/
	3.4 [X] Record/Report the progress

### 4) Refactor readResponse completely
	4.1 [ ] Simplify resp array generation
	4.2 [ ] Simplify detection process
	4.3 [ ] Minimize logging 

### 5) Make use of rectangular strips
	5.1 Think how would it be beneficial
	5.2 Find what methods are used to do this
	5.3 Think of adding it in code 

## Design Choices

### Function Means chart
| Function | Means 	| | | 
-----------------------------------------------------------------------
| Choosing ROI | [Four Circles] | Four Dots | Sidebars |
-----------------------------------------------------------------------
| Adding boxes |		|			  |						|
| in template  | Individual boxes | QGroup-wise | [Q-wise] |
-----------------------------------------------------------------------
