//Try out contours and see the marker
>> Centre of that contour will be close to that center!!
>> There's also hough transform, but it aint as flexible (needs min max radius)
>> If we ARE using a specific marker like the circle,
	Why not use a barcode? (Hint: test on perspectives)
	https://www.pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/
	^We can, but now you've DECIDED to use the circle and it's a good decision, so get on with it!

^>> So, findContours seems not reliable as is. Works a bit on binarizing the image, but same prob with xeroxed image would be here. And this approach will be subject to background changes.

	Rather can you find the sidelines so as to determine quadrants?
	>> **Orientation can be determined by DFT
		https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
	>> The sidelines are not as good markers as a filled strip!

>> Why not use a blob instead of circles?
	- Distance of sheet from camera variable.
	>> What if we add it to input constraint that biggest blob should be the one on OMR.
	- Template matching not as accurate as on concentric circles

** This Sobel operator can be used for aligning the template!!
	And morphing too -
	https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html


17 Feb
> Two layered warping - 1st page, 2nd marker.
Refactor more-
> globalized redundant arguments
> getROI and match_template_scaled. Line changes: 188 -> 152

18 Feb
> Why were you matching using eroded template?
--> Was in the blog, it was better than using gray one.
--> Seemed to remember the deeper concentric patterns, they were causing inaccurate results (many other white areas in image)
--> Binarized image be better!

20 Feb

// Tune the page cropping parameters

// Debug template matching


// Checkout this weird trick that makes marked bubbles and the template circle distinctly blobby-
	gray = gray - cv2.erode(gray, kernel=np.ones((5,5)),iterations=5)

// (Fri) Mail the progress (pic dumps)


///////////////
# Notes
Resizes in algo

template.py
	template = resize_util(template, int(template.shape[1]/templ_scale_down))

show()
 	img = resize_util(orig,display_width) if resize else orig

getBestMatch
    templ_scaled = imutils.resize(template_eroded_sub, height = int(h*s))

getROI
    image_norm = resize_util(image_norm, uniform_width_hd)
    # Resize back to uniform height
    warped_image_eroded_sub = resize_util(warped_image_eroded_sub, uniform_width_hd)
    warped_image_norm = resize_util(warped_image_norm, uniform_width_hd)
    # resize to best scale
	templ = imutils.resize(template_eroded_sub, height = int(template_eroded_sub.shape[0]*best_scale))

	# template fit resizing
    img = resize_util(img,1846,1500)

testImg
	img = resize_util(img,uniform_width_hd)
///////////////

TODO After MS:
- Refactor readResponse
//- Shifting ideation
- Apply DFT and IFT on cropped one just to see.
- Get more test data : On which template shifts currently

Excess:
	(done 17 Apr - getMaxCosine from app) Mechanisms to check if circles found form a rectangle-like shape
	Defeat the bossbg.jpg
	3D viz (online) of this templateMatch output(, Sobel eroded blobs) to see the peaks - of morph output

Shifting -
Problem: [Nonlinear Distortion] There's an uneven shift(mainly horizontal) of order of 2-3 pixels in the Qbox columns

Nope- Rect level soln
	> Easier to align
	> Assumes there's a bounding rect present for alignment
	> Still the rects are long, nonlinear distortion would still mess it up

Qbox level soln
	Naive approach (inefficient here)
		> Take each QBlock, move it around in 20% area and return the pos where an index is maximum.
			The index can be -
			1. Correlation : Error chances significant as there are smaller box lines that also correlate
			2. Correlation on Eroded image : Gotta try
				It is taken with white color anyway: so just take max white value.
		_/	3. Correlation on Gradient image : More consistent

	Different approach 1
		> We'd always have partially filled(gray or black) Qboxes, move them individually towards the black area
			- wont work on noise due to xerox bg!

** > readResponse works better on this (moderately) eroded_sub?
- Nope, the unfilled ones also become dark, only the boundary around filled ones can be utilised for something awesome


16 Mar 19
Major Changes:
	// Implemented shifting in align file


Minor Changes:
	Resize page(>>BOTH W & H) to fixed size after Warp layer 1
	==> No need of scaled checking?! : Still doing precautionary

19 Mar 19
// Implemented (naive) shifting in utils

Minor:
// Separated JSONs from template.py
// Rather, removed it-  scalePts was not acting since rect{} was introduced, also updated it with int() on pts to support fractional scaling

Some dilemmas and decisions over them-
> Template json contents : Should 'qblockDims' be written in the file or calculated from Gaps already present there
	--> Minimal redundancy should be followed, so calculate.
	--> Giving it in file also introduces interdependencies - changing qNos reqs recalculating qblockDims again.
	<-- But won't it be fixed for the template? And more accurate than calculating using gaps?

> Making QBlock class :
	<-- Keeping as array was simple, now more loops introduced
	--> The paradigm was demanded on need by my subcons, gotta trust my guts this time.


# Let's Apply adaptive threshold QBlock-wise.
# >> Nope, do column-wise or don't -
# Then need to separate the Qs array - doable, but extensible?
# >> No need! You just need QVals, can get it cumulating via QBlocks too, get it as you find shifts

Wish me clarity!!
>> (From Speech TA) 7x1 kernel erosion is damn perfect suggestion ==> Makes use of real morphology power
>> Not now, sometime later! - My Sobel be useful tho (feed the stumbling)
--> This is why you should work on thinking clearly first before acting.

Things that should have clicked(altho not useful everytime) when seen first time!
--> Denoise first using morphology! ( Well, didn't work as expected now!)
--> Blur then denoise - nop

Sobel :
	is an approximation to derivative _//
	sign of output doesn't matter _//
	(1,0) means horizontal gradient _//


21 Mar 19
// Implemented proper shifting in utils
	Nope, there's more accurate approach - Qval correlation instead of QBlock be most accurate
	- Area between Qboxes be checked : use a mask!
	>> also useful for future changes : if decided to align vertically as well
--> Further accurate: approach4 - move towards the blob - use MOMENTS!
	- move towards centroid until direction changes.

Minor:
	Nope- Q object now has endpts and a mask for aligning
		QBlock has cols attribute for vertical cols (even for orient='H') and No mask needed practically

- Test all on dark xerox now.
	- even white xeroxes making probs now, need to change aligning method to more robust


23 Mar 19
	: Shifting to its best! Scan match Centroid wins by a nice margin!
	 - On xerox : Bangalore_JE series
	 >> Its working decently on positive shifts,
	 - still many times it shifts wrongly, but detected correctly

	: How about, finding the first 0-255 jump? There ain't noise there!
		<- there could be in some cases
		<- first match wont solve alignment, total avg alignment be max
		-> There may not be many cases, even if they are- avg alignment method would also fail
		<- For negative shifts?
		<- Rather store initial centroids with QBlocks.

	(after discn with Mk)
	_//	->> Move to make white on both edges of the window.
	Works really well! Thanks mk, now I can totally move on from shifting. _// (23 mar 11:28PM)

	Now:
		// Refactor readResponse

	Major Changes:
		// Improve threshold selection : separate by QBlock.cols --> still needs a complicated DS
				**Note: Orient='H' still has to have vertical alternate lines (logically)
				Threshold from gap needs to be columnwise- Xerox grays are pretty bad
					- e.g. on HE_Bangalore_01 Xeroxed sheet it works miserably.
					--> Qblock class shall help here. Need colwise Pt list for qblock
					<-- What about no bubbles marked in column: then variance would be very low (add this condition).

			OMRresponse was 2-key dict only for combining the integer types. It should be done during evaluation rather.
			-> concatenation be done there itself

			No need for Q() class, moved attr to Pt class.
		// Gone column wise finally, Going Q wise no more required as multimarked wont depend on it
			- except for qwise plots --> Now it'd be more logical colwise plots, no MCQ/INT distinguishing too.


Next:
	// Apply DFT and IFT on cropped one just to see.
		- IFT recovers image really well
		- low pass on dft produces blur!
		- fftshift rotates the image

	Increase test data
		Nope, unnecessary - 5.9  [ ] Images with more than 4 circles, Less than 4 ?!
		(23 Apr) 5.10 [X] Brightness and contrast variations

	Update DetailedDos.md
	Design the demo scenario now.
		Live Image : https://github.com/adityaarora1/LiveEdgeDetection
		Uploading : https://github.com/amitshekhariitbhu/Fast-Android-Networking


"Camera API in Android is hard. Having 2 different API for new and old Camera does not make things any easier. But fret not, that is your lucky day! After several years of working with Camera, we came up with Fotoapparat."


2 Apr:
// Refactor processOMR
	# better go question-wise than dictionary wise.
	->  but now Q object is removed. qNo, qType is present with points!
	Choices : 1. Make Q again, or
			  2. Store one more variable having qNo to qType mapping. _/ -> its already there = templJSON[squad]

// change json format to include keys for qNos and have its dims, instead of typing q9.1, q9.2, etc

7 Apr : Learning more of Android -
from LiveEdgeDetection app:
Structure:
	ScanConstants, ScanUtils don't have context as they are not instantiated. They contain static members, unlike the views such as ScanSurfaceView.
		--> marker image be loaded in ScanSurfaceView and passed wherever required to ScanUtils.

In the beginning, everyone used Environment.getExternalStorageDirectory(), which pointed to the root of external storage.
This led to external storage being just a big basket of random content.
Later, Google offered more organization:
getExternalFilesDir() and getExternalCacheDir() on Context, pointing to an application-specific directory on external storage,
one that would be deleted when the app is uninstalled

    //    Android/data/appname  --> context.getFilesDir or getExternalFilesDir (Any files that are private to the application)
    //    emulated/0/Downloads --> Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
    // another source:  Environment.getExternalStorageDirectory().getAbsolutePath() + "/Downloads"

_/ Using inbuilt image ==> res/drawable. Thus putting default marker there.

8 Apr :
progress written in app repo's changes-summary.txt

9 Apr:
App:
	// multiple permissions issue solved
	// X ray resolve attempted - works well only in emulator
	// Major Refactoring - renames, remove redundants,
10 Apr:
	// debug warpPerspective
	Low FPS ==> reduce image sizes!
		Camera issue on phone ==> Try CvCameraViewListener - gives Mat, allows, setMaxFrameSize, simplifies return preview ==> MUCH faster
	-> Need more refactoring:  simplify workflow

13 Apr:
	// Complete the UI
	// resolve cancel button threading issue
	// Autocapture shall now be scheduled after confirming 4 quads

	Refactoring
		refactor evaluation code
		rename variables to answer why they really exist

17 Apr:
Code result tabulation
Find final tuning parameters
	- KSIZES - blur, morph,
    - Canny parameters
    - 

After report: 
	Try canvas on the new app

Some excessDos that won't happen after BTP (do them b4 poster!)
	> those brandDos on wikipedia, SO, etc

Note: looking back at this app dev, I was right about thinking so much before starting to code - just that I should have thought in a prioritized way on the features when it came to the low fps issue. That would have saved much time, anyway learning about how canvas works was quite helpful and it will get used later once core feats are working well.
Also, had I got some real mobile camera images in test set instead of the unit_tests, I would have discovered few bugs quite earlier

From drive data: 
	// Nope, its a limitation - # TODO: (remove closeUp bool) Automate the case of close up scan(incorrect page)-

18 Apr:
	// Nailed this one in solving process- 
		Morph not working well on real data?!
		-> Image is almost like that in hist_eq example on opencv
		-> low exposure is there too? - apply gamma correction
		-> checkout CLAHE again _// Solved by combining it with norm & truncated threshold
			(25 apr) ^It was trunc threshold only that was found useful.

	// findPage finding hard to remove bg coz of stringy edges touching paper
		-> bg substractions : some are motion based - works only on still cameras
		-> Make use of obtained WHITE blob of paper?
		-> But anyway mobile's gonna give the cropped image _//


	 	// (5 hrs! But worth it - epic victory!) fix colwise thresholding now 
	 		-> Not at all, gray is still getting as low as 35! - CLAHE has distorted its working: Its good only for edge detection?!
	 		-> Note Assumption about cols: 
	 			Colwise background color is either gray or white, but not alternating
	 			The bubbles are relatively close to each other(less than a bubble-length away) 

        	_// -> global QVals shall absolutely help here. Just run same funtion on total QVals instead of colwise _//
    		-> q5C in HE_19 from 2044 is having issue with MIN_JUMP = 20 as well, rest having prob with MIN_JUMP = 30
	        	>> Only global needs two big jumps! "The point of colwise was to reduce it to the biggest jump!"
        	note: # DISCRETION: Pretty critical factor in reading response

	 	//: remove Normal/Xerox concept
	 	app 
			//: put quad templ matchin
				//> accept only on 4 present, but show "hold_still on page"
			// : limit contours *drawing* to top 3
			//: CLAHE into app

			// Higher resolutions!
				// From an SO answer, found this line: 
					if (!connectCamera(getWidth(), getHeight())) ...
					maxAllowedW
					D/custom: 1920x1080 : supported!
					D/custom: 1440x1080 : supported!
					D/custom: 1072x1072 : supported! _// but this is fine

			//: put Folder structure : naming ids

			/(*) make use of white-ness of paper: intensity threshold non-white --> gray.
			-> could actually simplify the process easily!
			-> Further #ad-hoc : manually enhanced images for easy thresholding [TODO: scope for ensuring quality inputs. #ad-hoc to OMR!!]

	 	Prepare corner case dataset
		 	result data
		 		/> that GIF of images being scanned fast and accurately
		 		> gif of image transitions with current dimensions.
		 		// > Cornercase examples for : 
		 		// 	SHOW FOCUS ON CORE ALGO - Xerox and Align
		 		// 	slight creases,
		 		// 	template shift(slight bend),
		 		// 	Rotation
		 		// 	Persp
		 		// 	Effect on Whitener : 
		 		// 	bad lighting(shadow) - LOW / HIGH
		 		// 	readResponse testing:  
		 		// 		allMarked - normal_HE_01 from 2044,
		 		 		noneMarked,
		 		 		noneMarked except one,
		 		> Steps image strip should contain
		 		 - input, ?!?!
		 		> Error cases:
		 			Bad marker
		 			Extremes of above cases

			tuning data
				> plot rectangle max/min cosines graph
				# TODO: Plot and see performance of scaleRange
ExcessDos:
	//flash functionality
	//put signature inside app
	put signature inside code
	a good logo!


21 Apr:
Final Testbench - do only feasible samples!
All marked revealed case of - even slightly vertically shifted will give large jump_gap here, thus giving 0 as marked!

22 Apr:
6:20 AM : 
7:21 AM ->
7:56 AM ->
	// > create the hstack files! and generate output for all the images.
		- hstack should contain:
			raw input, morphCannyContours, preprocessed, DetectedCircles, initial_align, final_align, final_marked overlay without align boxes
			initial, morph1, morph, morph_eroded
	//8:47 AM

9:01 AM -> More stacks

14:34 PM -> Divide it into two stacks

ExcessDos :
	Make and release that '3D viz'


24 Apr;
Finish Yellow marks first.
	// > put getThreshold algo
	// > align template images
	// > results section!

// Bring examples of warped perspectives

25 Apr
//(1 hr without momentum) Write that processOMR
// Keep flag for saveimglevel too. (to check speed)
	> No change of speed! Still at .68s/OMR
// Speed test:
	only getROI: 0.21s/OMR
	getROI + readResp : 0.68s/OMR

Note: just saw how much difference without vert-align it is! Roll at 276 was perfect for scanned OMRs, for mobile OMRs it became 285
_// This ought to mess up if not done calmly. Not now:
	Major change: Add vertical alignment too-
		Not nw; > One-go try for vertical alignment!!

// Presentable code running.

App :
// Increase accept rate!!
	// > recheck flow of filters: All fine. 
	// > recheck template matching: its time to add basic scaled matching!
	/.. > CLAHE isn't giving useful output! Do some tuning there.
		-> Guess its not useful for canny. It increases the noise

Focus on Demo video
	// Nope, no need, can handle.> Save app's edited output on which markers detected without
	// > Make code runnable on Backup PC
	// > Install on others' phone
	> Take less intense folding dataset
	> Make separate bending set 

// Do some corrections in report
UPDATE FINAL REPORT!!
-->	App screenshots into report.
		// > Find places to put : input Images, ...?
		// > Put only 3 screenshots of guidelines & save
		// > Don't mention tunablity
	Final result image numbers.
	Related: <add checkIt here>
	abstract: <refine here> <put xerox align etc here>
	intro: <reduce here; move to related works>
	<(after results) 4 Concise points>

	Into latex:
		> Blur the names in all images used.
		Report speed of checking : 
		Accuracy in bubble detection : 
		Failing cases : 

Good news!: 16 out of 20 real images worked well. wrong thresh in omr_8(legit); no circle in omr_11(partial skew - wrong page boundary),omr_19(partial skew),omr_21(blurry and skewed - legit),

YKWYD - Start studying for queueing b4 dinner

26 Apr
	//Poster latex start
	//Review report once, then Print
	
	//Poster contents:
		//Reduce sections : Introduction, Motivation
		// remove sections :  Literature review
		// Block Diagram
		//Methodology Overview
		//> Adaptive Threshold
		//> Align Template

		//> Attracting n Defining image of the project: 
			//1. Bad looking OMR(Bad print, persp, *folded OMR) working
			//2. Good looking OMR with shadow







LATER:
	App:
		Try canvas on the new app
		> Try again for higher fps without drawing on canvas

5 May:
	Restructuring code-
		//Remove high space taking commits
			>> Make new repo, take important commits from old repo, checkout and commit into new repo.
			//>> Insert gitignore right from start.
			Commits to keep:
			//98e93e9 BTP done
			//a1984cc stacks showing functions improved
			// b578a12 some alignment morphing tweaks
			//cea0d6c (finally) fixed colwise thresholding, pretty robust now
			//3c7357d CLAHE+Norm for real images
			//4994ab7 Full test passed for align + colwise threshold
			// 06ab128 Mk's trick for aligning
			//f1ea959 Proper Alignment Method implemented
			// 15e5b5e test bench v2 - perfection ocd
			//7f2b562 testbench foundation complete
			// e154bbe Per-image thresholding done
			// 3e5a199 public limits commited
			//ce36323 Local variance done
			// f9312d7 Template code, Subplots done
			// fa10dce rebased for advanced
			//9ae6a28 Normalized and Tested
			//d66b365 init

7 May(ctd):
		//Lesser directories
		// Remove excess code (kv, excess constants)
		// Sample Images
		Refactor:
			Follow block diagram for functions
			Lesser functions
			remove unnecc globals, locals
		Update readme:
			multiple case gifs
			link to app
			how to contribute
			> Readme update new TODOs 
				- identify if its a closeUp : check no contours & make use of whites!
				- from code comments
				- Point at juniors from coding club: Write as a guide to dive into this code easily
			How to use for your org:
				> Inputs description
				> Brand techno's marking scheme here. Show the how marking scheme json can handle it.
		
		make code independent of Squads?
