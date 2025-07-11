Interesting tasks available in the whole range from Beginner to Advanced level. 
Looking for computer vision enthusiasts to take this project to the next level!

## Project Progress 
#### Current Goals Progress : 
### 🔲🔲🔲🔲🔲⏹⏹⏹⏹⏹
<!-- ### 🔵🔵🔵🔵🔵🔵⚪️⚪️⚪️ -->
<!-- ### 🥅__25__⚽️_____75______ 🏃‍♂️ -->

## Legend:
📝 Beginner Quests	

🏆 Intermediate Challenges

🎓 Advanced Experiments

The tasks are ordered in increasing levels of difficulty below.

## ToDos:

📝 Run code on your own images and add it to [samples](#)
<!-- Link to - How to add a sample folder -->

📝 Generate your own code-in-action gifs on those images 
<!-- Link to - mini_scripts -->

📝 Use cv2.putText to add appropriate labels to all output images
<!-- Give one example and steps here -->

📝 Add validation checks for configuration
<!-- PRELIM_CHECKS : if blank OMR available, do test on it too-->

📝 Implement Accuracy evaluation 
> There are 5-6 popular methods of evaluation available. We will be implementing all of them. See [this link](https://www.wikiwand.com/en/Multi-label_classification#/Statistics_and_evaluation_metrics).
> From above link, "Exact match" method is implemented in main.py (look for 'TEST_FILE').
> Need help in implementing any of the remaining methods.
> The ultimate plan is to create a reliable benchmark score for evaluating future algorithms.
> For any discussion/doubts, ask on [discord](https://discord.gg/HKw6juP).

<!-- from Coco: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py -->

📝 Contribute to Mobile images dataset using the [Android App](https://github.com/Udayraj123/AndroidOMRHelper)(Contact on [discord](https://discord.gg/HKw6juP)) 
<!-- > Show image for- Make separate bending set  -->

📝 	Add coloroma to output texts
<!-- link to colorama -->
<!-- Add colored bg terminal output -->

📝 	Product comparision articles: 
	Review and Compare with Existing OMR Softwares with this free software. For a start, see [Auto Multiple Choices](https://www.auto-multiple-choice.net/)
<!-- List of existing omr softwares -->

<!-- 📝 Suggest a good logo! -->

<!-- 🏆 [ONGOING] Running the code on Colab -->

🏆 Generate Template Layout directly from blank OMR image.
> Use methods like [morphology](./extras/Progress/2019-04-26/images/align_correct.PNG) and then blob detection to find presence of bubbles in a good quality image of a blank OMR Sheet like [this one](./extras/Original%20OMRs/OMR_JUNIORS/OMR_JUNIORS_front.jpg).
> Illustrative image coming soon.

<!-- Suggest blob detection morph outputs here -->
<!-- GUI guided better? future excess to think -->

🏆 Auto Rotating the OMR sheet

🏆 Calculate confidence using data
> Count times when only localTHR was used vs globalTHR over localTHR.
> Image contrast levels score.

🏆 Extract code snippets which may be re-usable in other projects and add them to gist

<!-- 🏆  Identifying if input is a closeUp : check no contours & make use of whites! -->
 
🏆 Put explanatory text on output images
> Show QBlock Labels and Column Orientations for template in the setLayout mode

<!-- 🏆 Making a Testing Benchmark ([Dataset coming very soon](https://drive.google.com/drive/folders/16Hlvv6D-25AlNXC65_vrsk-P4kVu7VKb?usp=sharing )!)  -->

🏆 Improve page boundary detection : Defeat the [bossbg.jpg](./extras/Test/Backgrounds/bossbg.jpg)

🎓 Refactor code to a software [design pattern](https://refactoring.guru/design-patterns/python)
<!-- 
Refactoring
	refactor evaluation code
	rename variables to answer why they really exist 
	Follow block diagram for functions
	Lesser functions
	remove unnecc globals, locals
-->

🎓 Faster Speeds : parallelization, pyrDowns, etc
<!-- better template matching -->
<!-- adding native cpp calls -->

🎓 Making more visualizations from available data. See [Visualization Ideas](#) for ideas on Flow diagrams, Animations and 3D outputs.
<!-- > animation frames for warped persp, markers match moving -->
<!-- []> Alignment gif!! (use mobile images) -->

<!-- > add wordcloud from related research papers contents -->
<!-- > Marker scale variation plot to justify ScaleRange -->
<!-- > Show flow diagram for File moving patterns -->
<!-- >> the all mean threshold hist barplot - highlight which cols are marked which are not. -->
<!-- >> Instead of csv, make Excel sheet output with color coding  -->

🎓 Explore methods to find global threshold more accurately(perhaps using ML)
<!-- 
>> mini AI: Train to give you correct threshold based on histogram array!!
			mini hovers of adaptive threshold plots in the template overlay image
			Marker-Manual cases : Add Marker guidance as first manual, those that still are errs will go into the guided manual.
			Who's saying you can have only one go at the whole data?
				-> Figure out ways to get suggestions for fine tuning based on your data.
					>> Especially on column alignment
					>> Make use of 123456789 type multimarks here
> Generate data for training to match the marker coords! (see test_translate.output)
 
 -->
<!-- 🎓 For r/dataisbeautiful : 3D viz of the images templateMatch output(, Sobel eroded blobs)to see the peaks in morph output as well as qStrips -->

<!-- 🎓 Auto Alignment horizontally based on col_orient -->

🎓 Introduce git submodules 

<!-- Get Full ToDo List in the [Kanban Board(w.i.p)](https://github.com/Udayraj123/OMRChecker/projects/1). -->