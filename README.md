# OMR Checker
Grade exams fast and accurately using a scanner 🖨 or your phone 🤳. 

![Accurate](https://img.shields.io/badge/Accurate-✔-green.svg?style=flat-square) 
![Robust](https://img.shields.io/badge/Robust-✔-green.svg?style=flat-square) 
![Fast](https://img.shields.io/badge/Fast-✔-green.svg?style=flat-square) 
![Lightweight](https://img.shields.io/badge/Lightweight-✔-green.svg?style=flat-square)
![Extensible](https://img.shields.io/badge/Extensible-✔-green.svg?style=flat-square)
![Large_Scale](https://img.shields.io/badge/Large_Scale-✔-green.svg?style=flat-square)

![Current_Speed](https://img.shields.io/badge/Speed-200_OMRs/m-brightgreen.svg?style=flat-square)
![Current_Size](https://img.shields.io/badge/Size-500KB-brightgreen.svg?style=flat-square) 
![Min Resolution](https://img.shields.io/badge/Min_Resolution-640x480-brightgreen.svg?style=flat-square) 

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/pull/new/master)
[![GitHub stars](https://img.shields.io/github/stars/udayraj123/OMRChecker.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/udayraj123/OMRChecker/stargazers/)
<!-- gif here -->
![document_scanner](https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/document_scanner.gif)

![checking_xeroxed_mobile](https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/checking_xeroxed_mobile.gif)

## 🎯 Features
A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. Support is also provided for a customisable marking scheme with section-wise marking, bonus questions, etc.

💯 **Accurate** - Currently nearly 100% accurate on good quality document scans; and about 90% accurate on mobile images.

💪🏿 **Robust** - Supports low resolution, xeroxed sheets. See [Robustness Wiki](https://github.com/Udayraj123/OMRChecker/wiki/Robustness.md)

⏩ **Fast** Current processing speed without any optimization is 200 OMRs/minute.

✅ **Extensible** - [Easily apply](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide) to different OMR layouts, surveys, etc.

📊 **[Visually Rich Outputs](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals)** - get insights to configure and debug easily.

🎈 **Lightweight** - Code size is less than 500 KB.

🏫 **Large Scale** - Already used on tens of thousands of OMRs at Technothlon([Fb Link](https://www.facebook.com/technothlon.techniche)).

<!-- 📄 **Simple Structure** - inputs and outputs style.  -->

👩🏿‍💻 **Dev Friendly** - [Well documented](https://github.com/Udayraj123/OMRChecker/wiki/) repository based on python and openCV.

<!-- 💁🏿‍♂️ **User Friendly** - WIP, Help by contributing! -->
See more at [Project Wiki](https://github.com/Udayraj123/OMRChecker/wiki/).

## 💡 What can OMRChecker do for me?
Just configure for your OMR layout once, then throw images of the sheets at it, you'll get back the graded responses in an excel sheet! 

Images can be taken from various angles as shown below-
![sample_input](https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/sample_input.PNG)
These images will be processed in the following manner: 
![rotation_stack](https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/rotation.PNG)

#### There is a lot more visuals [in the wiki](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals.md). Check them out!

### 🎯 Why is this software free?
Our Motto: 
> Don't reinvent the wheel, use good wheels to make great vehicles! 

After seeing it work fabulously at large scale on scanned OMR sheets at Technothlon, we have decided to open source the code and roll out mobile based scanning as well. The feedback from you all will be extremely valuable in making this idea become successful.

#### [![GitHub stars](https://img.shields.io/github/stars/udayraj123/OMRChecker.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/udayraj123/OMRChecker/stargazers/) ⭐ Help us reach 550 stars to become #1 on the "OMR" tag on github - [Currently #5](https://github.com/topics/omr)

### Activity: 
[![Chat](https://img.shields.io/badge/Chat-on_Discord-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)
[![Ask me](https://img.shields.io/badge/Ask_me-anything-purple.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/issues/5)

[![HitCount](http://hits.dwyl.io/udayraj123/OMRchecker.svg)](http://hits.dwyl.io/udayraj123/OMRchecker)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/udayraj123/OMRChecker/graphs/commit-activity)
[![GitHub pull-requests closed](https://img.shields.io/github/issues-pr-closed/udayraj123/OMRChecker.svg)](https://GitHub.com/udayraj123/OMRChecker/pull/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/udayraj123/OMRChecker.svg)](https://GitHub.com/udayraj123/OMRChecker/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub contributors](https://img.shields.io/github/contributors/udayraj123/OMRChecker.svg)](https://GitHub.com/udayraj123/OMRChecker/graphs/contributors/)


## Getting started ![Setup Time](https://img.shields.io/badge/Setup_Time-20_min-blue.svg)

#### Operating System
Although windows is supported, **Linux** is recommended for bug-free updates.

#### 1. Install dependencies 
![opencv 4.0.0](https://img.shields.io/badge/opencv-4.0.0-blue.svg) ![python 3.4](https://img.shields.io/badge/python-3.4-blue.svg)

```
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install opencv-python
sudo python3 -m pip install opencv-contrib-python
```
_Windows users replace 'sudo python3' with 'python'._ More details here: https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/ 

#### 2. Clone the repo
```
git clone https://github.com/udayraj123/OMRChecker
```

##### Install other requirements 
![imutils 0.5.2](https://img.shields.io/badge/imutils-0.5.2-blue.svg) ![matplotlib 3.0.2](https://img.shields.io/badge/matplotlib-3.0.2-blue.svg) ![pandas 0.24.0](https://img.shields.io/badge/pandas-0.24.0-blue.svg) ![numpy 1.16.0](https://img.shields.io/badge/numpy-1.16.0-blue.svg)

```
cd OMRChecker/
sudo python3 -m pip install -r requirements.txt
```
> **Note:** If you face an distutils error, use the `--ignore-installed` flag 
>	sudo python3 -m pip install --ignore-installed -r requirements.txt

<!-- Wiki should not get cloned -->
#### 3. Run the code
1. Make a copy of 'OMR_Files_sample' and **rename to 'OMR_Files'** (Do not make changes to other folder names)
2. Run OMRChecker: **` python3 main.py `**
3. Do a Smirk :smirk: :smirk: :smirk:
4. [Profit](https://knowyourmeme.com/memes/profit)!!

**General syntax:**

`python3 main.py [--noMarkers] [--setLayout] [--noCrop] [--noAlign]` 

Explanation for the arguments:

`--setLayout` : To setup template layout interactively(for custom OMRs). See Walkthrough [here](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide).

`--noMarker` : If the images do not have a marker.

`--noAlign` : If the images are captured without much bending of paper.
<!-- explanatory image here -->
`--noCrop` : If the images are using a document scanner or do not need cropping page boundary.

Note: Make sure the `outputs` folder is clean if you don't want to append to previous results.

<!-- ### Folder Structure 
<img align="center" src="https://raw.githubusercontent.com/udayraj123/OMRChecker/master/directory_structure.png" alt="Directory Structure" height="350">

This structure has been created to suit for better organization of OMRs (Citywise then Group-wise and Language-wise). Making changes to this would require changes in the code.
 -->
### Configuring for your own OMR Sheets
<!-- Template alignment image here -->
Follow the detailed [walkthrough in wiki](https://github.com/Udayraj123/OMRChecker/wiki/Home/).
	
<!-- #### Testing the code
Datasets to test on : 
Low Quality Dataset(For CV Based methods)) (1.5 GB)
Standard Quality Dataset(For ML Based methods) (3 GB)
High Quality Dataset(For custom processing) (6 GB) 
-->

<!-- Begin donate section -->
To keep my 💡 brain juices flowing and create more such projects, [Buy Me A Coffee ☕](https://www.buymeacoffee.com/udayraj123) 

If this project saved you large costs on OMR Software licenses and want to give me some credit on those savings-
paypal me here:  [![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_LG.gif)](https://www.paypal.me/udayraj123)

<!-- ![☕](https://miro.medium.com/fit/c/256/256/1*br7aoq_JVfxeg73x5tF_Sw.png) -->
<!-- [![paypal.me](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Z5BNNK7AVFVH8&source=url) -->
<!-- https://www.amazon.in/hz/wishlist/ls/3V0TDQBI3T8IL -->
<!-- End donate section -->

### Project Progress 
#### Current Goals Progress : 🥅__25__⚽️_____75______ 🏃‍♂️
**Note:** Interesting tasks from Beginner to Advanced level are there to solve. Looking for computer vision enthusiasts to take it to the next level!

#### Some of the ToDos: 
<!-- Add immediate TODOs here -->
**Legend:** 🏳 Beginner Quests	|	🚩 Intermediate Challenges	|	🏁 Advanced Experiments

🏳 Run code and tweak with configuration to find optimal values(use showimglvl for feedback)

🏳 Add simple validation checks for configuration

🏳 Use cv2.putText to add appropriate labels to all output images

🏳 Implement Accuracy evaluation 

🏳 Contribute to Mobile images dataset using Android App(Contact on discord)

<!-- 
🏳 	Add coloroma to output texts

🏳 Generate your own gifs from mini_scripts

🏳 Make a demo video of code your OMRs

🏳 Contribute to Wiki for your images

🏳 Review and Compare with Existing OMR Software
 -->
🚩 Auto Rotating the OMR sheet

🚩 Generate Template Layout directly from blank OMR image.

🚩 Refactor code to a software [design pattern](https://refactoring.guru/design-patterns/python)

<!-- 
🚩 Add more details about image in the Visuals

🚩 Making a Testing Benchmark ([Dataset coming very soon](https://drive.google.com/drive/folders/16Hlvv6D-25AlNXC65_vrsk-P4kVu7VKb?usp=sharing )!) 
-->

🏁 Faster Speeds : parallelization, pyrDowns, etc
<!-- adding native cpp calls -->
<!-- 🏁 Make more visualizations (Ideas for Flow diagrams, Animations and 3D outputs given in wiki) -->

🏁 More Robustness : explore methods to find global threshold more accurately

🏁 Introduce git submodules 

Get Full ToDo List in the [Kanban Board(w.i.p)](https://github.com/Udayraj123/OMRChecker/projects/1).

### ❓ Can I use this code in my work?
OMRChecker is [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/) and published under **GPLv3** license which is just to give a heads up to disclose usage of this software in your code. **OMRChecker can be forked and modified. You are encouraged to play with it and we would love to see your own projects in action!**

<!-- ### Credits 
Adrian
Satya
Sentdex
Some papers
Team Techno
-->

### License 
```
Copyright © 2019 Udayraj Deshmukh
OMRChecker : Grade exams fast and accurately using a scanner 🖨 or your phone 🤳
This is free software, and you are welcome to redistribute it under certain conditions;
```
For more details see [![GitHub license](https://img.shields.io/github/license/udayraj123/OMRChecker.svg)](https://github.com/udayraj123/OMRChecker/blob/master/LICENSE)

### Related Projects
[Android OMR Helper App(WIP)](https://github.com/Udayraj123/AndroidOMRHelper)

Here's a sneak peak into the app : 
![app_flow](https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/app_flow.PNG)
