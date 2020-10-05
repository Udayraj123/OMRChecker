# OMR Checker
Grade exams fast and accurately using a scanner 🖨 or your phone 🤳. 

[![HitCount](http://hits.dwyl.io/udayraj123/OMRchecker.svg)](http://hits.dwyl.io/udayraj123/OMRchecker)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/Udayraj123/OMRChecker/wiki/TODOs)
[![GitHub pull-requests closed](https://img.shields.io/github/issues-pr-closed/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/pulls?q=is%3Aclosed)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub contributors](https://img.shields.io/github/contributors/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/graphs/contributors/)

[![GitHub stars](https://img.shields.io/github/stars/Udayraj123/OMRChecker.svg?style=social&label=Stars✯)](https://GitHub.com/Udayraj123/OMRChecker/stargazers/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/pull/new/master)
[![Join](https://img.shields.io/badge/Join-Discord_group-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)
[![Ask me](https://img.shields.io/badge/Discuss-on_Github-purple.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/issues/5)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/Udayraj123/a125b1531c61cceed5f06994329cba66/omrchecker-on-cloud.ipynb) -->

#### **TLDR;** Jump to [Getting Started](#getting-started).


## 🎯 Features

A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. Support is also provided for a customisable marking scheme with section-wise marking, bonus questions, etc.

|       Specs      | ![Current_Speed](https://img.shields.io/badge/Speed-200_OMRs/m-blue.svg?style=flat-square) ![Current_Size](https://img.shields.io/badge/Code_Size-500KB-blue.svg?style=flat-square)  ![Min Resolution](https://img.shields.io/badge/Min_Resolution-640x480-blue.svg?style=flat-square) |
|:----------------:|-----------------------------------------------------------------------------------------------------------------------------------|
|  💯 **Accurate**  | Currently nearly 100% accurate on good quality document scans; and about 90% accurate on mobile images.                           |
| 💪🏿 **Robust**    | Supports low resolution, xeroxed sheets. See [**Robustness**](https://github.com/Udayraj123/OMRChecker/wiki/Robustness) for more. |
| ⏩ **Fast**       | Current processing speed without any optimization is 200 OMRs/minute.                                                             |
| ✅ **Extensible** | [**Easily apply**](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide) to different OMR layouts, surveys, etc.              |
| 📊 **Visually Rich Outputs** | [Get insights](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals) to configure and debug easily.              |
| 🎈 **Extremely lightweight** |  Core code size is **less than 500 KB**(Samples excluded).              |
| 🏫 **Large Scale** |  Used on tens of thousands of OMRs at [Technothlon](https://www.facebook.com/technothlon.techniche).              |
| 👩🏿‍💻 **Dev Friendly** |  [**Well documented**](https://github.com/Udayraj123/OMRChecker/wiki/) repository based on python and openCV with [an active discussion group](https://discord.gg/qFv2Vqf).              | 

Note: For solving live challenges, developers can checkout [**TODOs**](https://github.com/Udayraj123/OMRChecker/wiki/TODOs). 
See all details in [Project Wiki](https://github.com/Udayraj123/OMRChecker/wiki/).

<!-- 💁🏿‍♂️ **User Friendly** - WIP, Help by contributing! -->
## 💡 What can OMRChecker do for me?
Once you configure the OMR layout, just throw images of the sheets at the software; and you'll get back the graded responses in an excel sheet! 

Images can be taken from various angles as shown below-
<p align="center">
	<img alt="sample_input" width="400" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/sample_input.PNG">
</p>

### Code in action on images taken by scanner: 
<p align="center">
	<img alt="document_scanner" height="300" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/document_scanner.gif">

</p>

### Code in action on images taken by a mobile phone: 
<p align="center">
	<img alt="checking_xeroxed_mobile" height="300" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/checking_xeroxed_mobile.gif">
</p>

See step by step processing of any OMR sheet:
<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="rotation_stack" width="650" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/rotation.PNG">
	</a>
	<br>
	*Note: This image is generated by the code itself!*
</p>

Output: A CSV sheet containing the detected responses and evaluated scores:

<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="csv_output" width="550" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/csv_output.PNG">
	</a>
</p>

#### There are many visuals in the wiki. [Check them out!](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals)

## Getting started
![Setup Time](https://img.shields.io/badge/Setup_Time-20_min-blue.svg)

### Operating System
Although windows is supported, **Linux** is recommended for a bug-free experience.

### 1. Install dependencies 
![opencv 4.0.0](https://img.shields.io/badge/opencv-4.0.0-blue.svg) ![python 3.4](https://img.shields.io/badge/python-3.4-blue.svg)

_Note: To get a copy button for below commands, use [CodeCopy Chrome](https://chrome.google.com/webstore/detail/codecopy/fkbfebkcoelajmhanocgppanfoojcdmg) | [CodeCopy Firefox](https://addons.mozilla.org/en-US/firefox/addon/codecopy/)._
```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user opencv-python
python3 -m pip install --user opencv-contrib-python
```
More details on pip install openCV [here](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/).

> **Note:** On a fresh computer some of the libraries may get missing in above pip install. 

Install them using the [following commands](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/):
Windows users may skip this step.
```bash
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran
```

### 2. Clone the repo
```bash
# Shallow clone - takes latest code with minimal size
git clone https://github.com/Udayraj123/OMRChecker --depth=1
```
Note: Contributors should take a full clone(without the --depth flag).

#### Install other requirements 
![imutils 0.5.2](https://img.shields.io/badge/imutils-0.5.2-blue.svg) ![matplotlib 3.0.2](https://img.shields.io/badge/matplotlib-3.0.2-blue.svg) ![pandas 0.24.0](https://img.shields.io/badge/pandas-0.24.0-blue.svg) ![numpy 1.16.0](https://img.shields.io/badge/numpy-1.16.0-blue.svg)

```bash
cd OMRChecker/
python3 -m pip install --user -r requirements.txt
```
> **Note:** If you face a distutils error, use `--ignore-installed` flag in above command.

<!-- Wiki should not get cloned -->
### 3. Run the code

1. Put your data in inputs folder. You can copy sample data as shown below: 
	```bash
	# Note: you may remove previous inputs if any with `mv inputs/* ~/.trash` 
	cp -r ./samples/sample1 inputs/
	```
	_Note: Change the number N in sampleN to see more examples_
2. Run OMRChecker: 
	**` python3 main.py `**

These samples demonstrate different ways OMRChecker can be used.

#### Running it on your own OMR Sheets

1. First [create your own template.json](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide).
2. Open `globals.py` and check the tuning parameters.
<!-- 3. Add answer key( TODO: add answer key/marking scheme guide)  -->
3. Run OMRChecker with appropriate arguments.
	#### Full Usage 
	```
	python3 main.py [--setLayout] [--noCropping] [--autoAlign] [--inputDir dir1] [--outputDir dir1] [--template path/to/template.json]
	```
	Explanation for the arguments:

	`--setLayout`: Set up OMR template layout - modify your json file and run again until the template is set.

	`--autoAlign`: (experimental) Enables automatic template alignment - use if the scans show slight misalignments.

	`--noCropping`: Disables page contour detection - used when page boundary is not visible e.g. document scanner.

	`--inputDir`: Specify an input directory.

	`--outputDir`: Specify an output directory.

	`--template`: Specify a default template if no template file in input directories.

<!-- #### Testing the code
Datasets to test on : 
Low Quality Dataset(For CV Based methods)) (1.5 GB)
Standard Quality Dataset(For ML Based methods) (3 GB)
High Quality Dataset(For custom processing) (6 GB) 
-->

## 💡 Why is this software free?

The idea for this project began at Technothlon, which is a non-profit international school championship. After seeing it work fabulously at such a large scale, we decided to share this simple and powerful tool with the world to perhaps help revamp OMR checking processes and help greatly reduce the tediousness of the work involved.

And we believe in the power of open source! Currently, OMRChecker is in its initial stage where only developers can use it. We hope to see it become more user-friendly and even more robust with exposure to different inputs from you! 

[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

### Can I use this code in my work?

OMRChecker can be forked and modified. **You are encouraged to play with it and we would love to see your own projects in action!** The only requirement is **disclose usage** of this software in your code. It is published under the [**GPLv3 license**](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE) 

## Credits 
_A Huge thanks to :_
_The creative master **Adrian Rosebrock** for his blog :_ https://pyimagesearch.com 

_The legendary **Harrison** aka sentdex for his [video tutorials](https://www.youtube.com/watch?v=Z78zbnLlPUA&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq)._

_And the james bond of computer vision **Satya Mallic** for his blog:_ https://www.learnopencv.com

_And many other amazing people over the globe without whom this project would never have completed. Thank you!_

> _This project is dedicated to [Technothlon](https://www.facebook.com/technothlon.techniche) where the idea of making such solution was conceived. Technothlon is a logic-based examination organized by students of IIT Guwahati._

<!-- 
OpencV
matplotlib
some SO answers from roughworks
prof
-->

## License 
```
Copyright © 2019 Udayraj Deshmukh
OMRChecker : Grade exams fast and accurately using a scanner 🖨 or your phone 🤳
This is free software, and you are welcome to redistribute it under certain conditions;
```
For more details see [![GitHub license](https://img.shields.io/github/license/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE)

## Related Projects
Here's a sneak peak of the [Android OMR Helper App(WIP)](https://github.com/Udayraj123/AndroidOMRHelper): 
<p align="center">
	<a href="https://github.com/Udayraj123/AndroidOMRHelper">
		<img height="350" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/app_flow.PNG">
	</a>
</p>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/Udayraj123/OMRChecker.svg)](https://starchart.cc/Udayraj123/OMRChecker)
      
<!-- Begin donate section -->
### Other ways you can contribute:
- Help OMRChecker cross 750 stars ⭐ to become #1 ([Currently #3](https://github.com/topics/omr)). 
Current stars: [![GitHub stars](https://img.shields.io/github/stars/Udayraj123/OMRChecker.svg?style=social&label=Stars✯)](https://GitHub.com/Udayraj123/OMRChecker/stargazers/)

- [Buy Me A Coffee ☕](https://www.buymeacoffee.com/Udayraj123) - To keep my brain juices flowing and help me create more such projects 💡 

- If this project saved you large costs on OMR Software licenses, or saved efforts to make one. [![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_LG.gif)](https://www.paypal.me/Udayraj123/500)

<!-- ![☕](https://miro.medium.com/fit/c/256/256/1*br7aoq_JVfxeg73x5tF_Sw.png) -->
<!-- [![paypal.me](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Z5BNNK7AVFVH8&source=url) -->
<!-- https://www.amazon.in/hz/wishlist/ls/3V0TDQBI3T8IL -->

<!-- End donate section -->
*Find OMRChecker on* [***Product Hunt***](https://www.producthunt.com/posts/omr-checker/) **|** [***Hacker News***](https://news.ycombinator.com/item?id=20420602) **|** [***Reddit***](https://www.reddit.com/r/computervision/comments/ccbj6f/omrchecker_grade_exams_using_python_and_opencv/) **|** [***Swyya***](https://www.swyya.com/projects/omrchecker) **|** [![Join](https://img.shields.io/badge/Join-on_Discord-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)
