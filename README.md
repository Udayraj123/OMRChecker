# Note from Maintainer

Hello there! 👋

We're calling for contributions in Hacktoberfest 2022, Do check [here](https://github.com/Udayraj123/OMRChecker/issues) for issues to work on.
For the previous version of OMRChecker, visit [legacy code](https://github.com/Udayraj123/OMRChecker/tree/master-legacy).

# OMR Checker

Grade exams fast and accurately using a scanner 🖨 or your phone 🤳.

<!-- [![HitCount](http://hits.dwyl.io/udayraj123/OMRchecker.svg)](http://hits.dwyl.io/udayraj123/OMRchecker) -->

[![GitHub stars](https://img.shields.io/github/stars/Udayraj123/OMRChecker.svg?style=social&label=Stars✯)](https://GitHub.com/Udayraj123/OMRChecker/stargazers/)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/pull/new/master) <!-- [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/Udayraj123/OMRChecker/wiki/TODOs) -->
[![GitHub pull-requests closed](https://img.shields.io/github/issues-pr-closed/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/pulls?q=is%3Aclosed)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/issues?q=is%3Aissue+is%3Aclosed)

<!-- [![GitHub contributors](https://img.shields.io/github/contributors/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/graphs/contributors/) -->

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FUdayraj123%2FOMRChecker&count_bg=%2379C83D&title_bg=%23555555&icon=googlecast.svg&icon_color=%23E7E7E7&title=hits%28since+29+Sep+2022%29&edge_flat=false)](https://hits.seeyoufarm.com)
[![Join](https://img.shields.io/badge/Join-Discord_group-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)
[![Ask me](https://img.shields.io/badge/Discuss-on_Github-purple.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/issues/5)

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/Udayraj123/a125b1531c61cceed5f06994329cba66/omrchecker-on-cloud.ipynb) -->

#### **TLDR;** Jump to [Getting Started](#getting-started).

## 🎯 Features

A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. Support is also provided for a customisable marking scheme with section-wise marking, bonus questions, etc.

| Specs <img width=200/> | ![Current_Speed](https://img.shields.io/badge/Speed-200+_OMRs/min-blue.svg?style=flat-square) ![Min Resolution](https://img.shields.io/badge/Min_Resolution-640x480-blue.svg?style=flat-square) <img width=200/> |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 💯 **Accurate**        | Currently nearly 100% accurate on good quality document scans; and about 90% accurate on mobile images.                                                                                                          |
| 💪🏿 **Robust**          | Supports low resolution, xeroxed sheets. See [**Robustness**](https://github.com/Udayraj123/OMRChecker/wiki/Robustness) for more.                                                                                |
| ⏩ **Fast**            | Current processing speed without any optimization is 200 OMRs/minute.                                                                                                                                            |
| ✅ **Customizable**    | [Easily apply](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide) to custom OMR layouts, surveys, etc.                                                                                                    |
| 📊 **Visually Rich**   | [Get insights](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals) to configure and debug easily.                                                                                                        |
| 🎈 **Lightweight**     | Very minimal core code size.                                                                                                                                                                                     |
| 🏫 **Large Scale**     | Tested on a large scale at [Technothlon](https://en.wikipedia.org/wiki/Technothlon).                                                                                                                             |
| 👩🏿‍💻 **Dev Friendly**    | [Pylinted](http://pylint.pycqa.org/) and [Black formatted](https://github.com/psf/black) code. Also has a [developer community](https://discord.gg/qFv2Vqf) on discord.                                          |

Note: For solving interesting challenges, developers can checkout [**TODOs**](https://github.com/Udayraj123/OMRChecker/wiki/TODOs).

See complete guide and details at [Project Wiki](https://github.com/Udayraj123/OMRChecker/wiki/).

<!-- 💁🏿‍♂️ **User Friendly** - WIP, Help by contributing! -->

## 💡 What can OMRChecker do for me?

Once you configure the OMR layout, just throw images of the sheets at the software; and you'll get back the marked responses in an excel sheet!

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

## Visuals

### Processing steps

See step by step processing of any OMR sheet:

<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="rotation_stack" width="650" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/rotation.PNG">
	</a>
	<br>
	*Note: This image is generated by the code itself!*
</p>

### Output

Get a CSV sheet containing the detected responses and evaluated scores:

<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="csv_output" width="550" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/csv_output.PNG">
	</a>
</p>

#### There are many more visuals in the wiki. Check them out [here!](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals)

## Getting started

![Setup Time](https://img.shields.io/badge/Setup_Time-20_min-blue.svg)

**Operating system:** OSX or Linux is recommended although windows is also supported currently.

### 1. Install global dependencies

![opencv 4.0.0](https://img.shields.io/badge/opencv-4.0.0-blue.svg) ![python 3.4+](https://img.shields.io/badge/python-3.4+-blue.svg)

Install opencv (Any installation method is fine.)

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user opencv-python
python3 -m pip install --user opencv-contrib-python
```

More details on pip install openCV [here](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/).

**Installing missing libraries(if any):**

_Windows and MacOS users may skip this step._

On a fresh computer, some of the libraries may get missing in event after a successful pip install. Install them using following commands[(ref)](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/):

```bash
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran
```

### 2. Install project dependencies

<!-- ![imutils 0.5.2](https://img.shields.io/badge/imutils-0.5.2-blue.svg)  -->

![matplotlib 3.0.2](https://img.shields.io/badge/matplotlib-3.0.2-blue.svg) ![pandas 0.24.0](https://img.shields.io/badge/pandas-0.24.0-blue.svg) ![numpy 1.16.0](https://img.shields.io/badge/numpy-1.16.0-blue.svg) ![deepmerge 0.1.0](https://img.shields.io/badge/deepmerge-0.1.0-blue.svg) ![dotmap 1.3.17](https://img.shields.io/badge/dotmap-1.3.17-blue.svg)

#### Clone the repo

```bash
git clone https://github.com/Udayraj123/OMRChecker
cd OMRChecker/
```

#### Install pip requirements

```bash
python3 -m pip install --user -r requirements.txt
```

**Friendly Note:** If you face a distutils error in pip, use `--ignore-installed` flag in above command.

<!-- Wiki should not get cloned -->

### 3. Run the code

1. First copy and examine the sample data to know how to structure your inputs:
   ```bash
   cp -r ./samples/sample1 inputs/
   # Note: you may remove previous inputs (if any) with `mv inputs/* ~/.trash`
   # Change the number N in sampleN to see more examples
   ```
2. Run OMRChecker:
   ```bash
   python3 main.py
   ```

Each example in samples folder demonstrates different ways in which OMRChecker can be used.


_**Issues you might face while installing :**_
<details>
<summary>Issues</summary>
<br>
#1.<ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'> On Windows    after using the command: 
	
	
	$ python3 -m pip install --user -r requirements.txt
	
	
	
![P1](https://user-images.githubusercontent.com/111011580/193443625-04798d34-98ba-4d28-b3db-0975be936289.png)
	
	
 **Solution-->**
	
You can resolve it by following the steps : 

--> It can happen if you are not in the directory where you have cloned your project .
	

 1. Navigate to the directory where you have cloned your project .
 
Example -  I had made OMR Checker Directory to clone the project ,to navigate to the directory use cd <Name of Directory>
 
![cd](https://user-images.githubusercontent.com/111011580/193441708-b133dd86-c85b-4da0-b140-3c50a81abac8.png)


2. To look contents inside your directory use ls 

![ls](https://user-images.githubusercontent.com/111011580/193441759-024f0210-75ca-4bfd-9878-371b338c7745.png)
	

3. Now navigate to this folder using cd OMRChecker

![cd2](https://user-images.githubusercontent.com/111011580/193441808-9678ab0d-df51-434a-86d9-7c628c08be9e.png)
	

4. Now use command:  python3 -m pip install --user -r requirements.txt


</details>


## OMRChecker for custom OMR Sheets

1. First [create your own template.json](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide).
2. Configure the tuning parameters (help needed for documentation!).
<!-- 3. Add answer key( TODO: add answer key/marking scheme guide)  -->
3. Run OMRChecker with appropriate arguments (See full usage).

## Full Usage

```
python3 main.py [--setLayout] [--autoAlign] [--inputDir dir1] [--outputDir dir1]
```

Explanation for the arguments:

`--setLayout`: Set up OMR template layout - modify your json file and run again until the template is set.

`--autoAlign`: (experimental) Enables automatic template alignment - use if the scans show slight misalignments.

`--inputDir`: Specify an input directory.

`--outputDir`: Specify an output directory.

**Note:** The `--noCropping` flag has been replaced with including/excluding a 'CropPage' plugin in "preProcessors" of the template.json(see [samples](https://github.com/Udayraj123/OMRChecker/tree/master/samples)).

<!-- #### Testing the code
Datasets to test on :
Low Quality Dataset(For CV Based methods)) (1.5 GB)
Standard Quality Dataset(For ML Based methods) (3 GB)
High Quality Dataset(For custom processing) (6 GB)
-->

## FAQ

### Why is this software free?

This project was born out of a student-led organization called as [Technothlon](https://technothlon.techniche.org.in). It is a logic-based international school championship organized by students of IIT Guwahati. Being a non-profit organization, and after seeing it work fabulously at such a large scale we decided to share this tool with the world. The OMR checking processes still involves so much tediousness which we aim to reduce dramatically.

We believe in the power of open source! Currently, OMRChecker is in an intermediate stage where only developers can use it. We hope to see it become more user-friendly as well as robust from exposure to different inputs from you all!

[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

### Can I use this code in my work?

OMRChecker can be forked and modified. You are encouraged to play with it and we would love to see your own projects in action!

The only requirement is **disclose usage** of this software in your code.

It is published under the [GPLv3 license](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE).

## Credits

_A Huge thanks to :_
_**Adrian Rosebrock** for his exemplary blog :_ https://pyimagesearch.com

_**Harrison Kinsley** aka sentdex for his [video tutorials](https://www.youtube.com/watch?v=Z78zbnLlPUA&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq) and many other resources._

_**Satya Mallic** for his resourceful blog:_ https://www.learnopencv.com

_And to other amazing people from all over the globe who've made significant improvements in this project. Thank you!_

<!--
OpencV
matplotlib
some SO answers from roughworks
prof
-->

## Related Projects

Here's a sneak peak of the [Android OMR Helper App (WIP)](https://github.com/Udayraj123/AndroidOMRHelper):

<p align="center">
	<a href="https://github.com/Udayraj123/AndroidOMRHelper">
		<img height="350" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/app_flow.PNG">
	</a>
</p>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/Udayraj123/OMRChecker.svg)](https://starchart.cc/Udayraj123/OMRChecker)

### Ways to contribute:

- Help OMRChecker reach more people by giving a star!

  The Goal is to reach top position for the [OMR Topic](https://github.com/topics/omr)

- If this project saved you large costs on OMR Software licenses, or saved efforts to make one. Consider donating an amount of your choice below:

  [**Buy Me A Coffee ☕**](https://www.buymeacoffee.com/Udayraj123)

  [![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_LG.gif)](https://www.paypal.me/Udayraj123/500)

- Join the [Discord](https://discord.gg/qFv2Vqf) community to help others with using OMRChecker

<!-- ![☕](https://miro.medium.com/fit/c/256/256/1*br7aoq_JVfxeg73x5tF_Sw.png) -->
<!-- [![paypal.me](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Z5BNNK7AVFVH8&source=url) -->
<!-- https://www.amazon.in/hz/wishlist/ls/3V0TDQBI3T8IL -->

### License

[![GitHub license](https://img.shields.io/github/license/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE)

For more details see [LICENSE](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE).

_Find OMRChecker on_ [**_Product Hunt_**](https://www.producthunt.com/posts/omr-checker/) **|** [**_Reddit_**](https://www.reddit.com/r/computervision/comments/ccbj6f/omrchecker_grade_exams_using_python_and_opencv/) **|** [**Discord**](https://discord.gg/qFv2Vqf)

 <!-- [***Hacker News***](https://news.ycombinator.com/item?id=20420602) **|** -->
 <!-- **|** [***Swyya***](https://www.swyya.com/projects/omrchecker) -->
