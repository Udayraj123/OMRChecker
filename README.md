 OMR Checker
Grade exams fast and accurately using a scanner 🖨 or your phone 🤳. 

![Accurate](https://img.shields.io/badge/Accurate-✔-green.svg?style=flat-square) 
![Robust](https://img.shields.io/badge/Robust-✔-green.svg?style=flat-square)![Large_Scale](https://img.shields.io/badge/Large_Scale-✔-green.svg?style=flat-square) 
![Fast](https://img.shields.io/badge/Fast-✔-green.svg?style=flat-square) 
![Lightweight](https://img.shields.io/badge/Lightweight-✔-green.svg?style=flat-square)
![Extensible](https://img.shields.io/badge/Extensible-✔-green.svg?style=flat-square)


[![UserGuide](https://img.shields.io/badge/UserGuide-✔-green.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/wiki/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/pull/new/master)
[![GitHub stars](https://img.shields.io/github/stars/Udayraj123/OMRChecker.svg?style=social&label=Stars✯)](https://GitHub.com/Udayraj123/OMRChecker/stargazers/)

#### **TLDR;** Jump to [Getting Started](#getting-started).

<!-- gif here -->
### Code in action on images taken by scanner: 🔥
<p align="center">
	<img alt="document_scanner" width="450" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/document_scanner.gif">

</p>

### Code in action on images taken by a mobile phone: 🔥
<p align="center">
	<img alt="checking_xeroxed_mobile" width="450" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/mini_scripts/outputs/gif/checking_xeroxed_mobile.gif">
</p>

## 🎯 Features

![Current_Speed](https://img.shields.io/badge/Speed-200_OMRs/m-brightgreen.svg?style=flat-square)
![Current_Size](https://img.shields.io/badge/Code_Size-500KB-brightgreen.svg?style=flat-square) 
![Min Resolution](https://img.shields.io/badge/Min_Resolution-640x480-brightgreen.svg?style=flat-square) 

A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. Support is also provided for a customisable marking scheme with section-wise marking, bonus questions, etc.

💯 **Accurate** - Currently nearly 100% accurate on good quality document scans; and about 90% accurate on mobile images.

💪🏿 **Robust** - Supports low resolution, xeroxed sheets. See [**Robustness**](https://github.com/Udayraj123/OMRChecker/wiki/Robustness) for more.

⏩ **Fast** Current processing speed without any optimization is 200 OMRs/minute.

✅ **Extensible** - [**Easily apply**](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide) to different OMR layouts, surveys, etc.

📊 **Visually Rich Outputs** - [get insights](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals) to configure and debug easily.

🎈 **Lightweight** - Core code size is **less than 500 KB**(Samples excluded).

🏫 **Large Scale** - Already used on tens of thousands of OMRs at [Technothlon](https://www.facebook.com/technothlon.techniche).

<!-- 📄 **Simple Structure** - inputs and outputs style.  -->

👩🏿‍💻 **Dev Friendly** - [**Well documented**](https://github.com/Udayraj123/OMRChecker/wiki/) repository based on python and openCV. 

Devs can look at [**TODOs**](https://github.com/Udayraj123/OMRChecker/wiki/TODOs) for live challenges. See more at [Project Wiki](https://github.com/Udayraj123/OMRChecker/wiki/).

For any help: [![Join](https://img.shields.io/badge/Join-on_Discord-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)

<!-- 💁🏿‍♂️ **User Friendly** - WIP, Help by contributing! -->


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
git clone https://github.com/Udayraj123/OMRChecker
```

#### Install other requirements 
![imutils 0.5.2](https://img.shields.io/badge/imutils-0.5.2-blue.svg) ![matplotlib 3.0.2](https://img.shields.io/badge/matplotlib-3.0.2-blue.svg) ![pandas 0.24.0](https://img.shields.io/badge/pandas-0.24.0-blue.svg) ![numpy 1.16.0](https://img.shields.io/badge/numpy-1.16.0-blue.svg)

```bash
cd OMRChecker/
python3 -m pip install --user -r requirements.txt
```
> **Note:** If you face a distutils error, use the `--ignore-installed` flag 

>	sudo python3 -m pip install --ignore-installed -r requirements.txt

<!-- Wiki should not get cloned -->
### 3. Run the code(on sample images)
<!-- (Do not make changes to other folder names) -->
1. Copy sample data: 
	```bash
	cp -r inputs/samples/sample1/* inputs/
	```
	**Note:** If you are not running this for the first time, delete previous inputs first
	```bash
	mv inputs/OMR_Files/ ~/.trash # delete permanently with: rm -rf inputs/OMR_Files/
	cp -r inputs/samples/sample1/* inputs/
	```
2. Run OMRChecker: **` python3 main.py `**
3. ???
4. [Profit](https://knowyourmeme.com/memes/profit)!!

_**Note:** For subsequent runs, if you don't want the outputs to append to previous results- 
Make sure the `outputs` folder is clean : `rm -rf outputs/`._

<!-- ### Folder Structure 
<img align="center" src="https://raw.githubusercontent.com/Udayraj123/OMRChecker/master/directory_structure.png" alt="Directory Structure" height="350">

This structure has been created to suit for better organization of OMRs (Citywise then Group-wise and Language-wise). Making changes to this would require changes in the code.
-->
## Applying it on your own OMR Sheets
<!-- Template alignment image here -->
1. Follow the [User Guide](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide/) to create your own Template Layout.
2. Open `globals.py` and play with the parameters (although most of them would need no change :smile:)
3. Run the code with [appropriate arguments](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide#Run-the-code).
<!-- 3. Add answer key( TODO: add answer key/marking scheme guide)  -->

<!-- #### Testing the code
Datasets to test on : 
Low Quality Dataset(For CV Based methods)) (1.5 GB)
Standard Quality Dataset(For ML Based methods) (3 GB)
High Quality Dataset(For custom processing) (6 GB) 
-->

## 💡 What can OMRChecker do for me?
Once you configure the OMR layout, just throw images of the sheets at the software; and you'll get back the graded responses in an excel sheet! 

Images can be taken from various angles as shown below-
<p align="center">
	<img alt="sample_input" width="400" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/sample_input.PNG">
</p>

These images will be processed in the following manner: 
<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="rotation_stack" width="650" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/rotation.PNG">
	</a>
	<br>
	*Note: This image is generated by the code itself!*
</p>

Finally a CSV file will also be generated containing all the responses and scoring

<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals">
		<img alt="csv_output" width="550" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/csv_output.PNG">
	</a>
</p>

### There are a lot more visuals in the wiki. [Check them out!](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals)

## 💡 Why is this software free?
Our Motto: 
> Don't reinvent the wheel, use good wheels to make great vehicles! 

After seeing it work fabulously at large scale on scanned OMR sheets at Technothlon, we have decided to open source the code and roll out mobile based scanning as well. The feedback from you all will be extremely valuable in making this idea become successful.

### Can I use this code in my work?
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

OMRChecker is completely open source and published under **GPLv3** license which is just to give you a heads up to **disclose usage** of this software in your code. 

OMRChecker can be forked and modified. **You are encouraged to play with it and we would love to see your own projects in action!**

## Credits 
_A Huge thanks to :_
_The creative master **Adrian Rosebrock** for his blog :_ https://pyimagesearch.com 

_The legendary **Harrison** aka sentdex for his [video tutorials](https://www.youtube.com/watch?v=Z78zbnLlPUA&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq)._

_And the james bond of computer vision **Satya Mallic** for his blog:_ https://www.learnopencv.com

_And many other rockstars without whom this project would never have completed._
<!-- Some papers Team Techno -->
_Thank you!_

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

## Activity: 
[![Chat](https://img.shields.io/badge/Join-on_Discord-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)
[![Ask me](https://img.shields.io/badge/Ask_me-anything-purple.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/issues/5)

[![HitCount](http://hits.dwyl.io/udayraj123/OMRchecker.svg)](http://hits.dwyl.io/udayraj123/OMRchecker)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Udayraj123/OMRChecker/wiki/TODOs)
[![GitHub pull-requests closed](https://img.shields.io/github/issues-pr-closed/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/pulls?q=is%3Aclosed)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub contributors](https://img.shields.io/github/contributors/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/graphs/contributors/)


#### Help us reach 550 stars ⭐ to become #1 ([Currently #4](https://github.com/topics/omr)) on the "OMR" tag on github [![GitHub stars](https://img.shields.io/github/stars/Udayraj123/OMRChecker.svg?style=social&label=Stars✯)](https://GitHub.com/Udayraj123/OMRChecker/stargazers/)

<!-- Begin donate section -->
> To keep my 💡 brain juices flowing and create more such projects, [☕ Buy Me A Coffee](https://www.buymeacoffee.com/Udayraj123) 

> If this project saved you large costs on OMR Software licenses, or saved efforts to make one, or simply want to give me some credit: [![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_LG.gif)](https://www.paypal.me/Udayraj123/500) 

<<<<<<< HEAD
> _Note: As per non-profit traditions of [Technothlon](https://www.facebook.com/technothlon.techniche), all of your your donations will go into funding good initiatives in the future._

=======
> If this project saved you large costs on OMR Software licenses, or saved efforts to make one, or simply want to give me some credit: [![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_LG.gif)](https://www.paypal.me/Udayraj123/500) 

> _Note: Following non-profit traditions of [Technothlon](https://www.facebook.com/technothlon.techniche), all of your your donations will go into funding good initiatives._

> To keep my 💡 brain juices flowing and create more such projects, [☕ Buy Me A Coffee](https://www.buymeacoffee.com/Udayraj123) 

<!-- ![☕](https://miro.medium.com/fit/c/256/256/1*br7aoq_JVfxeg73x5tF_Sw.png) -->
<!-- [![paypal.me](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Z5BNNK7AVFVH8&source=url) -->
<!-- https://www.amazon.in/hz/wishlist/ls/3V0TDQBI3T8IL -->

<!-- End donate section -->

