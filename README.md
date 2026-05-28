# OMR Checker

Read OMR sheets fast and accurately using a scanner 🖨 or your phone 🤳.

## What is OMR?

OMR stands for Optical Mark Recognition, used to detect and interpret human-marked data on documents. OMR refers to the process of reading and evaluating OMR sheets, commonly used in exams, surveys, and other forms.

#### **Quick Links**

- [Installation](#getting-started)
- [User Guide](https://github.com/Udayraj123/OMRChecker/wiki)
- [Contributor Guide](https://github.com/Udayraj123/OMRChecker/blob/master/CONTRIBUTING.md)
- [Project Ideas List](https://github.com/users/Udayraj123/projects/2/views/1)

<hr />

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/pull/new/master) <!-- [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/Udayraj123/OMRChecker/wiki/TODOs) -->
[![GitHub pull-requests closed](https://img.shields.io/github/issues-pr-closed/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/pulls?q=is%3Aclosed)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/issues?q=is%3Aissue+is%3Aclosed)
[![Ask me](https://img.shields.io/badge/Discuss-on_Github-purple.svg?style=flat-square)](https://github.com/Udayraj123/OMRChecker/issues/5)

<!-- [![GitHub contributors](https://img.shields.io/github/contributors/Udayraj123/OMRChecker.svg)](https://GitHub.com/Udayraj123/OMRChecker/graphs/contributors/) -->

[![GitHub stars](https://img.shields.io/github/stars/Udayraj123/OMRChecker.svg?style=social&label=Stars✯)](https://GitHub.com/Udayraj123/OMRChecker/stargazers/)
[![Join](https://img.shields.io/badge/Join-Discord_group-purple.svg?style=flat-square)](https://discord.gg/qFv2Vqf)

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/Udayraj123/a125b1531c61cceed5f06994329cba66/omrchecker-on-cloud.ipynb) -->

## 🎯 Features

A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color.

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

Note: For solving interesting challenges, developers can check out [**TODOs**](https://github.com/Udayraj123/OMRChecker/wiki/TODOs).

See the complete guide and details at [Project Wiki](https://github.com/Udayraj123/OMRChecker/wiki/).

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

See step-by-step processing of any OMR sheet:

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

We now support [colored outputs](https://github.com/Udayraj123/OMRChecker/wiki/%5Bv2%5D-About-Evaluation) as well. Here's a sample output on another image -
<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/%5Bv2%5D-About-Evaluation">
		<img alt="colored_output" width="550" src="./docs/assets/colored_output.jpg">
	</a>
</p>

#### There are many more visuals in the wiki. Check them out [here!](https://github.com/Udayraj123/OMRChecker/wiki/Rich-Visuals)

## Getting started

![Setup Time](https://img.shields.io/badge/Setup_Time-20_min-blue.svg)

**Operating system:** OSX or Linux is recommended although Windows is also supported.

### 1. Install global dependencies

![opencv 4.0.0](https://img.shields.io/badge/opencv-4.0.0-blue.svg) ![python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)

To check if python3 and pip is already installed:

```bash
python3 --version
python3 -m pip --version
```

<details>
	<summary><b>Install Python3</b></summary>

To install python3 follow instructions [here](https://www.python.org/downloads/)

To install pip - follow instructions [here](https://pip.pypa.io/en/stable/installation/)

</details>
<details>
<summary><b>Install OpenCV</b></summary>

**Any installation method is fine.**

Recommended:

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user opencv-python
python3 -m pip install --user opencv-contrib-python
```

More details on pip install openCV [here](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/).

</details>

<details>

<summary><b>Extra steps(for Linux users only)</b></summary>

<b>Installing missing libraries(if any):</b>

On a fresh computer, some of the libraries may get missing in event after a successful pip install. Install them using following commands[(ref)](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/):

```bash
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran
```

</details>

### 2. Install project dependencies

Clone the repo

```bash
git clone https://github.com/Udayraj123/OMRChecker
cd OMRChecker/
```

Install pip requirements

```bash
python3 -m pip install --user -r requirements.txt
```

_**Note:** If you face a distutils error in pip, use `--ignore-installed` flag in above command._

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

Alternatively you can also use `python3 main.py -i ./samples/sample1`.

Each example in the samples folder demonstrates different ways in which OMRChecker can be used.

### Common Issues

<details>
<summary>
	1. [Windows] ERROR: Could not open requirements file<br>
	</summary>
Command: <code>python3 -m pip install --user -r requirements.txt</code>
<br>
	Link to Solution:  <a href="https://github.com/Udayraj123/OMRChecker/issues/54#issuecomment-1264569006">#54</a>
</details>
<details>
<summary>
2. [Linux] ERROR: No module named pip<br>
</summary>
Command: <code>python3 -m pip install --user --upgrade pip</code>
<br>
	Link to Solution: <a href="https://github.com/Udayraj123/OMRChecker/issues/70#issuecomment-1268094136">#70</a>
</details>

## OMRChecker for custom OMR Sheets

1. First, [create your own template.json](https://github.com/Udayraj123/OMRChecker/wiki/User-Guide).
2. Configure the tuning parameters.
3. Run OMRChecker with appropriate arguments (See full usage).
<!-- 4. Add answer key( TODO: add answer key/marking scheme guide)  -->

## Full Usage

```
python3 main.py [--setLayout] [--inputDir dir1] [--outputDir dir1]
```

Explanation for the arguments:

`--setLayout`: Set up OMR template layout - modify your json file and run again until the template is set.

`--inputDir`: Specify an input directory.

`--outputDir`: Specify an output directory.

<details>
<summary>
 <b>Deprecation logs</b>
</summary>

- The old `--noCropping` flag has been replaced with the 'CropPage' plugin in "preProcessors" of the template.json(see [samples](https://github.com/Udayraj123/OMRChecker/tree/master/samples)).
- The `--autoAlign` flag is deprecated due to low performance on a generic OMR sheet
- The `--template` flag is deprecated and instead it's recommended to keep the template file at the parent folder containing folders of different images
</details>

<!-- #### Testing the code
Datasets to test on :
Low Quality Dataset(For CV Based methods)) (1.5 GB)
Standard Quality Dataset(For ML Based methods) (3 GB)
High Quality Dataset(For custom processing) (6 GB)
-->

## FAQ

<details>
<summary>
<b>Why is this software free?</b>
</summary>

This project was born out of a student-led organization called as [Technothlon](https://technothlon.techniche.org.in). It is a logic-based international school championship organized by students of IIT Guwahati. Being a non-profit organization, and after seeing it work fabulously at such a large scale we decided to share this tool with the world. The OMR checking processes still involves so much tediousness which we aim to reduce dramatically.

We believe in the power of open source! Currently, OMRChecker is in an intermediate stage where only developers can use it. We hope to see it become more user-friendly as well as robust from exposure to different inputs from you all!

[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

</details>

<details>
<summary>
<b>Can I use this code in my (public) work?</b>
</summary>

OMRChecker can be forked and modified. You are encouraged to play with it and we would love to see your own projects in action!

It is published under the [MIT license](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE).

</details>

<details>
<summary>
<b>What are the ways to contribute?</b>
</summary>

<!-- - Help OMRChecker reach more people by giving a star! The Goal is to reach top position for the [OMR Topic](https://github.com/topics/omr) -->

- Join the developer community on [Discord](https://discord.gg/qFv2Vqf) to fix [issues](https://github.com/Udayraj123/OMRChecker/issues) with OMRChecker.

- If this project saved you large costs on OMR Software licenses, or saved efforts to make one. Consider donating an amount of your choice(donate section).

<!-- ![☕](https://miro.medium.com/fit/c/256/256/1*br7aoq_JVfxeg73x5tF_Sw.png) -->
<!-- [![paypal.me](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=Z5BNNK7AVFVH8&source=url) -->
<!-- https://www.amazon.in/hz/wishlist/ls/3V0TDQBI3T8IL -->

</details>

## Credits

_A Huge thanks to:_
_**Adrian Rosebrock** for his exemplary blog:_ https://pyimagesearch.com

_**Harrison Kinsley** aka sentdex for his [video tutorials](https://www.youtube.com/watch?v=Z78zbnLlPUA&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq) and many other resources._

_**Satya Mallic** for his resourceful blog:_ https://www.learnopencv.com

_And to other amazing people from all over the globe who've made significant improvements in this project._

_Thank you!_

<!--
OpencV
matplotlib
some SO answers from roughworks
prof
-->

## Related Projects

Here's a snapshot of the [Android OMR Helper App (archived)](https://github.com/Udayraj123/AndroidOMRHelper):

<p align="center">
	<a href="https://github.com/Udayraj123/AndroidOMRHelper">
		<img height="300" src="https://raw.githubusercontent.com/wiki/Udayraj123/OMRChecker/extras/Progress/2019-04-26/images/app_flow.PNG">
	</a>
</p>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/Udayraj123/OMRChecker.svg)](https://starchart.cc/Udayraj123/OMRChecker)

---

<h2 align="center">Made with ❤️ by Awesome Contributors</h2>

<a href="https://github.com/Udayraj123/OMRChecker/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Udayraj123/OMRChecker" />
</a>

---

### License

[![GitHub license](https://img.shields.io/github/license/Udayraj123/OMRChecker.svg)](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE)

For more details see [LICENSE](https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE).

### Donate

<a href="https://www.buymeacoffee.com/Udayraj123" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a> [![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_LG.gif)](https://www.paypal.me/Udayraj123/500)

_Find OMRChecker on_ [**_Product Hunt_**](https://www.producthunt.com/posts/omr-checker/) **|** [**_Reddit_**](https://www.reddit.com/r/computervision/comments/ccbj6f/omrchecker_grade_exams_using_python_and_opencv/) **|** [**Discord**](https://discord.gg/qFv2Vqf) **|** [**Linkedin**](https://www.linkedin.com/pulse/open-source-talks-udayraj-udayraj-deshmukh/) **|** [**goodfirstissue.dev**](https://goodfirstissue.dev/language/python) **|** [**codepeak.tech**](https://www.codepeak.tech/) **|** [**fossoverflow.dev**](https://fossoverflow.dev/projects) **|** [**Interview on Console by CodeSee**](https://console.substack.com/p/console-140) **|** [**Open Source Hub**](https://opensourcehub.io/udayraj123/omrchecker)

 <!-- [***Hacker News***](https://news.ycombinator.com/item?id=20420602) **|** -->
 <!-- **|** [***Swyya***](https://www.swyya.com/projects/omrchecker) -->
