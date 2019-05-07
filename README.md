# OMR-Checker
A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. With support for a customizable marking scheme with section-wise marking, bonus questions, etc. 

## How to run
#### Clone the repo
Use `--depth=1` if you want only the latest version(less than half download size)
```
git clone --depth=1 https://github.com/udayraj123/OMRChecker
```

#### Install dependencies

Install opencv
More details here: https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/ 
```
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install opencv-python
```

Install other requirements
```
sudo python3 -m pip install -r requirements.txt
```
<!-- 
## Code in action (To be updated)
#### Normal scans
<img src="./progress/in_action/light_action.gif">
<br>
#### Xerox scans
<img src="./progress/in_action/dark_action.gif">
 -->