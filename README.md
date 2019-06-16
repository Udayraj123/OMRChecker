# OMR-Checker
A full-fledged OMR checking software that can read and evaluate OMR sheets scanned at any angle and having any color. With support for a customizable marking scheme with section-wise marking, bonus questions, etc. 

## How to run
#### 1. Clone the repo
Use `--depth=1` if you want only the latest version(less than half download size)
```
git clone --depth=1 https://github.com/udayraj123/OMRChecker
```

#### 2. Install dependencies

###### Install opencv
More details here: https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/ 
```
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install opencv-python
```

###### Install other requirements
```
sudo python3 -m pip install -r requirements.txt
```

#### 3. Run the code

1. Make a copy of 'OMR_Files_sample' and rename it to 'OMR_Files' (Do not make changes to the directory structure)
2. Run code
	```
	python3 main.py
	```
3. :smirk: :smirk: :smirk:
4. Profit!!

### Directory Structure 
![Directory Structure](https://raw.githubusercontent.com/udayraj123/OMRChecker/master/directory_structure.png)
This structure has been created to suit for better organization of OMRs (Citywise, then Group-wise and Language-wise)

#### Configuring for your OMR Sheets (W.I.P.)
	1. Put your OMR images in `inputs/OMR_Files/CityName/JE` (You can rename CityName)
	2. Put template layout(s) in `inputs/Layouts` (Guide coming soon)
	3. Put marker crop at `inputs/omr_marker.jpg`
	4. (optional) Advanced configuration can be done in globals.py
	5. Run code

<!-- 
## Code in action (To be updated)
#### Normal scans
<img src="./progress/in_action/light_action.gif">
<br>
#### Xerox scans
<img src="./progress/in_action/dark_action.gif">
 -->