## Step by step walkthrough for creating a basic template
<!-- **Note for contributors:** There's a [TODO Challenge](./TODOs) to automate this process using image processing.  -->

This tutorial will show you how to create template layout files using a simple example.

<!-- TODO explain directory structure here -->

First let's make a layout for a sample OMR from [Adrian's blog](https://pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/).
<!-- image here -->
<p align="center">
  <img alt="Adrian OMR" width="350" src="./images/AdrianSample/HE/adrian_omr.png">
</p>

1. Create a directory for your files, say `inputs/AdrianSamples`. Note that all directories in `inputs/` directory will be processed by default.

2. Download above OMR image and put it into `inputs/AdrianSamples/`.

3. Create a file `inputs/template.json`. Putting the following starter json in it.

```
{
  "pageDimensions": [ 300, 400 ],
  "bubbleDimensions": [ 20, 20 ],
  "customLabels": {},
  "fieldBlocks": {
    "MCQBlock1": {
      "fieldType": "QTYPE_MCQ5",
      "origin": [ 0, 0 ],
      "fieldLabels": ["q1", "q2", "q3", "q4", "q5"],
      "bubblesGap": 30,
      "labelsGap": 30
    }
  },
  "preProcessors": [
    {
      "name": "CropPage",
      "options": {
        "morphKernel": [ 10, 10 ]
      }
    }
  ]
}
```

Now run `python3 main.py --setLayout`. The page should get cropped automatically and show a basic overlay of the template.
Note that we have put `"origin": [0, 0],` which means the overlay will start from the top left corner.

<p align="center">
  <img alt="Initial Layout" width="400" src="./images/initial_layout.png">
</p>
Now let's adjust the top left corner(origin). Change origin from [0,0] to a better coordinate, say [50, 50] and run above command again. After multiple trials, you should find that origin is best fit at [65, 60]. Update the origin in json file : 

```
    "origin": [65, 60],
```
Run the command again.
<!-- Put origin_step here -->
<p align="center">
  <img alt="Origin Step" width="400" src="./images/origin_step.png">
</p>

Now let's tweak over the two gaps `bubblesGap` and `labelsGap`. 
Clearly we need to update the gaps to be bigger. Also, horizontal gaps are smaller than vertical ones. Tweaked gaps come out to be- 
```
    "bubblesGap" : 41,
    "labelsGap" : 52,
```
The bubbles also should be made slightly bigger
```
  "bubbleDimensions": [25, 25 ],
```
Run the command again to get the arranged layout.
<!-- put final_layout here -->
<p align="center">
  <img alt="Final Layout" width="400" src="./images/final_layout.png">
</p>

Note the "preProcessors" array, there are various plugins to use. Each plugin is described with a `name` and an `options` object that contains the configuration of the plugin. In our case, we use the 'CropPage' plugin with a (default) option of using morph kernel of size [10, 10].

Above is the simplest version of what the template.json can do. 

For more templates see [sample folders](https://github.com/Udayraj123/OMRChecker/tree/master/samples).

To understand how rest of the parameters work in template.json, checkout [About Templates](./About-Templates)

### Note for capturing using mobile phones

Please check the `sample1/` folder to understand the use of `omr_marker.jpg`. If you can modify your OMR sheets with these markers, it will give you much higher accuracy when scanning using mobile camera. We enable the markers plugin using the following snippet.

```js
{
  // ...
  "preProcessors": [
    // ...
    {
      "name": "CropOnMarkers",
      "options": {
        "relativePath": "omr_marker.jpg",
      }
    }
  ]
}
```

<!-- bummer: do not change the header text as it's linked -->
## Running OMRChecker

Run `python3 main.py` to generate outputs for input file.

Note: For full usage refer the [Project Readme](https://github.com/Udayraj123/OMRChecker#full-usage)