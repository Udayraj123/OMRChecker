# How to contribute
So you want to write code and get it landed in the official OMRChecker repository?
First, fork our repository into your own GitHub account, and create a local clone of it as described in the installation instructions.
The latter will be used to get new features implemented or bugs fixed.

Once done and you have the code locally on the disk, you can get started. We advise you to not work directly on the master branch,
but to create a separate branch for each issue you are working on. That way you can easily switch between different work,
and you can update each one for the latest changes on the upstream master individually.


# Writing Code
For writing the code just follow the [Pep8 Python style](https://peps.python.org/pep-0008/) guide, If there is something unclear about the style, just look at existing code which might help you to understand it better.

Also, try to use commits with [conventional messages](https://www.conventionalcommits.org/en/v1.0.0/#summary).


# Code Formatting
Before committing your code, make sure to run the following command to format your code according to the PEP8 style guide:
```.sh
pip install -r requirements.dev.txt && pre-commit install
```

Run `pre-commit` before committing your changes:
```.sh
git add .
pre-commit run -a
```

# Where to contribute from

- You can pickup any open [issues](https://github.com/Udayraj123/OMRChecker/issues) to solve.
- You can also check out the [ideas list](https://github.com/users/Udayraj123/projects/2/views/1)
