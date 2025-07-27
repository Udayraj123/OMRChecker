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
Make sure to run uv ruff check (with optionally --fix) to format your code
```
uv ruff check
```

# Documentation
- Update Contributors.md file with your name
- Any changes in the functionality of the tool, especially in the configuration schema should be documented. For documentation we are using mike(internally uses mkdocs).
- To view documentation site locally, use mike:
```bash
# https://github.com/mkdocs-material/example-versioning/blob/master/README.md
# List versions -
uv run mike list
# Update and commit given version's docs in the gh-pages branch
uv run mike deploy <version>
# Serve the docs
uv run mike serve
```

# Where to contribute from

- You can pickup any open [issues](https://github.com/Udayraj123/OMRChecker/issues) to solve.
- You can also check out the [ideas list](https://github.com/users/Udayraj123/projects/2/views/1)
