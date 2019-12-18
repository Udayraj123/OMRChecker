'''
Extension/Extension framework
Adapated from https://github.com/gdiepen/python_extension_example
'''

import inspect
import os
import pkgutil


class Extension:
    '''Base class that each extension must inherit from.
    '''
    def __init__(self):
        self.description = 'UNKNOWN'


class ImagePreprocessor(Extension):
    '''Base class for extension that applies some preprocessing to the input image
    '''
    def __init__(self, options, path):
        raise NotImplementedError

    def apply_filter(self, image, filename):
        ''' Apply filter to the image and returns modified image
        '''
        raise NotImplementedError

    def exclude_files(self):
        ''' Returns a list of file paths that should be excluded from processing
        '''
        return []


class ExtensionManager:
    """Upon creation, this class will read the extensions package for modules
    that contain a class definition that is inheriting from the Extension class
    """

    def __init__(self, extension_package):
        """Constructor that initiates the reading of all available extensions
        when an instance of the ExtensionCollection object is created
        """
        self.extension_package = extension_package
        self.reload_extensions()


    def reload_extensions(self):
        """Reset the list of all extensions and initiate the walk over the main
        provided extension package to load all available extensions
        """
        self.extensions = {}
        self.seen_paths = []
        print()
        print(f'Looking for extensions under package {self.extension_package}')
        self.walk_package(self.extension_package)


    def walk_package(self, package):
        """Recursively walk the supplied package to retrieve all extensions
        """
        imported_package = __import__(package, fromlist=['blah'])

        for _, extensionname, ispkg in pkgutil.walk_packages(imported_package.__path__, imported_package.__name__ + '.'):
            if not ispkg:
                extension_module = __import__(extensionname, fromlist=['blah'])
                clsmembers = inspect.getmembers(extension_module, inspect.isclass)
                for (_, c) in clsmembers:
                    # Only add classes that are a sub class of Extension, but NOT Extension itself
                    if issubclass(c, Extension) & (c is not Extension):
                        print(f'    Found extension class: {c.__module__}.{c.__name__}')
                        self.extensions[c.__name__] = c


        # Now that we have looked at all the modules in the current package, start looking
        # recursively for additional modules in sub packages
        all_current_paths = []
        if isinstance(imported_package.__path__, str):
            all_current_paths.append(imported_package.__path__)
        else:
            all_current_paths.extend([x for x in imported_package.__path__])

        for pkg_path in all_current_paths:
            if pkg_path not in self.seen_paths:
                self.seen_paths.append(pkg_path)

                # Get all sub directory of the current package path directory
                child_pkgs = [p for p in os.listdir(pkg_path) if os.path.isdir(os.path.join(pkg_path, p))]

                # For each sub directory, apply the walk_package method recursively
                for child_pkg in child_pkgs:
                    self.walk_package(package + '.' + child_pkg)
