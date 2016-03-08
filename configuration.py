"""Module acting as a singleton object for storing the configuration module.
"""

import importlib


_CONFIG_DIR = "configurations"

_config = None


def set_configuration(configuration):
    """Imports and initialises the configuration module."""
    global _config
    _config = importlib.import_module("%s.%s" % (_CONFIG_DIR, configuration))
    print "loaded", _config


def config():
    """Fetches the currently loaded configuration module."""
    return _config