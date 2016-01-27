import importlib

_config = None


def set_configuration(configuration):
    global _config
    _config = importlib.import_module("configurations.%s" % configuration)
    print "loaded", _config


def config():
    return _config
