import importlib

_config = None


def set_configuration(config_name):
    global _config
    _config = importlib.import_module("configurations.%s" % config_name)
    print "loaded", _config


def config():
    return _config
