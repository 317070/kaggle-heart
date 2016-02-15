import importlib

_config = None
_subconfig = None


def set_configuration(config_name):
    global _config
    _config = importlib.import_module("configurations.%s" % config_name)
    print "loaded", _config


def set_subconfiguration(config_name):
    global _subconfig
    _subconfig = importlib.import_module("configurations.%s" % config_name)
    print "loaded", _subconfig


def config():
    return _config


def subconfig():
    return _subconfig
