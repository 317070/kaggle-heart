#!/usr/bin/env python
from contextlib import contextmanager

"""
This changes the print-function, such that all output is also copied to a log file
"""
@contextmanager
def print_to_file(filename):
    import sys
    old_stdout = sys.stdout
    try:
        logfile = open(filename, "w")
    except:
        yield

    class CustomPrint():
        def __init__(self, stdout, logfile):
            self.old_stdout = stdout
            self.logfile = logfile

        def write(self, text):
            self.old_stdout.write(text)
            self.logfile.write(text)

        def flush(self):
            self.old_stdout.flush()
            self.logfile.flush()

    sys.stdout = CustomPrint(old_stdout, logfile)
    sys.stderr = CustomPrint(old_stdout, logfile)

    try:
        yield
    finally:
        sys.stdout = old_stdout

