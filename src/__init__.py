from pathlib import Path;
import sys;

SRC_DIR = Path( __file__ ).parent;
if sys.path.count( SRC_DIR ) == 0:
    sys.path.append( SRC_DIR )