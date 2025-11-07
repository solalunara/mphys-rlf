from pathlib import Path;
import sys;

SRC_DIR = Path( __file__ ).parent.parent;
if sys.path.count( str( SRC_DIR ) ) == 0:
    sys.path.append( str( SRC_DIR ) );