import sys
import os

# Add the path to the parent directory to sys.path
# without it pytest will not be able to find the module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))