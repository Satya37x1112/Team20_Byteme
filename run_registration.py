"""
Launcher script for Face Registration System
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_registration.main import main

if __name__ == "__main__":
    main()
