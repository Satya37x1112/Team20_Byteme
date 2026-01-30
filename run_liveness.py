"""
Launcher script for Random Challenge Liveness Verification System
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from liveness_system.main import main

if __name__ == "__main__":
    main()
