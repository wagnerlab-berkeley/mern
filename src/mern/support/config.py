"""
Configuration settings for MeRN support tools.
"""

import os
from pathlib import Path


# Get the package directory
PACKAGE_DIR = Path(__file__).parent

# KEGG directory for storing KEGG data (relative to package)
KEGG_DIR = str(PACKAGE_DIR / 'data' / 'kegg')
