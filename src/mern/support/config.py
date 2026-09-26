"""
Configuration settings for MeRN support tools.
"""

import os
from pathlib import Path


# Get the package directory
PACKAGE_DIR = Path(__file__).parent

# Versioned KEGG-derived resources distributed with MERN.
KEGG_DIR = str(PACKAGE_DIR / 'data' / 'kegg')

# KEGG files downloaded by the user are kept outside the installed package.
KEGG_CACHE_DIR = str(
    Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'mern' / 'kegg'
)
