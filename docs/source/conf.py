# Configuration file for the Sphinx documentation builder.

project = "MeRN"
copyright = "2026, Adelina Chau"
author = "Adelina Chau"
release = "0.1"

extensions = [
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", ".DS_Store", "**/.ipynb_checkpoints"]
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "MeRN Documentation"

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

nb_execution_mode = "off"
nb_merge_streams = True
