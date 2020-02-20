import io
import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))


# Package information
NAME = 'SciNet'
VERSION = "0.0.1"

DESCRIPTION = 'PHYS490 Final Project - SciNet recreation'
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()

# Contributor information
AUTHOR = 'Nolan Dickson, Jesse Thompson, Joshua Mcpherson, Veronica Chatrath'
CONTACT_EMAIL = 'nmdickso@edu.uwaterloo.ca'

# Installation information
# TODO should actually read this from requirements.txt
REQUIRED = ['numpy', 'torch', 'matplotlib']
REQUIRES_PYTHON = '>=3.7'

# setup parameters
setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',

    author=AUTHOR,
    author_email=CONTACT_EMAIL,

    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,

    packages=['scinet'],
    # entry_points={},
)
