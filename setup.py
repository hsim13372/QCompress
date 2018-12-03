
import os
from setuptools import setup, find_packages

# This reads the __version__ variable
exec(open('src/qcompress/_version.py').read())

# README file as long_description
long_description = open('README.rst').read()

# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(
    name='qcompress',
    version=__version__,
    description='A Python framework for the quantum autoencoder algorithm',
    long_description=long_description,
    install_requires=requirements,
    url='https://github.com/hsim13372/QCompress',
    author='hannahsim',
    author_email='hsim13372@gmail.com',
    license='Apache-2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src', 'qcompress', 'data', '*.hdf5'),
             os.path.join('images', '*.png')]
    },
    python_requires=">=3.6"
    )
