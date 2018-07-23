from __future__ import print_function
from setuptools import setup, find_packages
import os
import sys
import re


requires = [
    'h5py',
    'matplotlib',
    'mttkinter',
    'numpy',
    'paramiko',
    'scipy',
    'tqdm',
    'zmq',
    'sklearn',
    'hdbscan',
    'pyopencl',
    'filelock',
    'vispy',
    'filelock'
]


if sys.version_info < (2, 7):
    raise RuntimeError('Only Python versions >= 2.7 are supported')

current_directory = os.path.dirname(os.path.realpath(__file__))

init_path = os.path.join(current_directory, 'circusort', '__init__.py')
with open(init_path, mode='r') as init_file:
    version = re.search(r"__version__ = '([^']+)'", init_file.read()).group(1)

readme_path = os.path.join(current_directory, 'README.rst')
with open(readme_path, mode='r') as readme_file:
    long_description = readme_file.read()

use_2to3 = (sys.version_info.major == 2)


setup(
    name='circusort',
    version=version,
    description='Online spike sorting by template matching',
    long_description=long_description,
    url='http://spyking-circus.rtfd.org',
    author='Pierre Yger, Baptiste Lefebvre and Olivier Marre',
    author_email='pierre.yger@inserm.fr',
    license='License :: OSI Approved :: UPMC CNRS INSERM Logiciel Libre License, version 2.1 (CeCILL-2.1)',
    keywords="spike sorting template matching tetrodes extracellular real-time",
    packages=find_packages(),
    setup_requires=['setuptools>0.18'],
    install_requires=requires,
    use_2to3=use_2to3,
    entry_points={
        'console_scripts': [
            'spyking-circus-ort = circusort:main',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    zip_safe=False
)
