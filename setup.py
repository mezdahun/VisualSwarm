from distutils.core import setup

from setuptools import find_packages

setup(
    name='VisualSwarm',
    description='Implementation of a minimal vision algorithm published by Bastien & Romanczuk (2020) on a Raspberry '
                'PI for movement control of Thymio II robots.',
    version='0.0.1',
    url='https://github.com/mezdahun/VisualSwarm',
    maintainer='David Mezey @ HU, TU-SciOI, BCCN',
    packages=find_packages(exclude=['tests']),
    package_data={'visualswarm': ['data/*']},
    python_requires=">=3.7",
    install_requires=[
        'opencv-python>=4.4.0.46',
        'picamera>=1.13'
    ],
    extras_require={
        'test': [
            'bandit',
            'flake8',
            'isort',
            'pytest',
            'pytest-cov',
            'safety',
            'fake-rpi'
        ]
    },
    entry_points={
        'console_scripts': [
            'visualswarm-health=visualswarm.app:health'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: Other OS',
        'Programming Language :: Python :: 3.8'
    ],
    test_suite='tests',
    zip_safe=False
)
