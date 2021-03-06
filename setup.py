from distutils.core import setup

from setuptools import find_packages

setup(
    name='VisualSwarm',
    description='Implementation of a minimal vision algorithm published by Bastien & Romanczuk (2020) on a Raspberry '
                'PI for movement control of Thymio II robots.',
    version='0.1.3',
    url='https://github.com/mezdahun/VisualSwarm',
    maintainer='David Mezey @ HU, TU-SciOI, BCCN',
    packages=find_packages(exclude=['tests']),
    package_data={'visualswarm': ['data/*']},
    python_requires=">=3.7",
    install_requires=[
        'opencv-python==4.4.0.46',
        'numpy==1.20.1',
        'picamera==1.13',
        'pandas==1.2.0',
        'influxdb==5.3.1',
        'scipy==1.6.0',
        'psutil==5.8.0',
        'pycairo==1.20.0',
        'PyGObject==3.38.0',
        'dbus-python==1.2.16',
        'typing-extensions==3.7.4.3'
    ],
    extras_require={
        'test': [
            'bandit',
            'flake8',
            'pytest',
            'pytest-cov',
            'safety',
            'fake-rpi',
            'freezegun'
        ]
    },
    entry_points={
        'console_scripts': [
            'vswrm-health=visualswarm.app:health',
            'vswrm-start-vision=visualswarm.app:start_application',
            'vswrm-start=visualswarm.app:start_application_with_control'
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
