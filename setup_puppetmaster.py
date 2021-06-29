from distutils.core import setup

from setuptools import find_packages

setup(
    name='VisualSwarm',
    description='Implementation of a minimal vision algorithm published by Bastien & Romanczuk (2020) on a Raspberry '
                'PI for movement control of Thymio II robots. This is a version of the original setup.py that is used'
                'to overwrite the package setup in case of WeBots simulations with venv environments. Check the readme'
                'and the corresponding wiki pages for more details and correct usage.'
                ''
                'PUPPETMASTER SUBVERSION: control multiple VSWRM robots over SSH with fabric',
    version='0.1.4',
    url='https://github.com/mezdahun/VisualSwarm',
    maintainer='David Mezey @ HU, TU-SciOI, BCCN',
    packages=find_packages(exclude=['tests']),
    package_data={'visualswarm': ['data/*']},
    python_requires=">=3.7",
    install_requires=[
        'numpy==1.20.1',
        'pandas==1.2.0',
        'scipy==1.6.0',
        'psutil==5.8.0',
        'typing-extensions==3.7.4.3',
        'freezegun==1.1.0',
        'matplotlib',
        'fabric==2.6.0',
        'google-cloud-logging==2.5.0'
    ],
    extras_require={
        'test': [
            'bandit',
            'flake8',
            'pytest',
            'pytest-cov',
            'safety',
            'freezegun'
        ]
    },
    entry_points={
        'console_scripts': [
            'vswrm-masterpuppet=visualswarm.app_puppetmaster:start_swarm'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: Other OS',
        'Programming Language :: Python :: 3.7'
    ],
    test_suite='tests',
    zip_safe=False
)
