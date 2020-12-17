# Python Package VisualSwarm
This is the README of the python package VisualSwarm. The package is created to include and group software elements necessary to implement 
vision based flocking of Thymio 2 robots using Python and OpenCV on a Raspberry Pi 4.

## Prerequisites
The following prerequisites must be installed to use VisualSwarm:
* python 3.7 (version strictly equal)
* packages as defined in `setup.py`
* find more information about preparing the SW environment [on the wiki](https://github.com/mezdahun/VisualSwarm/wiki/Software-Setup)

## Modules
The modules are created according to the axis of information flow. A `vision` module includes all vision related methods from establishing 
the raw visual stream to process this raw stream further until we get a gigh level stream that can be consumed by control processes. All
behavior related control process is grouped in the `behavior` module.

### Vision (visualswarm.vision)

### Behavior (visualswarm.behavior)

## Entrypoints
To use the package you can run one of the entrypoint commands as defined in `setup.py`.
* `vswrm-health` : check if package is installed properly
* `vswrm-start-vision` : test raw and processed continous camera stream (If you run this via SSH you will need `-Y` to forward graphical 
output from openCV, more info [here](https://github.com/mezdahun/VisualSwarm/wiki/Remote-Access-(SSH)))
