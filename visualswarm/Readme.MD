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
the raw visual stream to process this raw stream further until we get a higher level stream that can be consumed by high level processes. All
behavior related calculation is grouped in the `behavior` module. Changing hyperparameters are stored in the `contrib` module. The `control` module
includes all motor control related process. The `monitoring` module records and stores system rescources in an InfluxDB instance
that can be followed from a Grafana board on the local network.

### Contrib (visualswarm.contrib)
Includes tunable parameters of the stack grouped by the process names that the parameters belong to.

### Monitoring (visualswarm.monitoring)
Includes methods to record system status in an InfluxDB instance locally that can be visualized on a Grafana board on
the local network.

### Vision (visualswarm.vision)
This submodule holds methods to acquire raw camera images, fulfills color space segmentation to find target objects and projects this processed
image into a 1D binary visual projection field. (VPF)

### Behavior (visualswarm.behavior)
This submodule includes methods to calculate state variables (agent velocity and heading direction) from the VPF calculated by lower order
processes.

### Control (visualswarm.control)
This submodule includes methods to translate the calculated state variables into high level motor commands. Furthermore, this 
submodule is reliable to connect to the Thymio hardware and transmit high level commands to it.

## Entrypoints
To use the package you can run one of the entrypoint commands as defined in `setup.py`.
* `vswrm-health` : check if package is installed properly
* `vswrm-start-vision` : test vision and behavior processes (If you run this via SSH you will need `-Y` to forward graphical 
output from openCV, more info [here](https://github.com/mezdahun/VisualSwarm/wiki/Remote-Access-(SSH)))
* `vswrm-start` : start full stack with motor control 