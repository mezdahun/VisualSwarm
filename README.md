# VisualSwarm
Implementation of a minimal vision-based flocking algorithm published by Bastien &amp; Romanczuk (2020) on a Raspberry PI for movement control of Thymio II robots.

## Welcome
Welcome to Project VisualSwarm. If you are interested in the theoretical background of the project, or HW/SW configuration before using the software please visit the [project wiki](https://github.com/mezdahun/VisualSwarm/wiki). 

If you are looking for code documentation you can find it under the individual module and submodule README files and in the individual docstrings of the main python methods. An additional detailed code documentation with examples is planned to be created after v1.0. 

If you are interested in other publications and projects of our group, visit us at one of the following pages:
- http://lab.romanczuk.de/
- https://www.scienceofintelligence.de/

If you have any questions or remarks about the project, don't hesitate to contact us via visualswarm.scioi@gmail.com

# Technical Details and the Repository
## Environment and Test Suite
 * The project is written in python (=3.7)
 * Your environment and SW should be prepared as described on [this wiki page](https://github.com/mezdahun/VisualSwarm/wiki/Software-Setup)
 * structured according to general [packaging guidelines](https://packaging.python.org/)
 * designed to run on a debian based system on a Respberry Pi or in a similar virtual machine
 * general code quality is enforced via quality checks in [tox](https://tox.readthedocs.io/en/latest/) framework
 * reproducibility is enforced via unittests and a minimum of 90% code coverage
 * for more information on prerequisites and used packages see `setup.py`

### Run tests locally
To run the test suit and therefore test if the current code status is of high quality, run the following from your terminal. 

Before running be sure that you 
 * cloned the repo in a specific folder 
 * you are on the desired branch
 * you are in the folder where `tox.ini` is
 
```bash
sudo tox
```

tox will prepare a dedicated test environment virtually and will run the predefined tests (as requested in `tox.ini`) in it.

At the and you should see a green success message. If you see errors, it can come from multiple sources such as:
 * code quality is not sufficient (formatting or import errors, general deviations from PEP)
 * failed unittests
 * code coverage is not sufficient

In any case, do the fixes according to the tox report and command line messages. **Do not push anything until you do not get a passing test suite. 
If it is unevitable use the `#temp` flag in the beginning of your commit message.**

## Application
To run the application or the test suite it is assumed that the SW is prepared as described on [this wiki page](https://github.com/mezdahun/VisualSwarm/wiki/Software-Setup)

### Run locally on Pi4
To avoid conflicts with your global environment first create a virtualenv from the root project folder as
```bash
sudo pipenv --python 3.7
sudo pipenv shell
```

If you would like to use a non-Raspberry machine, some pip packages will fail to build that are specialized on Raspberry HW.
To still be able to install the package first run
```bash
export READTHEDOCS=True
```

Once the empty env is created and you are in it (this should be automatically done after `shell`) you should now install the app
```bash
pip install .
```

To check if the app was installed you can run the command line entrypoint simply as
```bash
vswrm-health
```

This should give you a _"VisualSwarm application OK!"_ message and return.

To start the stack without motor control use the entrypoint
```bash
vswrm-start-vision
```

To start the full stack with motor control use the entrypoint
```bash
vswrm-start
```

## Git and GitHub guidelines
### Branches/Naming
 * **master**: protected branch only for releases, a merge into this branch requires review from other developers
 * **develop**: main branch for developing and merging feature branches together
 * **feture/example-feature**: must be opened from develop and implements a feature
 * **fix/example-fix**: can be opened from any branch and fixes an issue

Do not directly push anything to develop.

### Actions and CI/CD
 * **Tox check**: Currently all feature branches shall pass on the tox check defined in the tox file before merging. If
tox fails, the feature branch can not be merged. This also shows that develop is indeed only for finalized features.
