# VisualSwarm
Implementation of a minimal vision algorithm published by Bastien &amp; Romanczuk (2020) on a Raspberry PI for movement control of Thymio II robots.

## Environment and Test Suite
 * The project is written in python (>=3.7)
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
### Run locally
To avoid conflicts with your global environment first create a virtualenv from the root project folder as
```bash
sudo pipenv shell
```

Once the empty env is created and you are in it (this should be automatically done after `shell`) you should now install the app
```bash
pip install .
```

To check if the app was installed you can run the command line entrypoint simply as
```bash
visualswarm-health
```

This should give you a _"VisualSwarm application OK!"_ message and return.