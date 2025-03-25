
This repository contains the code for the ICML2025 submission titled "SHAC++: A Neural Network to Rule All Differentiable Simulators".

### Pre-requisites
The prerequisites are contained in the file `requirements.txt`. You can install them by running `pip install -r requirements.txt`. However, we suggest using a virtual environment to avoid conflicts with other packages.
`python3 -m venv venv`
`source venv/bin/activate`
`python3 -m pip install -r requirements.txt`

### Execution
The experiments are all generated in the makefile `makefile.mk`. You can simply run all experiments by using the command `make -f makefile.mk all`. The directory `data` will be created and populated with models, logs, and checkpoints.
