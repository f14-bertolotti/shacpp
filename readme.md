# SHAC++

This repository contains the code for the [paper](https://ecai2025.org/accepted-papers/) titled "SHAC++: A Neural Network to Rule All Differentiable Simulators".
=======
This repository contains the code for the ECAI2025 submission titled "SHAC++: A Neural Network to Rule All Differentiable Simulators". [paper][]

## tldr
We present SHAC++, an extension of the SHAC reinforcement learning algorithm. Unlike [SHAC](https://arxiv.org/abs/2204.07137), which relies on differentiable environments for backpropagation through time, SHAC++ leverages neural networks trained alongside the policy to model the environment.

## Useful Resources:

- *Web Page*: [link](https://f14-bertolotti.github.io/shacpp/)
- *Paper*: [link](TODO)

## Pre-requisites
The prerequisites are contained in the file `requirements.txt`. You can install them by running `pip install -r requirements.txt`. However, we suggest using a virtual environment to avoid conflicts with other packages.
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Execution
The experiments are all generated in the makefile `makefile.mk`. You can simply run all experiments by using the command `make -f makefile.mk all`. The directory `data` will be created and populated with models, logs, and checkpoints.

### BibTeX
```
@inproceedings{shacpp,
  title={SHAC++: A Neural Network to Rule All Differentiable Simulators},
  author={Bertolotti, Francesco and Aguzzi, Gianluca and Cazzola, Walter and Viroli, Mirko},
  booktitle={European Conference on Artificial Intelligence (ECAI)},
  year={2025},
  url={https://github.com/f14-bertolotti/shacpp}
}
```
