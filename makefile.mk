
venv/bin/python3:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

data/shacwm-a3/eval.log: venv/bin/python3
	venv/bin/python3 src/experiments.py shacwm-a3


data/shacwm-a7/eval.log: venv/bin/python3
	venv/bin/python3 src/experiments.py shacwm-a7

data/shacrm-a3/eval.log: venv/bin/python3
	venv/bin/python3 src/experiments.py shacrm-a3

data/ppo-a3/eval.log: venv/bin/python3
	venv/bin/python3 src/experiments.py ppo-a3

all: \
	data/shacwm-a3/eval.log \
	data/shacwm-a7/eval.log \
	data/shacrm-a3/eval.log \
	data/ppo-a3/eval.log
