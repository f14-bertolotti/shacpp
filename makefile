PYTHON := /bin/usr/python3

venv/bin/python3:
	rm -rf venv
	${PYTHON} -m venv venv
	venv/bin/python3 -m pip install -r requirements.txt

models/ppo/agent.pkl: venv/bin/python3
	venv/bin/python3 src/cli.py \
    cli agent transformer-agent \
    cli environment scattered \
    cli loss ppo \
    cli optimizer adam \
    cli scheduler cosine \
    cli trajectory ppo \
    cli storage default \
    cli logger file \
    cli train

 
