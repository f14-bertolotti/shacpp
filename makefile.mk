EPISODES=20000
COMPILE=True

venv/bin/python3:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

# rule to generate all possible targets
define make-targets
data/$1/$2/$3/mlp/$4/done: venv/bin/python3
	CUBLAS_WORKSPACE_CONFIG=":4096:8" venv/bin/python3 src/experiments.py \
		modelcfg --model policy --nn mlp --hidden-size $(shell echo $$((32 * $(3)))) \
		modelcfg --model value  --nn mlp --hidden-size $(shell echo $$((32 * $(3)))) \
		modelcfg --model reward --nn mlp --hidden-size $(shell echo $$((32 * $(3)))) \
		run --alg-name $1 --env-name $2 --agents $3 --seed $4 --compile $(COMPILE) --episodes $(EPISODES)

data/$1/$2/$3/transformer/$4/done: venv/bin/python3
	CUBLAS_WORKSPACE_CONFIG=":4096:8" venv/bin/python3 src/experiments.py \
		modelcfg --model policy --nn transformer --hidden-size $(shell echo $$((32 * $(3)))) \
		modelcfg --model value  --nn transformer --hidden-size $(shell echo $$((32 * $(3)))) \
		modelcfg --model reward --nn transformer --hidden-size $(shell echo $$((32 * $(3)))) \
		run --alg-name $1 --env-name $2 --agents $3 --seed $4 --compile $(COMPILE) --episodes $(EPISODES)

clean-$1-$2-$3-mlp-$4:
	rm -rf data/$1/$2/$3/mlp/$4

clean-$1-$2-$3-transformer-$4:
	rm -rf data/$1/$2/$3/transformer/$4

endef

# generate all possible targets
ALGOS=ppo shac shacrm shacwm
ENVS=transport dispersion sampling discovery
AGENTS=1 3 5
MODELS=mlp transformer
SEEDS=42 43 44
$(foreach a,$(ALGOS),$(foreach e,$(ENVS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),$(eval $(call make-targets,$a,$e,$g,$s))))))

# groups of targets
transport-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/transport/$g/mlp/$s/done)))
transport-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/transport/$g/transformer/$s/done)))
sampling-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/sampling/$g/mlp/$s/done)))
sampling-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/sampling/$g/transformer/$s/done)))

# the following two are not differentiable, so no point in using shac or shacrm
ALGOS=ppo shacwm
dispersion-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/dispersion/$g/mlp/$s/done)))
dispersion-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/dispersion/$g/transformer/$s/done)))
discovery-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/discovery/$g/mlp/$s/done)))
discovery-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),data/$a/discovery/$g/transformer/$s/done)))

clean-transport-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-transport-$g-mlp-$s)))
clean-transport-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-transport-$g-transformer-$s)))
clean-dispersion-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-dispersion-$g-mlp-$s)))
clean-dispersion-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-dispersion-$g-transformer-$s)))
clean-sampling-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-sampling-$g-mlp-$s)))
clean-sampling-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-sampling-$g-transformer-$s)))
clean-discovery-mlp: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-discovery-$g-mlp-$s)))
clean-discovery-transformer: $(foreach a,$(ALGOS),$(foreach g,$(AGENTS),$(foreach s,$(SEEDS),clean-$a-discovery-$g-transformer-$s)))

all: \
	transport-mlp \
	transport-transformer \
	dispersion-mlp \
	dispersion-transformer \
	sampling-mlp \
	sampling-transformer \
	discovery-mlp \
	discovery-transformer

# clean target
clean: \
	clean-transport-mlp \
	clean-transport-transformer \
	clean-dispersion-mlp \
	clean-dispersion-transformer \
	clean-sampling-mlp \
	clean-sampling-transformer \
	clean-discovery-mlp \
	clean-discovery-transformer

