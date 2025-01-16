# To use this makefile you need to have jet (commit 1616195) installed, see:
# https://github.com/f14-bertolotti/jet

PPO_COLOR=1 0.5 0
SHAC_COLOR=1 0.1 0.1
SHACRM_COLOR=0.1 .1 1
SHACWM_COLOR=.6 0.2 .6
REF_COLOR=0 0 0

define make-discovery
data/discovery-$1-$2.pdf: 
	jet init --shape 1 1 --font-size 24 \
	jet mod --x-bins 7 \
	jet line --color ${PPO_COLOR} --input-path data/ppo/discovery/$1/$2/44/eval.log --input-path data/ppo/discovery/$1/$2/43/eval.log  --input-path data/ppo/discovery/$1/$2/42/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${SHACWM_COLOR}  --input-path data/shacwm/discovery/$1/$2/44/eval.log  --input-path data/shacwm/discovery/$1/$2/43/eval.log  --input-path data/shacwm/discovery/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${REF_COLOR}  --input-path data/ppo/discovery/$1/$2/44/eval.log --input-path data/ppo/discovery/$1/$2/43/eval.log  --input-path data/ppo/discovery/$1/$2/42/eval.log --input-path data/shacwm/discovery/$1/$2/44/eval.log  --input-path data/shacwm/discovery/$1/$2/43/eval.log  --input-path data/shacwm/discovery/$1/$2/42/eval.log --x message/episode --y message/max_reward --label max --linestyle "--" \
	jet legend \
		--frameon False \
		--line ppo ${PPO_COLOR} 1 "-" \
		--line shac++ ${SHACWM_COLOR} 1 "-" \
		--line max ${REF_COLOR} 1 "--" \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/discovery-$1-$2.pdf

data/discovery-ablation-$1-$2.pdf:
	jet init --shape 1 1 --font-size 24 \
	jet mod --x-bins 7 \
	jet line --color ${SHACRM_COLOR} --input-path data/shacrm/discovery/$1/$2/44/eval.log  --input-path data/shacrm/discovery/$1/$2/43/eval.log  --input-path data/shacrm/discovery/$1/$2/42/eval.log --x message/episode --y message/reward --label shacrm \
	jet line --color ${SHACWM_COLOR}  --input-path data/shacwm/discovery/$1/$2/44/eval.log  --input-path data/shacwm/discovery/$1/$2/43/eval.log  --input-path data/shacwm/discovery/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${REF_COLOR}  --input-path data/shacrm/discovery/$1/$2/44/eval.log  --input-path data/shacrm/discovery/$1/$2/43/eval.log  --input-path data/shacrm/discovery/$1/$2/42/eval.log --input-path data/shacwm/discovery/$1/$2/44/eval.log  --input-path data/shacwm/discovery/$1/$2/43/eval.log  --input-path data/shacwm/discovery/$1/$2/42/eval.log --x message/episode --y message/max_reward --label max --linestyle "--" \
	jet legend \
		--frameon False \
		--line shac+ ${SHACRM_COLOR} 1 "-" \
		--line shac++ ${SHACWM_COLOR} 1 "-" \
		--line max ${REF_COLOR} 1 "--" \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/discovery-ablation-$1-$2.pdf

endef

define make-transport
data/transport-$1-$2.pdf: 
	jet init --shape 1 1 --font-size 24 \
	jet mod --x-bins 7 \
	jet line --color ${PPO_COLOR}   --input-path data/ppo/transport/$1/$2/44/eval.log --input-path data/ppo/transport/$1/$2/43/eval.log --input-path data/ppo/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${SHAC_COLOR}   --input-path data/shac/transport/$1/$2/44/eval.log --input-path data/shac/transport/$1/$2/43/eval.log --input-path data/shac/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label shac \
	jet line --color ${SHACWM_COLOR} --input-path data/shacwm/transport/$1/$2/44/eval.log --input-path data/shacwm/transport/$1/$2/43/eval.log --input-path data/shacwm/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${REF_COLOR} --input-path data/shacwm/transport/$1/$2/44/eval.log --input-path data/shacwm/transport/$1/$2/43/eval.log --input-path data/shacwm/transport/$1/$2/42/eval.log --input-path data/ppo/transport/$1/$2/44/eval.log --input-path data/ppo/transport/$1/$2/43/eval.log --input-path data/ppo/transport/$1/$2/42/eval.log  --input-path data/shac/transport/$1/$2/44/eval.log --input-path data/shac/transport/$1/$2/43/eval.log --input-path data/shac/transport/$1/$2/42/eval.log  --x message/episode --y message/max_reward --label max --linestyle "--" \
	jet legend \
		--frameon False \
		--line ppo ${PPO_COLOR} 1 "-" \
		--line shac ${SHAC_COLOR} 1 "-" \
		--line shac++ ${SHACWM_COLOR} 1 "-" \
		--line max ${REF_COLOR} 1 "--" \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/transport-$1-$2.pdf

data/transport-ablation-$1-$2.pdf:
	jet init --shape 1 1 --font-size 24 \
	jet mod --x-bins 7 \
	jet line --color ${SHACRM_COLOR}  --input-path data/shacrm/transport/$1/$2/44/eval.log  --input-path data/shacrm/transport/$1/$2/43/eval.log  --input-path data/shacrm/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label shacrm \
	jet line --color ${SHACWM_COLOR}  --input-path data/shacwm/transport/$1/$2/44/eval.log  --input-path data/shacwm/transport/$1/$2/43/eval.log  --input-path data/shacwm/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${REF_COLOR} --input-path data/shacrm/transport/$1/$2/44/eval.log  --input-path data/shacrm/transport/$1/$2/43/eval.log  --input-path data/shacrm/transport/$1/$2/42/eval.log  --input-path data/shacwm/transport/$1/$2/44/eval.log  --input-path data/shacwm/transport/$1/$2/43/eval.log  --input-path data/shacwm/transport/$1/$2/42/eval.log --x message/episode --y message/max_reward --label max --linestyle "--" \
	jet legend \
		--frameon False \
		--line shac+ ${SHACRM_COLOR} 1 "-" \
		--line shac++ ${SHACWM_COLOR} 1 "-" \
		--line max ${REF_COLOR} 1 "--" \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/transport-ablation-$1-$2.pdf

endef

define make-dispersion
data/dispersion-$1-$2.pdf: 
	jet init --shape 1 1 --font-size 24 \
	jet mod --x-bins 7 \
	jet line --color ${PPO_COLOR}   --input-path data/ppo/dispersion/$1/$2/44/eval.log --input-path data/ppo/dispersion/$1/$2/43/eval.log  --input-path data/ppo/dispersion/$1/$2/42/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${SHACWM_COLOR}  --input-path data/shacwm/dispersion/$1/$2/44/eval.log  --input-path data/shacwm/dispersion/$1/$2/43/eval.log  --input-path data/shacwm/dispersion/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${REF_COLOR} --input-path data/ppo/dispersion/$1/$2/44/eval.log --input-path data/ppo/dispersion/$1/$2/43/eval.log  --input-path data/ppo/dispersion/$1/$2/42/eval.log  --input-path data/shacwm/dispersion/$1/$2/44/eval.log  --input-path data/shacwm/dispersion/$1/$2/43/eval.log  --input-path data/shacwm/dispersion/$1/$2/42/eval.log --x message/episode --y message/max_reward --label max --linestyle "--" \
	jet legend \
		--frameon False \
		--line ppo ${PPO_COLOR} 1 "-" \
		--line shac++ ${SHACWM_COLOR} 1 "-" \
		--line max ${REF_COLOR} 1 "--" \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/dispersion-$1-$2.pdf

data/dispersion-ablation-$1-$2.pdf:
	jet init --shape 1 1 --font-size 24 \
	jet mod --x-bins 7 \
	jet line --color ${SHACRM_COLOR}  --input-path data/shacrm/dispersion/$1/$2/44/eval.log  --input-path data/shacrm/dispersion/$1/$2/43/eval.log  --input-path data/shacrm/dispersion/$1/$2/42/eval.log --x message/episode --y message/reward --label shacrm \
	jet line --color ${SHACWM_COLOR}  --input-path data/shacwm/dispersion/$1/$2/44/eval.log  --input-path data/shacwm/dispersion/$1/$2/43/eval.log  --input-path data/shacwm/dispersion/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${REF_COLOR}  --input-path data/shacrm/dispersion/$1/$2/44/eval.log  --input-path data/shacrm/dispersion/$1/$2/43/eval.log  --input-path data/shacrm/dispersion/$1/$2/42/eval.log --input-path data/shacwm/dispersion/$1/$2/44/eval.log  --input-path data/shacwm/dispersion/$1/$2/43/eval.log  --input-path data/shacwm/dispersion/$1/$2/42/eval.log --x message/episode --y message/max_reward --label max --linestyle "--" \
	jet legend \
		--frameon False \
		--line shac+ ${SHACRM_COLOR} 1 "-" \
		--line shac++ ${SHACWM_COLOR} 1 "-" \
		--line max ${REF_COLOR} 1 "--" \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/dispersion-ablation-$1-$2.pdf

endef

AGENTS=1 3 5
MODELS=mlp transformer

$(foreach g,$(AGENTS),$(foreach m,$(MODELS),$(eval $(call make-transport,$g,$m))))
$(foreach g,$(AGENTS),$(foreach m,$(MODELS),$(eval $(call make-dispersion,$g,$m))))
$(foreach g,$(AGENTS),$(foreach m,$(MODELS),$(eval $(call make-discovery,$g,$m))))

transport-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/transport-$g-$m.pdf))
dispersion-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/dispersion-$g-$m.pdf))
discovery-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/discovery-$g-$m.pdf))
dispersion-ablation-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/dispersion-ablation-$g-$m.pdf))
transport-ablation-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/transport-ablation-$g-$m.pdf))
discovery-ablation-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/discovery-ablation-$g-$m.pdf))

data/grads-transformer-transport.pdf:
	jet init --shape 1 1 --font-size 24 \
	jet mod --x-bins 7 \
	jet line --color ${SHAC_COLOR}   --input-path data/withgrads/shac/transport/5/transformer/42/policy.log   --x message/episode --y message/grads \
	jet line --color ${SHACWM_COLOR} --input-path data/withgrads/shacwm/transport/5/transformer/42/policy.log --x message/episode --y message/grads \
	jet legend \
		--frameon False \
		--line shac++ ${SHACWM_COLOR} 1 "-" \
		--line shac ${SHAC_COLOR} 1 "-" \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "||∇f(x)||" \
	jet plot --show True --output-path $@


all: transport-fig disperion-fig transport-ablation-fig dispersion-ablation-fig discovery-fig discovery-ablation-fig
clean:
	rm -f data/*.pdf
