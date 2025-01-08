# To use this makefile you need to have jet (commit fded644) installed, see:
# https://github.com/f14-bertolotti/jet

PPO_COLOR=1 0.5 0
SHAC_COLOR=1 0.1 0.1
SHACRM_COLOR=0.1 .1 1
SHACWM_COLOR=.6 0.2 .6

define make-transport
data/transport-$1-$2.pdf: 
	jet init --shape 1 1 --font-size 20 \
	jet mod --x-bins 7 \
	jet line --color ${PPO_COLOR}    --input-path data/ppo/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${PPO_COLOR}    --input-path data/ppo/transport/$1/$2/43/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${PPO_COLOR}    --input-path data/ppo/transport/$1/$2/44/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${SHAC_COLOR}   --input-path data/shac/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label shac \
	jet line --color ${SHAC_COLOR}   --input-path data/shac/transport/$1/$2/43/eval.log --x message/episode --y message/reward --label shac \
	jet line --color ${SHAC_COLOR}   --input-path data/shac/transport/$1/$2/44/eval.log --x message/episode --y message/reward --label shac \
	jet line --color ${SHACWM_COLOR} --input-path data/shacwm/transport/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${SHACWM_COLOR} --input-path data/shacwm/transport/$1/$2/43/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${SHACWM_COLOR} --input-path data/shacwm/transport/$1/$2/44/eval.log --x message/episode --y message/reward --label shacwm \
	jet legend \
		--frameon False \
		--line ppo ${PPO_COLOR} 1 \
		--line shac ${SHAC_COLOR} 1 \
		--line shac++ ${SHACWM_COLOR} 1 \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/transport-$1-$2.pdf
endef

define make-dispersion
data/dispersion-$1-$2.pdf: 
	jet init --shape 1 1 --font-size 20 \
	jet mod --x-bins 7 \
	jet line --color ${PPO_COLOR}    --input-path data/ppo/dispersion/$1/$2/42/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${PPO_COLOR}    --input-path data/ppo/dispersion/$1/$2/43/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${PPO_COLOR}    --input-path data/ppo/dispersion/$1/$2/44/eval.log --x message/episode --y message/reward --label ppo \
	jet line --color ${SHACWM_COLOR} --input-path data/shacwm/dispersion/$1/$2/42/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${SHACWM_COLOR} --input-path data/shacwm/dispersion/$1/$2/43/eval.log --x message/episode --y message/reward --label shacwm \
	jet line --color ${SHACWM_COLOR} --input-path data/shacwm/dispersion/$1/$2/44/eval.log --x message/episode --y message/reward --label shacwm \
	jet legend \
		--frameon False \
		--line ppo ${PPO_COLOR} 1 \
		--line shac++ ${SHACWM_COLOR} 1 \
	jet mod \
		--right-spine False \
		--top-spine False \
		--x-label "episode" \
		--y-label "reward" \
	jet plot --show False --output-path data/dispersion-$1-$2.pdf
endef

AGENTS=1 3 5
MODELS=mlp transformer



$(foreach g,$(AGENTS),$(foreach m,$(MODELS),$(eval $(call make-transport,$g,$m))))
$(foreach g,$(AGENTS),$(foreach m,$(MODELS),$(eval $(call make-dispersion,$g,$m))))

transport-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/transport-$g-$m.pdf))
dispersion-fig: $(foreach g,$(AGENTS),$(foreach m,$(MODELS),data/dispersion-$g-$m.pdf))
all: transport-fig dispersion-fig
