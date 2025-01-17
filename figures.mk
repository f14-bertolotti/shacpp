PPO_COLOR=1 0.5 0
SHAC_COLOR=1 0.1 0.1
SHACRM_COLOR=0.1 .1 1
SHACWM_COLOR=.6 0.2 .6
REF_COLOR=0 0 0

data/main-transformer.pdf:
	jet init --shape 4 3 \
	jet line --legend none --ax 0 0 --input-path data/ppo/dispersion/1/mlp/42/eval.log --input-path data/ppo/dispersion/1/mlp/43/eval.log --input-path data/ppo/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 0 0 --input-path data/shacwm/dispersion/1/mlp/42/eval.log --input-path data/shacwm/dispersion/1/mlp/43/eval.log --input-path data/shacwm/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 0 --value 1 --color $(REF_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/ppo/dispersion/3/transformer/42/eval.log --input-path data/ppo/dispersion/3/transformer/43/eval.log --input-path data/ppo/dispersion/3/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/shacwm/dispersion/3/transformer/42/eval.log --input-path data/shacwm/dispersion/3/transformer/43/eval.log --input-path data/shacwm/dispersion/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 1 --value 3 --color $(REF_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/ppo/dispersion/5/transformer/42/eval.log --input-path data/ppo/dispersion/5/transformer/43/eval.log --input-path data/ppo/dispersion/5/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/shacwm/dispersion/5/transformer/42/eval.log --input-path data/shacwm/dispersion/5/transformer/43/eval.log --input-path data/shacwm/dispersion/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 2 --value 5 --color $(REF_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/ppo/discovery/1/mlp/42/eval.log --input-path data/ppo/discovery/1/mlp/43/eval.log --input-path data/ppo/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/shacwm/discovery/1/mlp/42/eval.log --input-path data/shacwm/discovery/1/mlp/43/eval.log --input-path data/shacwm/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 0 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/ppo/discovery/3/transformer/42/eval.log --input-path data/ppo/discovery/3/transformer/43/eval.log --input-path data/ppo/discovery/3/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/shacwm/discovery/3/transformer/42/eval.log --input-path data/shacwm/discovery/3/transformer/43/eval.log --input-path data/shacwm/discovery/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 1 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/ppo/discovery/5/transformer/42/eval.log --input-path data/ppo/discovery/5/transformer/43/eval.log --input-path data/ppo/discovery/5/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/shacwm/discovery/5/transformer/42/eval.log --input-path data/shacwm/discovery/5/transformer/43/eval.log --input-path data/shacwm/discovery/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 2 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/ppo/transport/1/mlp/42/eval.log --input-path data/ppo/transport/1/mlp/43/eval.log --input-path data/ppo/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shacwm/transport/1/mlp/42/eval.log --input-path data/shacwm/transport/1/mlp/43/eval.log --input-path data/shacwm/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shac/transport/1/mlp/42/eval.log --input-path data/shac/transport/1/mlp/43/eval.log --input-path data/shac/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 2 0 --value 90 --color $(REF_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/ppo/transport/3/transformer/42/eval.log --input-path data/ppo/transport/3/transformer/43/eval.log --input-path data/ppo/transport/3/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shacwm/transport/3/transformer/42/eval.log --input-path data/shacwm/transport/3/transformer/43/eval.log --input-path data/shacwm/transport/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shac/transport/3/transformer/42/eval.log --input-path data/shac/transport/3/transformer/43/eval.log --input-path data/shac/transport/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 2 1 --value 260 --color $(REF_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/ppo/transport/5/transformer/42/eval.log --input-path data/ppo/transport/5/transformer/43/eval.log --input-path data/ppo/transport/5/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shacwm/transport/5/transformer/42/eval.log --input-path data/shacwm/transport/5/transformer/43/eval.log --input-path data/shacwm/transport/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shac/transport/5/transformer/42/eval.log --input-path data/shac/transport/5/transformer/43/eval.log --input-path data/shac/transport/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 2 2 --value 430 --color $(REF_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/ppo/sampling/1/mlp/42/eval.log --input-path data/ppo/sampling/1/mlp/43/eval.log --input-path data/ppo/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shacwm/sampling/1/mlp/42/eval.log --input-path data/shacwm/sampling/1/mlp/43/eval.log --input-path data/shacwm/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shac/sampling/1/mlp/42/eval.log --input-path data/shac/sampling/1/mlp/43/eval.log --input-path data/shac/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 3 0 --value 130 --color $(REF_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/ppo/sampling/3/transformer/42/eval.log --input-path data/ppo/sampling/3/transformer/43/eval.log --input-path data/ppo/sampling/3/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shacwm/sampling/3/transformer/42/eval.log --input-path data/shacwm/sampling/3/transformer/43/eval.log --input-path data/shacwm/sampling/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shac/sampling/3/transformer/42/eval.log --input-path data/shac/sampling/3/transformer/43/eval.log --input-path data/shac/sampling/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 3 1 --value 600 --color $(REF_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/ppo/sampling/5/transformer/42/eval.log --input-path data/ppo/sampling/5/transformer/43/eval.log --input-path data/ppo/sampling/5/transformer/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shacwm/sampling/5/transformer/42/eval.log --input-path data/shacwm/sampling/5/transformer/43/eval.log --input-path data/shacwm/sampling/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shac/sampling/5/transformer/42/eval.log --input-path data/shac/sampling/5/transformer/43/eval.log --input-path data/shac/sampling/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 3 2 --value 1100 --color $(REF_COLOR) \
	jet mod --ax 0 0 --x-ticks none --y-ticks "0.0,0.5,1.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Dispersion" --title "1 agent" \
	jet mod --ax 0 1 --x-ticks none --y-ticks "0.0,1.5,3.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "3 agents" \
	jet mod --ax 0 2 --x-ticks none --y-ticks "0.0,2.5,5.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "5 agents" \
	jet mod --ax 1 0 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Discovery" \
	jet mod --ax 1 1 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 1 2 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 0 --x-ticks none --y-ticks "0.0,45,90"    --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Transport" \
	jet mod --ax 2 1 --x-ticks none --y-ticks "0.0,130,260"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 2 --x-ticks none --y-ticks "0.0,215,430"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 3 0 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,65,130"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "Sampling" \
	jet mod --ax 3 1 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,300,600" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet mod --ax 3 2 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,550,1100" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet plot --show False --output-path $@

data/main-mlp.pdf:
	jet init --shape 4 3 \
	jet line --legend none --ax 0 0 --input-path data/ppo/dispersion/1/mlp/42/eval.log --input-path data/ppo/dispersion/1/mlp/43/eval.log --input-path data/ppo/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 0 0 --input-path data/shacwm/dispersion/1/mlp/42/eval.log --input-path data/shacwm/dispersion/1/mlp/43/eval.log --input-path data/shacwm/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 0 --value 1 --color $(REF_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/ppo/dispersion/3/mlp/42/eval.log --input-path data/ppo/dispersion/3/mlp/43/eval.log --input-path data/ppo/dispersion/3/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/shacwm/dispersion/3/mlp/42/eval.log --input-path data/shacwm/dispersion/3/mlp/43/eval.log --input-path data/shacwm/dispersion/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 1 --value 3 --color $(REF_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/ppo/dispersion/5/mlp/42/eval.log --input-path data/ppo/dispersion/5/mlp/43/eval.log --input-path data/ppo/dispersion/5/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/shacwm/dispersion/5/mlp/42/eval.log --input-path data/shacwm/dispersion/5/mlp/43/eval.log --input-path data/shacwm/dispersion/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 2 --value 5 --color $(REF_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/ppo/discovery/1/mlp/42/eval.log --input-path data/ppo/discovery/1/mlp/43/eval.log --input-path data/ppo/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/shacwm/discovery/1/mlp/42/eval.log --input-path data/shacwm/discovery/1/mlp/43/eval.log --input-path data/shacwm/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 0 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/ppo/discovery/3/mlp/42/eval.log --input-path data/ppo/discovery/3/mlp/43/eval.log --input-path data/ppo/discovery/3/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/shacwm/discovery/3/mlp/42/eval.log --input-path data/shacwm/discovery/3/mlp/43/eval.log --input-path data/shacwm/discovery/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 1 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/ppo/discovery/5/mlp/42/eval.log --input-path data/ppo/discovery/5/mlp/43/eval.log --input-path data/ppo/discovery/5/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/shacwm/discovery/5/mlp/42/eval.log --input-path data/shacwm/discovery/5/mlp/43/eval.log --input-path data/shacwm/discovery/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 2 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/ppo/transport/1/mlp/42/eval.log --input-path data/ppo/transport/1/mlp/43/eval.log --input-path data/ppo/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shacwm/transport/1/mlp/42/eval.log --input-path data/shacwm/transport/1/mlp/43/eval.log --input-path data/shacwm/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shac/transport/1/mlp/42/eval.log --input-path data/shac/transport/1/mlp/43/eval.log --input-path data/shac/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 2 0 --value 90 --color $(REF_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/ppo/transport/3/mlp/42/eval.log --input-path data/ppo/transport/3/mlp/43/eval.log --input-path data/ppo/transport/3/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shacwm/transport/3/mlp/42/eval.log --input-path data/shacwm/transport/3/mlp/43/eval.log --input-path data/shacwm/transport/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shac/transport/3/mlp/42/eval.log --input-path data/shac/transport/3/mlp/43/eval.log --input-path data/shac/transport/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 2 1 --value 260 --color $(REF_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/ppo/transport/5/mlp/42/eval.log --input-path data/ppo/transport/5/mlp/43/eval.log --input-path data/ppo/transport/5/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shacwm/transport/5/mlp/42/eval.log --input-path data/shacwm/transport/5/mlp/43/eval.log --input-path data/shacwm/transport/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shac/transport/5/mlp/42/eval.log --input-path data/shac/transport/5/mlp/43/eval.log --input-path data/shac/transport/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 2 2 --value 430 --color $(REF_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/ppo/sampling/1/mlp/42/eval.log --input-path data/ppo/sampling/1/mlp/43/eval.log --input-path data/ppo/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shacwm/sampling/1/mlp/42/eval.log --input-path data/shacwm/sampling/1/mlp/43/eval.log --input-path data/shacwm/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shac/sampling/1/mlp/42/eval.log --input-path data/shac/sampling/1/mlp/43/eval.log --input-path data/shac/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 3 0 --value 130 --color $(REF_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/ppo/sampling/3/mlp/42/eval.log --input-path data/ppo/sampling/3/mlp/43/eval.log --input-path data/ppo/sampling/3/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shacwm/sampling/3/mlp/42/eval.log --input-path data/shacwm/sampling/3/mlp/43/eval.log --input-path data/shacwm/sampling/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shac/sampling/3/mlp/42/eval.log --input-path data/shac/sampling/3/mlp/43/eval.log --input-path data/shac/sampling/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 3 1 --value 600 --color $(REF_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/ppo/sampling/5/mlp/42/eval.log --input-path data/ppo/sampling/5/mlp/43/eval.log --input-path data/ppo/sampling/5/mlp/44/eval.log --x message/episode --y message/reward --color $(PPO_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shacwm/sampling/5/mlp/42/eval.log --input-path data/shacwm/sampling/5/mlp/43/eval.log --input-path data/shacwm/sampling/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shac/sampling/5/mlp/42/eval.log --input-path data/shac/sampling/5/mlp/43/eval.log --input-path data/shac/sampling/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHAC_COLOR) \
	jet reference --ax 3 2 --value 1100 --color $(REF_COLOR) \
	jet mod --ax 0 0 --x-ticks none --y-ticks "0.0,0.5,1.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Dispersion" --title "1 agent" \
	jet mod --ax 0 1 --x-ticks none --y-ticks "0.0,1.5,3.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "3 agents" \
	jet mod --ax 0 2 --x-ticks none --y-ticks "0.0,2.5,5.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "5 agents" \
	jet mod --ax 1 0 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Discovery" \
	jet mod --ax 1 1 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 1 2 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 0 --x-ticks none --y-ticks "0.0,45,90"    --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Transport" \
	jet mod --ax 2 1 --x-ticks none --y-ticks "0.0,130,260"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 2 --x-ticks none --y-ticks "0.0,215,430"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 3 0 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,65,130"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "Sampling" \
	jet mod --ax 3 1 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,300,600" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet mod --ax 3 2 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,550,1100" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet plot --show False --output-path $@

data/ablation-transformer.pdf:
	jet init --shape 4 3 \
	jet line --legend none --ax 0 0 --input-path data/shacrm/dispersion/1/mlp/42/eval.log --input-path data/shacrm/dispersion/1/mlp/43/eval.log --input-path data/shacrm/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 0 0 --input-path data/shacwm/dispersion/1/mlp/42/eval.log --input-path data/shacwm/dispersion/1/mlp/43/eval.log --input-path data/shacwm/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 0 --value 1 --color $(REF_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/shacrm/dispersion/3/transformer/42/eval.log --input-path data/shacrm/dispersion/3/transformer/43/eval.log --input-path data/shacrm/dispersion/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/shacwm/dispersion/3/transformer/42/eval.log --input-path data/shacwm/dispersion/3/transformer/43/eval.log --input-path data/shacwm/dispersion/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 1 --value 3 --color $(REF_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/shacrm/dispersion/5/transformer/42/eval.log --input-path data/shacrm/dispersion/5/transformer/43/eval.log --input-path data/shacrm/dispersion/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/shacwm/dispersion/5/transformer/42/eval.log --input-path data/shacwm/dispersion/5/transformer/43/eval.log --input-path data/shacwm/dispersion/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 2 --value 5 --color $(REF_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/shacrm/discovery/1/mlp/42/eval.log --input-path data/shacrm/discovery/1/mlp/43/eval.log --input-path data/shacrm/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/shacwm/discovery/1/mlp/42/eval.log --input-path data/shacwm/discovery/1/mlp/43/eval.log --input-path data/shacwm/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 0 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/shacrm/discovery/3/transformer/42/eval.log --input-path data/shacrm/discovery/3/transformer/43/eval.log --input-path data/shacrm/discovery/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/shacwm/discovery/3/transformer/42/eval.log --input-path data/shacwm/discovery/3/transformer/43/eval.log --input-path data/shacwm/discovery/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 1 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/shacrm/discovery/5/transformer/42/eval.log --input-path data/shacrm/discovery/5/transformer/43/eval.log --input-path data/shacrm/discovery/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/shacwm/discovery/5/transformer/42/eval.log --input-path data/shacwm/discovery/5/transformer/43/eval.log --input-path data/shacwm/discovery/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 2 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shacrm/transport/1/mlp/42/eval.log --input-path data/shacrm/transport/1/mlp/43/eval.log --input-path data/shacrm/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shacwm/transport/1/mlp/42/eval.log --input-path data/shacwm/transport/1/mlp/43/eval.log --input-path data/shacwm/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 2 0 --value 90 --color $(REF_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shacrm/transport/3/transformer/42/eval.log --input-path data/shacrm/transport/3/transformer/43/eval.log --input-path data/shacrm/transport/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shacwm/transport/3/transformer/42/eval.log --input-path data/shacwm/transport/3/transformer/43/eval.log --input-path data/shacwm/transport/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 2 1 --value 260 --color $(REF_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shacrm/transport/5/transformer/42/eval.log --input-path data/shacrm/transport/5/transformer/43/eval.log --input-path data/shacrm/transport/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shacwm/transport/5/transformer/42/eval.log --input-path data/shacwm/transport/5/transformer/43/eval.log --input-path data/shacwm/transport/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 2 2 --value 430 --color $(REF_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shacrm/sampling/1/mlp/42/eval.log --input-path data/shacrm/sampling/1/mlp/43/eval.log --input-path data/shacrm/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shacwm/sampling/1/mlp/42/eval.log --input-path data/shacwm/sampling/1/mlp/43/eval.log --input-path data/shacwm/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 3 0 --value 130 --color $(REF_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shacrm/sampling/3/transformer/42/eval.log --input-path data/shacrm/sampling/3/transformer/43/eval.log --input-path data/shacrm/sampling/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shacwm/sampling/3/transformer/42/eval.log --input-path data/shacwm/sampling/3/transformer/43/eval.log --input-path data/shacwm/sampling/3/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 3 1 --value 600 --color $(REF_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shacrm/sampling/5/transformer/42/eval.log --input-path data/shacrm/sampling/5/transformer/43/eval.log --input-path data/shacrm/sampling/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shacwm/sampling/5/transformer/42/eval.log --input-path data/shacwm/sampling/5/transformer/43/eval.log --input-path data/shacwm/sampling/5/transformer/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 3 2 --value 1100 --color $(REF_COLOR) \
	jet mod --ax 0 0 --x-ticks none --y-ticks "0.0,0.5,1.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Dispersion" --title "1 agent" \
	jet mod --ax 0 1 --x-ticks none --y-ticks "0.0,1.5,3.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "3 agents" \
	jet mod --ax 0 2 --x-ticks none --y-ticks "0.0,2.5,5.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "5 agents" \
	jet mod --ax 1 0 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Discovery" \
	jet mod --ax 1 1 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 1 2 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 0 --x-ticks none --y-ticks "0.0,45,90"    --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Transport" \
	jet mod --ax 2 1 --x-ticks none --y-ticks "0.0,130,260"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 2 --x-ticks none --y-ticks "0.0,215,430"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 3 0 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,65,130"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "Sampling" \
	jet mod --ax 3 1 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,300,600" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet mod --ax 3 2 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,550,1100" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet plot --show False --output-path $@


data/ablation-mlp.pdf:
	jet init --shape 4 3 \
	jet line --legend none --ax 0 0 --input-path data/shacrm/dispersion/1/mlp/42/eval.log --input-path data/shacrm/dispersion/1/mlp/43/eval.log --input-path data/shacrm/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 0 0 --input-path data/shacwm/dispersion/1/mlp/42/eval.log --input-path data/shacwm/dispersion/1/mlp/43/eval.log --input-path data/shacwm/dispersion/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 0 --value 1 --color $(REF_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/shacrm/dispersion/3/mlp/42/eval.log --input-path data/shacrm/dispersion/3/mlp/43/eval.log --input-path data/shacrm/dispersion/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 0 1 --input-path data/shacwm/dispersion/3/mlp/42/eval.log --input-path data/shacwm/dispersion/3/mlp/43/eval.log --input-path data/shacwm/dispersion/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 1 --value 3 --color $(REF_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/shacrm/dispersion/5/mlp/42/eval.log --input-path data/shacrm/dispersion/5/mlp/43/eval.log --input-path data/shacrm/dispersion/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 0 2 --input-path data/shacwm/dispersion/5/mlp/42/eval.log --input-path data/shacwm/dispersion/5/mlp/43/eval.log --input-path data/shacwm/dispersion/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 0 2 --value 5 --color $(REF_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/shacrm/discovery/1/mlp/42/eval.log --input-path data/shacrm/discovery/1/mlp/43/eval.log --input-path data/shacrm/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 1 0 --input-path data/shacwm/discovery/1/mlp/42/eval.log --input-path data/shacwm/discovery/1/mlp/43/eval.log --input-path data/shacwm/discovery/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 0 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/shacrm/discovery/3/mlp/42/eval.log --input-path data/shacrm/discovery/3/mlp/43/eval.log --input-path data/shacrm/discovery/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 1 1 --input-path data/shacwm/discovery/3/mlp/42/eval.log --input-path data/shacwm/discovery/3/mlp/43/eval.log --input-path data/shacwm/discovery/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 1 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/shacrm/discovery/5/mlp/42/eval.log --input-path data/shacrm/discovery/5/mlp/43/eval.log --input-path data/shacrm/discovery/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 1 2 --input-path data/shacwm/discovery/5/mlp/42/eval.log --input-path data/shacwm/discovery/5/mlp/43/eval.log --input-path data/shacwm/discovery/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 1 2 --value 7 --color $(REF_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shacrm/transport/1/mlp/42/eval.log --input-path data/shacrm/transport/1/mlp/43/eval.log --input-path data/shacrm/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 2 0 --input-path data/shacwm/transport/1/mlp/42/eval.log --input-path data/shacwm/transport/1/mlp/43/eval.log --input-path data/shacwm/transport/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 2 0 --value 90 --color $(REF_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shacrm/transport/3/mlp/42/eval.log --input-path data/shacrm/transport/3/mlp/43/eval.log --input-path data/shacrm/transport/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 2 1 --input-path data/shacwm/transport/3/mlp/42/eval.log --input-path data/shacwm/transport/3/mlp/43/eval.log --input-path data/shacwm/transport/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 2 1 --value 260 --color $(REF_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shacrm/transport/5/mlp/42/eval.log --input-path data/shacrm/transport/5/mlp/43/eval.log --input-path data/shacrm/transport/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 2 2 --input-path data/shacwm/transport/5/mlp/42/eval.log --input-path data/shacwm/transport/5/mlp/43/eval.log --input-path data/shacwm/transport/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 2 2 --value 430 --color $(REF_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shacrm/sampling/1/mlp/42/eval.log --input-path data/shacrm/sampling/1/mlp/43/eval.log --input-path data/shacrm/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 3 0 --input-path data/shacwm/sampling/1/mlp/42/eval.log --input-path data/shacwm/sampling/1/mlp/43/eval.log --input-path data/shacwm/sampling/1/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 3 0 --value 130 --color $(REF_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shacrm/sampling/3/mlp/42/eval.log --input-path data/shacrm/sampling/3/mlp/43/eval.log --input-path data/shacrm/sampling/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 3 1 --input-path data/shacwm/sampling/3/mlp/42/eval.log --input-path data/shacwm/sampling/3/mlp/43/eval.log --input-path data/shacwm/sampling/3/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 3 1 --value 600 --color $(REF_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shacrm/sampling/5/mlp/42/eval.log --input-path data/shacrm/sampling/5/mlp/43/eval.log --input-path data/shacrm/sampling/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACRM_COLOR) \
	jet line --legend none --ax 3 2 --input-path data/shacwm/sampling/5/mlp/42/eval.log --input-path data/shacwm/sampling/5/mlp/43/eval.log --input-path data/shacwm/sampling/5/mlp/44/eval.log --x message/episode --y message/reward --color $(SHACWM_COLOR) \
	jet reference --ax 3 2 --value 1100 --color $(REF_COLOR) \
	jet mod --ax 0 0 --x-ticks none --y-ticks "0.0,0.5,1.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Dispersion" --title "1 agent" \
	jet mod --ax 0 1 --x-ticks none --y-ticks "0.0,1.5,3.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "3 agents" \
	jet mod --ax 0 2 --x-ticks none --y-ticks "0.0,2.5,5.0"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" --title "5 agents" \
	jet mod --ax 1 0 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Discovery" \
	jet mod --ax 1 1 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 1 2 --x-ticks none --y-ticks "0.0,5.0,10"   --y-lim 0 10 --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 0 --x-ticks none --y-ticks "0.0,45,90"    --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "Transport" \
	jet mod --ax 2 1 --x-ticks none --y-ticks "0.0,130,260"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 2 2 --x-ticks none --y-ticks "0.0,215,430"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine False --x-label "" --y-label "" \
	jet mod --ax 3 0 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,65,130"  --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "Sampling" \
	jet mod --ax 3 1 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,300,600" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet mod --ax 3 2 --x-ticks default --x-ticks "0,10000,20000" --y-ticks "0.0,550,1100" --x-lim 0 20000 --right-spine False --top-spine False --bottom-spine True --x-label "Episode" --y-label "" \
	jet plot --show False --output-path $@

all: \
	data/main-transformer.pdf \
	data/main-mlp.pdf \
	data/ablation-transformer.pdf \
	data/ablation-mlp.pdf
