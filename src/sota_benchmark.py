from pathlib import Path
from typing import Optional, Callable, Dict, List


from benchmarl.algorithms import MappoConfig, IppoConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments.common import Task
from benchmarl.eval_results import load_and_merge_json_dicts, Plotting
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.utils import DEVICE_TYPING
from matplotlib import pyplot as plt
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase, VmasEnv

from scenarios import Dispersion


class ProxyTask(Task):
    DISPERSION = None
    DISPERSION_4 = None
    DISPERSION_6 = None
    DISPERSION_8 = None
    DISPERSION_10 = None

    def get_agents_from_self(self):
        if self == self.DISPERSION_4:
            return 4
        elif self == self.DISPERSION_6:
            return 6
        elif self == self.DISPERSION_8:
            return 8
        elif self == self.DISPERSION_10:
            return 10

    def get_env_fun(
            self,
            num_envs: int,
            continuous_actions: bool,
            seed: Optional[int],
            device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        self.config = {
            'max_steps': 100,
            'n_agents': self.get_agents_from_self(),
            'n_food': 4,
            'share_rew': True,
            'food_radius': 0.02,
            'penalise_by_time': False
        }

        return lambda: VmasEnv(
            scenario=Dispersion(
                device="cuda:0",
                radius=.05,
                agents=4,
            ),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            group_map={"agents": [f"agent_{i}" for i in range(self.config["n_agents"])]},
            **self.config,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        return None

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.unbatched_observation_spec.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        info_spec = env.unbatched_observation_spec.clone()
        for group in self.group_map(env):
            del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        else:
            return None

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.unbatched_action_spec

    @staticmethod
    def env_name() -> str:
        return "vmas"


configuration = ExperimentConfig.get_from_yaml()
configuration.max_n_frames = 6000
configuration.train_device = "cuda:0"

benchmark = Benchmark(
    algorithm_configs=[
        MappoConfig.get_from_yaml(), IppoConfig.get_from_yaml()
    ],
    tasks=[
        ProxyTask.DISPERSION_4, ProxyTask.DISPERSION_6, ProxyTask.DISPERSION_8, ProxyTask.DISPERSION_10
    ],
    seeds={0},
    experiment_config=configuration,
    model_config=MlpConfig.get_from_yaml(),
    critic_model_config=MlpConfig.get_from_yaml(),
)
# For each experiment, run it and get its output file name
experiments = benchmark.get_experiments()
experiments_json_files = []
for experiment in experiments:
    exp_json_file = str(
        Path(experiment.folder_name) / Path(experiment.name + ".json")
    )
    experiments_json_files.append(exp_json_file)
    experiment.run()

raw_dict = load_and_merge_json_dicts(experiments_json_files)

# Load and process experiment outputs
# raw_dict = load_and_merge_json_dicts(experiments_json_files)
processed_data = Plotting.process_data(raw_dict)
(
    environment_comparison_matrix,
    sample_efficiency_matrix,
) = Plotting.create_matrices(processed_data, env_name="vmas")

# Plotting
Plotting.performance_profile_figure(
    environment_comparison_matrix=environment_comparison_matrix
).savefig("performance_profile.png")
(fig, _) = Plotting.aggregate_scores(
    environment_comparison_matrix=environment_comparison_matrix
)
fig.savefig("aggregate_scores.png")
Plotting.environemnt_sample_efficiency_curves(
    sample_effeciency_matrix=sample_efficiency_matrix
).savefig("environment_sample_efficiency_curves.png")
Plotting.task_sample_efficiency_curves(
    processed_data=processed_data, env="vmas", task="dispersion_4"
).savefig("task_sample_efficiency_curves.png")
Plotting.probability_of_improvement(
    environment_comparison_matrix,
    algorithms_to_compare=[["ippo", "mappo"]],
).savefig("probability_of_improvement.png")

