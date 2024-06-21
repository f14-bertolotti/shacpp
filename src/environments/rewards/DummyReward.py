from environments.rewards.Reward import reward
import click

class DummyReward:
    def __call__(*args, **kwargs): pass

@reward.group(invoke_without_command=True)
@click.pass_obj
def dummy_reward(trainer):
    trainer.environment.set_reward_nn(DummyReward())


