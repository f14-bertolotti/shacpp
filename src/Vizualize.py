import torch, tqdm, click

from moviepy.editor import ImageSequenceClip

@click.command()
@click.pass_obj
@click.option("--steps"      , "steps"     , type=int          , default=64        , help="no steps for the gif")
@click.option("--output-path", "outputpath", type=click.Path() , default="play.gif", help="path in which to save the gif")
@click.option("--use-logits" , "uselogits" , type=bool         , default=True      , help="True if logits should be used instead of sampled actions")
@staticmethod
def viz(trainer, steps, uselogits, outputpath):

    frames, current_observation = [], trainer.environment.reset()

    for step in tqdm.tqdm(range(0, steps)):
        agent_result = trainer.agent.get_action(trainer.environment.normalize(current_observation))

        envir_result = trainer.environment.step(agent_result["logits"] if uselogits else agent_result["actions"])
        current_observation = envir_result["observation"]

        frames.append(trainer.environment.render(mode="rgb_array"))
   
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_gif(outputpath, fps=30)

