import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib, numpy, torch, click
import tqdm

from moviepy.editor import ImageSequenceClip

@click.command()
@click.pass_obj
@click.option("--steps"      , "steps"     , type=int          , default=1024   , help="no steps for the gif")
@click.option("--x-lim"      , "xlim"      , type=(float,float), default=(-3,+3), help="left and right limits for the x axis")
@click.option("--y-lim"      , "ylim"      , type=(float,float), default=(-3,+3), help="left and right limits for the y axis")
@click.option("--interval"   , "interval"  , type=int          , default=60     , help="matplolib animation interval")
@click.option("--output-path", "outputpath", type=click.Path() , default=None   , help="path in which to save the gif")
@click.option("--show"       , "show"      , type=bool         , default=False  , help="True if the animation should be displayed right away")
@click.option("--use-logits" , "uselogits" , type=bool         , default=True   , help="True if logits should be used instead of sampled actions")
@staticmethod
def viz(trainer, steps, uselogits, xlim, ylim, interval, show, outputpath):

    # get test trajectory 
    frames, observations, current_observation = [], [], torch.stack(trainer.environment.reset()).transpose(0,1) 

    for step in tqdm.tqdm(range(0, steps)):
        with torch.no_grad(): agent_result = trainer.agent.get_action(trainer.environment.normalize(current_observation.unsqueeze(0)).squeeze(0))
        
        actions = agent_result["logits"] if uselogits else agent_result["actions"]
        actions = actions.transpose(0,1)

        next_observations, rewards, done, info  = trainer.environment.step(actions)
        current_observation = torch.stack(next_observations).transpose(0,1)
        frames.append(trainer.environment.render(mode="rgb_array"))
        observations.append(current_observation.cpu().squeeze(0).detach())
   
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_gif(f'play.gif', fps=30)
    #observations = torch.stack(observations)

    ## plot said trajectory
    #fig, ax = plt.subplots()

    ## this is highly specific for the environment, should be refactored
    #scatterplot = ax.scatter(
    #    x = observations[0,:,0].numpy(),
    #    y = observations[0,:,1].numpy(),
    #    color = "black"
    #)
    #
    #ax.set_xlim(*xlim)
    #ax.set_ylim(*ylim)
    #
    #def update(frame):
    #    x = observations[frame,:,0]
    #    y = observations[frame,:,1]
    #    data = numpy.stack([x, y]).T
    #    scatterplot.set_offsets(data)
    #    return scatterplot
    #
    #animation = matplotlib.animation.FuncAnimation(fig=fig, func=update, frames=observations.size(0), interval=interval)
    #if outputpath: animation.save(outputpath)
    #if show: plt.show()



