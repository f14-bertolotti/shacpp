import jsonpickle, click

def make_command(where, root):
    @root.group(invoke_without_command=True)
    @click.option("--path", "path", type=click.Path(), default="agent.pkl")
    @click.pass_obj
    def save_configuration(trainer, path):
        def wrapper(episode, **kwargs):
            if episode == 1: 
                with open(path, "w") as file: 
                    file.write(str(jsonpickle.encode(trainer,indent=4)))
    
            return {}
        where.add_callback(wrapper)
    
