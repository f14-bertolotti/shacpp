import loggers, click

def make_command(where, root):
    @root.group(invoke_without_command=True)
    @click.option("--path", "path", type=click.Path(), default="./train.log")
    @click.pass_obj
    def trainlog(trainer, path):
        file_logger = loggers.File(path)
    
        def wrapper(train_result, **kwargs):
            for data in train_result: file_logger.log(data)
            return {}
        where.add_callback(wrapper)
    
