import click


def make_command(where, root):
    @root.group(invoke_without_command=True)
    @click.pass_obj
    def bar(trainer):
    
        def wrapped(bar, episode, prev_result, **kwargs):
            wrapped.last_eval_result = prev_result.get("eval_reward", wrapped.last_eval_result)
            bar.set_description(f"e:{episode}, r:{wrapped.last_eval_result: <8.5f}")
            return {}
    
        wrapped.last_eval_result = float("NaN")
        where.add_callback(wrapped)
    
