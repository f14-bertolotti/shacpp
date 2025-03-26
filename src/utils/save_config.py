import json
import os

def save_config(dir:str, config:dict, name="locals.json", indent=4) -> None:
    """ Save the locals dictionary to a json file """
    return json.dump(config, open(os.path.join(dir, name), "w"), indent=indent)

