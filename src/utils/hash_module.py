import hashlib
import torch 

def hash_module(module:torch.nn.Module) -> str:
    encoded_string = (str(module) + "".join([str(params) for params in module.parameters()])).encode('utf-8')
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()


