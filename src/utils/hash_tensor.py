import hashlib
import torch

def hash_tensor(tensor:torch.Tensor) -> str:
    encoded_string = str(tensor).encode('utf-8')
    hash_object = hashlib.sha256(encoded_string)
    return hash_object.hexdigest()


