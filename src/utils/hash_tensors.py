import hashlib

def hash_tensors(*tensors):
    return hashlib.md5(str([e for tensor in tensors for e in tensor.reshape(-1).tolist()]).encode("utf-8")).hexdigest()
