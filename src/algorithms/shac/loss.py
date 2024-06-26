
def loss(new, old):
    return ((new["values"] - old["target_values"])**2).mean()


