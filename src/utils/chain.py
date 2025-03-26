def chain(*decs):
    def deco(f):
        for dec in reversed(decs): f = dec(f)
        return f
    return deco


