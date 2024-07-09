
def decochain(*decs):
    " chains a sequence of decorators into one "

    def deco(f):
        for dec in reversed(decs): f = dec(f)
        return f

    return deco
