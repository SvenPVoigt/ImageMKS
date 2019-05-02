def circular_check(r, size):
    if size is None:
        return (round(2*r+1), round(2*r+1))
    else:
        assert isinstance(size, list) or isinstance(size, tuple)
        return size


def oriented_check(r, size):
    if size is None:
        return (round(2*r+1), round(2*r+1))
