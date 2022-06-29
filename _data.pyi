import numpy


class TripleDataBatch:
    h: numpy.array
    r: numpy.array
    t: numpy.array

    def __init__(self, h: numpy.array, r: numpy.array, t: numpy.array) -> None: ...
