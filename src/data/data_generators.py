def get_x_and_y_generator(filepath):
    stream = open(filepath)

    def x_and_y_generator():
        x = next(stream)
        y = next(stream)
        yield x, y

    return x_and_y_generator()


class XAndYGenerator:
    """Data files in mathematics dataset are written with inputs and targets on alternate lines.
       This class implements a generator that yields (input, targets) tuples for any file written in that format."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.stream = open(filepath)

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.stream)
        y = next(self.stream)
        return x, y

    def close(self):
        if not self.stream.closed:
            self.stream.close()
