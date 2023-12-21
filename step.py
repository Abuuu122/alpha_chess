class Step:  # 走法类
    def __init__(self, from_x=-1, from_y=-1, to_x=-1, to_y=-1):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.score = 0

    def __cmp__(self, other):
        return self.score < other.score

    def __lt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.score == other.score

    def __str__(self):
        return "[{}:{}]".format(self.__class__.__name__, ",".join("{}={}".format(k, getattr(self, k)) for k in self.__dict__.keys()))