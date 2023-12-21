import numpy as np
class history_table:  # 历史启发算法
    def __init__(self):
        self.table = np.zeros((2, 90, 90))

    def get_history_score(self, who,  step):
        return self.table[who, step.from_x * 9 + step.from_y, step.to_x * 9 + step.to_y]

    def add_history_score(self, who,  step, depth):
        self.table[who, step.from_x * 9 + step.from_y, step.to_x * 9 + step.to_y] += 2 << depth
# t = history_table()
