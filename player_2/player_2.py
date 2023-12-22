import random

from utils import get_legal_actions
# import xxx    # Here may be other package you want to import
import history_heuristic as hh
import chess_constants as cc
from step import Step
import my_relation as mr



class Player(): # please do not change the class name

    def __init__(self, side: str):
        """
        Variables:
            - self.side: specifies which side your agent takes. It must be "red" or "black".
            - self.history: records history actions.
            - self.move and self.move_back: when you do "search" or "rollout", you can utilize these two methods 
                to simulate the change of the board as the effect of actions and update self.history accordingly.
            - self.name : for you to set a name for your player. It is "Player" by default.

        Methods:
            - policy: the core method for you to implement. It must return a legal action according to the input 
                board configuration. Return values must be a four-element tuple or list in the form 
                of (old_x, old_y, new_x, new_y), with the x coordinate representing the column number 
                and the y coordinate representing the row number.
            - move: simulating movement, moving a piece from (old_x, old_y) to (new_x, new_y) 
                and eating a piece when overlap happens.
            - move_back: restoring the last move. You need to use it when backtracing along a path during a search,
                 so that both the board and self.history are reverted correctly.
        """

        self.side = side    # don't change
        self.history = []   # don't change
        self.name = "Player"    # please change to your group name
        self.max_depth = 4
        self.history_table = hh.history_table()
        self.best_move = Step()

    def policy(self, board: tuple): # the core method for you to implement
        """
        You should complement this method.

        Args:
            - board is a 10×9 matrix, showing current game state.
                board[i][j] > 0 means a red piece is on position (i,j)
                board[i][j] < 0 means a black piece is on position (i,j)
                board[i][j] = 0 means position (i,j) is empty.

        Returns:
            - Your return value is a four-element tuple (i,j,x,y), 
              which means your next action is to move your piece from (i,j) to (x,y).
            Note that your return value must be illegal. Otherwise you will lose the game directly.
        """
        
        # action_list = get_legal_actions(board, self.side, self.history) # get all actions that are legal to choose from
        # return random.choice(action_list)   # here we demonstrate the most basic player
        self.alpha_beta(board, self.max_depth, cc.min_val, cc.max_val)
        return (self.best_move.from_x, self.best_move.from_y, self.best_move.to_x, self.best_move.to_y)


    def move(self, board, old_x, old_y, new_x, new_y):  # don't change
        """utility function provided by us: simulate the effect of a movement"""

        eaten_id = board[new_x][new_y]
        board[new_x][new_y] = board[old_x][old_y]
        board[old_x][old_y] = 0
        self.history.append((old_x,old_y,new_x,new_y,eaten_id))

    def move_back(self, board, old_x, old_y, new_x, new_y): # don't change
        """utility function provided by us: restore or reverse the effect of a movement"""

        board[old_x][old_y] = board[new_x][new_y]
        board[new_x][new_y] = self.history[-1][4]
        self.history.pop()

    def update_history(self, current_game_history: list): 
        """to refresh your self.history after each actual play, which is taken care externally"""

        self.history = current_game_history
        
    def get_name(self):
        """used by the external logger"""

        return self.name

    def your_method(self): # Here may be other method you want to add in Player
        pass

    def is_game_over(self, who, board):  # 判断游戏是否结束
        cnt = 0
        for i in range(10):
            for j in range(9):
                if abs(board[i][j]) == 7:
                    cnt = cnt + 1
        if cnt == 2:
            return False
        return True

    def alpha_beta(self, board, depth, alpha, beta):  # alpha-beta剪枝，alpha是大可能下界，beta是最小可能上界
        who = (self.max_depth - depth) % 2  # 那个玩家
        if self.is_game_over(who, board):  # 判断是否游戏结束，如果结束了就不用搜了
            return cc.min_val
        if depth == 1:  # 搜到指定深度了，也不用搜了
            # print(self.evaluate(who))
            return self.evaluate(who, board)
        ## 返回所有能走的方法
        _move_list = get_legal_actions(board, self.side, self.history)
        move_list = []
        for move in _move_list:
            move_list.append(Step(move[0], move[1], move[2], move[3]))
        # 利用历史表0
        for i in range(len(move_list)):
            move_list[i].score = self.history_table.get_history_score(who, move_list[i])
        move_list.sort()  # 为了让更容易剪枝利用历史表得分进行排序
        # # for item in move_list:
        # #     print(item.score)
        # # print('----------------------')
        best_step = move_list[0]

        score_list = []
        for step in move_list:
            # temp = self.move_to(step)
            self.move(board, step.from_x, step.from_y, step.to_x, step.to_y)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha)  # 因为是一层选最大一层选最小，所以利用取负号来实现
            score_list.append(score)
            # self.undo_move(step, temp)
            self.move_back(board, step.from_x, step.from_y, step.to_x, step.to_y)
            if score > alpha:
                alpha = score
                if depth == self.max_depth:
                    self.best_move = step
                best_step = step
            if alpha >= beta:
                best_step = step
                break
        # print(score_list)
        # 更新历史表
        if best_step is not None:
            self.history_table.add_history_score(who, best_step, depth)
        return alpha

    def evaluate(self, who, board):  # who表示该谁走，返回评分值
        # self.cnt += 1
        # print('====================================================================================')
        relation_list = self.init_relation_list()
        base_val = [0, 0]
        pos_val = [0, 0]
        mobile_val = [0, 0]
        relation_val = [0, 0]
        for x in range(10):
            for y in range(9):
                now_chess = board[x][y]
                type = abs(now_chess)
                if type == 0:
                    continue
                # now = 0 if who == 0 else 1
                if self.side == "red":
                    if who == 1:
                        if now_chess > 0:
                            now = 0
                        else:
                            now = 1
                    else:
                        if now_chess < 0:
                            now = 0
                        else:
                            now = 1
                elif self.side == "black":
                    if who == 1:
                        if now_chess < 0:
                            now = 0
                        else:
                            now = 1
                    else:
                        if now_chess > 0:
                            now = 0
                        else:
                            now = 1
                pos = x * 9 + y
                # temp_move_list = self.board.get_chess_move(x, y, now, True)
                #计算基础价值
                base_val[now] += cc.base_val[type]
                # 计算位置价值
                # if now == 0:  # 如果是要求最大值的玩家
                #     pos_val[now] += cc.pos_val[type][pos]
                # else:
                #     pos_val[now] += cc.pos_val[type][89 - pos]
                if self.side == "red":
                    if now == 0:  # 如果是要求最大值的玩家
                        pos_val[now] += cc.pos_val[type][pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {cc.pos_val[type][pos]}")
                    elif now == 1:
                        pos_val[now] += cc.pos_val[type][89 - pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {cc.pos_val[type][89-pos]}")
                elif self.side == "black":
                    if now == 0:  # 如果是要求最大值的玩家
                        pos_val[now] += cc.pos_val[type][89 - pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {cc.pos_val[type][89 - pos]}")
                    elif now == 1:
                        pos_val[now] += cc.pos_val[type][pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {cc.pos_val[type][pos]}")

        #         # 计算机动性价值，记录关系信息
        #         for item in temp_move_list:
        #             # print('----------------')
        #             # print(item)
        #             temp_chess = self.board.board[item.to_x][item.to_y]  # 目的位置的棋子
        #
        #             if temp_chess.chess_type == cc.kong:  # 如果是空，那么加上机动性值
        #                 # print('ok')
        #                 mobile_val[now] += cc.mobile_val[type]
        #                 # print(mobile_val[now])
        #                 continue
        #             elif temp_chess.belong != now:  # 如果不是自己一方的棋子
        #                 # print('ok1')
        #                 if temp_chess.chess_type == cc.jiang:  # 如果能吃了对方的将，那么就赢了
        #                     if temp_chess.belong != who:
        #                         # print(self.board.board[item.from_x][item.from_y])
        #                         # print(temp_chess)
        #                         # print(item)
        #                         # print('bug here')
        #                         return cc.max_val
        #                     else:
        #                         relation_val[1 - now] -= 20  # 如果不能，那么就相当于被将军，对方要减分
        #                         continue
        #                 # 记录攻击了谁
        #                 relation_list[x][y].attack[relation_list[x][y].num_attack] = temp_chess.chess_type
        #                 relation_list[x][y].num_attack += 1
        #                 relation_list[item.to_x][item.to_y].chess_type = temp_chess.chess_type
        #                 # print(item)
        #                 # 记录被谁攻击
        #                 # if item.to_x == 4 and item.to_y == 1:
        #                 #     print('--------------')
        #                 #     print(now_chess.chess_type)
        #                 #     print(item.from_x, item.from_y)
        #                 #     print('*************')
        #                 #     print(temp_chess.chess_type)
        #                 #     print(item.to_x, item.to_y)
        #                 #     print(relation_list[item.to_x][item.to_y].num_attacked)
        #                 #     print([relation_list[item.to_x][item.to_y].attacked[j] for j in range(relation_list[item.to_x][item.to_y].num_attacked)])
        #                 #     if relation_list[item.to_x][item.to_y].num_attacked == 5:
        #                 #         print('###################')
        #                 #         self.board.print_board()
        #                 #         print('###################')
        #
        #                 relation_list[item.to_x][item.to_y].attacked[
        #                     relation_list[item.to_x][item.to_y].num_attacked] = type
        #                 relation_list[item.to_x][item.to_y].num_attacked += 1
        #             elif temp_chess.belong == now:
        #                 # print('ok2')
        #                 if temp_chess.chess_type == cc.jiang:  # 保护自己的将没有意义，直接跳过
        #                     continue
        #                 # 记录关系信息-guard
        #                 # print(item)
        #                 # if item.to_x == 4 and item.to_y == 1:
        #                 #     print('--------------')
        #                 #     print(now_chess.chess_type)
        #                 #     print(item)
        #                 #     print('*************')
        #                 #     print(temp_chess.chess_type)
        #                 #     print(relation_list[item.to_x][item.to_y].num_guarded)
        #                 #     print([relation_list[item.to_x][item.to_y].guarded[j] for j in range(relation_list[item.to_x][item.to_y].num_guarded)])
        #                 #     if relation_list[item.to_x][item.to_y].num_guarded == 5:
        #                 #         print('###################')
        #                 #         print(x, y, who)
        #                 #         self.board.print_board(True)
        #                 #         print('###################')
        #                 relation_list[x][y].guard[relation_list[x][y].num_guard] = temp_chess
        #                 relation_list[x][y].num_guard += 1
        #                 relation_list[item.to_x][item.to_y].chess_type = temp_chess.chess_type
        #                 relation_list[item.to_x][item.to_y].guarded[relation_list[item.to_x][item.to_y].num_guarded] = type
        #                 relation_list[item.to_x][item.to_y].num_guarded += 1
        #             # relation_list[x][y].chess_type = type
        # for x in range(10):
        #     for y in range(9):
        #         num_attacked = relation_list[x][y].num_attacked
        #         num_guarded = relation_list[x][y].num_guarded
        #         now_chess = board[x][y]
        #         type = abs(now_chess)
        #         if self.side == "red":
        #             if now_chess > 0:
        #                 now = 0
        #             else:
        #                 now = 1
        #         elif self.side == "black":
        #             if now_chess < 0:
        #                 now = 0
        #             else:
        #                 now = 1
        #         unit_val = cc.base_val[type] >> 3
        #         sum_attack = 0  # 被攻击总子力
        #         sum_guard = 0
        #         min_attack = 999  # 最小的攻击者
        #         max_attack = 0  # 最大的攻击者
        #         max_guard = 0
        #         flag = 999  # 有没有比这个子的子力小的
        #         if type == 0:
        #             continue
        #         # 统计攻击方的子力
        #         for i in range(num_attacked):
        #             temp = cc.base_val[relation_list[x][y].attacked[i]]
        #             flag = min(flag, min(temp, cc.base_val[type]))
        #             min_attack = min(min_attack, temp)
        #             max_attack = max(max_attack, temp)
        #             sum_attack += temp
        #         # 统计防守方的子力
        #         for i in range(num_guarded):
        #             temp = cc.base_val[relation_list[x][y].guarded[i]]
        #             max_guard = max(max_guard, temp)
        #             sum_guard += temp
        #         if num_attacked == 0:
        #             relation_val[now] += 5 * relation_list[x][y].num_guarded
        #         else:
        #             muti_val = 5 if who != now else 1
        #             if num_guarded == 0:  # 如果没有保护
        #                 relation_val[now] -= muti_val * unit_val
        #             else:  # 如果有保护
        #                 if flag != 999:  # 存在攻击者子力小于被攻击者子力,对方将愿意换子
        #                     relation_val[now] -= muti_val * unit_val
        #                     relation_val[1 - now] -= muti_val * (flag >> 3)
        #                 # 如果是二换一, 并且最小子力小于被攻击者子力与保护者子力之和, 则对方可能以一子换两子
        #                 elif num_guarded == 1 and num_attacked > 1 and min_attack < cc.base_val[type] + sum_guard:
        #                     relation_val[now] -= muti_val * unit_val
        #                     relation_val[now] -= muti_val * (sum_guard >> 3)
        #                     relation_val[1 - now] -= muti_val * (flag >> 3)
        #                 # 如果是三换二并且攻击者子力较小的二者之和小于被攻击者子力与保护者子力之和,则对方可能以两子换三子
        #                 elif num_guarded == 2 and num_attacked == 3 and sum_attack - max_attack < cc.base_val[type] + sum_guard:
        #                     relation_val[now] -= muti_val * unit_val
        #                     relation_val[now] -= muti_val * (sum_guard >> 3)
        #                     relation_val[1 - now] -= muti_val * ((sum_attack - max_attack) >> 3)
        #                 # 如果是n换n，攻击方与保护方数量相同并且攻击者子力小于被攻击者子力与保护者子力之和再减去保护者中最大子力,则对方可能以n子换n子
        #                 elif num_guarded == num_attacked and sum_attack < cc.base_val[now_chess.chess_type] + sum_guard - max_guard:
        #                     relation_val[now] -= muti_val * unit_val
        #                     relation_val[now] -= muti_val * ((sum_guard - max_guard) >> 3)
        #                     relation_val[1 - now] -= sum_attack >> 3
        # # print('-------------------------')
        # print(base_val[0], pos_val[0], mobile_val[0], relation_val[0])
        # print(base_val[1], pos_val[1], mobile_val[1], relation_val[1])
        my_max_val = base_val[0] + pos_val[0] + mobile_val[0] + relation_val[0]
        my_min_val = base_val[1] + pos_val[1] + mobile_val[1] + relation_val[1]
        if who == 0:
            return my_max_val - my_min_val
        else:
            return my_min_val - my_max_val

    def init_relation_list(self):
        res_list = []
        for i in range(10):
            res_list.append([])
            for j in range(9):
                res_list[i].append(mr.relation())
        return res_list