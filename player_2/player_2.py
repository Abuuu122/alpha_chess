import random

from utils import get_legal_actions
# import xxx    # Here may be other package you want to import
from player_2 import history_table
from player_2 import constants as con
from player_2.step import Step



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
        self.name = "Player_2"    # please change to your group name
        self.max_depth = 4
        self.history_table = history_table.history_table()
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
        self.alpha_beta(board, self.max_depth, con.min_val, con.max_val)
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

    def alpha_beta(self, board, depth, alpha, beta):  # alpha-beta pruning
        who = (self.max_depth - depth) % 2  # who is the player
        if self.is_game_over(who, board):  # search end when the game is over
            return con.min_val
        if depth == 1:  # the leave node
            # print(self.evaluate(who))
            return self.evaluate(who, board)
        # get all the legal moves
        if self.side == "black":
            side = "red" if who else "black"
        elif self.side == "red":
            side = "black" if who else "red"
        _move_list = get_legal_actions(board, side, self.history)
        move_list = []
        for move in _move_list:
            move_list.append(Step(move[0], move[1], move[2], move[3]))
        # use the history table
        for i in range(len(move_list)):
            move_list[i].score = self.history_table.get_history_score(who, move_list[i])
        move_list.sort()
        # # for item in move_list:
        # #     print(item.score)
        # # print('----------------------')
        best_step = move_list[0] if len(move_list) > 0 else None

        score_list = []
        for step in move_list:
            # temp = self.move_to(step)
            self.move(board, step.from_x, step.from_y, step.to_x, step.to_y)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
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
        # update the history table
        if best_step is not None:
            self.history_table.add_history_score(who, best_step, depth)
        self.best_move = best_step
        return alpha


    def evaluate(self, who, board):  # return the value
        # print('====================================================================================')
        base_val = [0, 0]
        pos_val = [0, 0]
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
                # base value
                base_val[now] += con.base_val[type]
                # position value
                if self.side == "red":
                    if now == 0:  # max node
                        pos_val[now] += con.pos_val[type][pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {con.pos_val[type][pos]}")
                    elif now == 1:
                        pos_val[now] += con.pos_val[type][89 - pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {con.pos_val[type][89-pos]}")
                elif self.side == "black":
                    if now == 0:  # max node
                        pos_val[now] += con.pos_val[type][89 - pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {con.pos_val[type][89 - pos]}")
                    elif now == 1:
                        pos_val[now] += con.pos_val[type][pos]
                        # print(f"now == {now}: type = {type} pos = {pos} pos_val = {con.pos_val[type][pos]}")


        # # print('-------------------------')
        # print(base_val[0], pos_val[0])
        # print(base_val[1], pos_val[1])
        my_max_val = base_val[0] + pos_val[0]
        my_min_val = base_val[1] + pos_val[1]
        if who == 0:
            return my_max_val - my_min_val
        else:
            return my_min_val - my_max_val