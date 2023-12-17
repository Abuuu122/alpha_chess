import random
import numpy as np
from utils import get_legal_actions, change_round
import copy
import net

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
        
        action_list = get_legal_actions(board, self.side, self.history) # get all actions that are legal to choose from
        mcts = MCTSPlayer(net.PolicyValueNet.policy_value_fn)
        state = Board(board,action_list, self.history, self.side)
        action = mcts.get_action(state)
        return action   

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
    
string2array = dict({ 6 : np.array([1, 0, 0, 0, 0, 0, 0]), 5:np.array([0, 1, 0, 0, 0, 0, 0]),
                    4:np.array([0, 0, 1, 0, 0, 0, 0]), 3:np.array([0, 0, 0, 1, 0, 0, 0]),
                    2:np.array([0, 0, 0, 0, 1, 0, 0]), 1:np.array([0, 0, 0, 0, 0, 1, 0]),
                    7:np.array([0, 0, 0, 0, 0, 0, 1]), -6:np.array([-1, 0, 0, 0, 0, 0, 0]),
                    -5:np.array([0, -1, 0, 0, 0, 0, 0]), -4:np.array([0, 0, -1, 0, 0, 0, 0]),
                    -3:np.array([0, 0, 0, -1, 0, 0, 0]), -2:np.array([0, 0, 0, 0, -1, 0, 0]),
                    -1:np.array([0, 0, 0, 0, 0, -1, 0]), -7:np.array([0, 0, 0, 0, 0, 0, -1]),
                    0:np.array([0, 0, 0, 0, 0, 0, 0])
})

def state_list2state_array(state_list):
    _state_array = np.zeros([10, 9, 7])
    for i in range(10):
        for j in range(9):
            _state_array[i][j] = string2array[state_list[i][j]]
    return _state_array

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class Board():
    def __init__(self, board, action_list, history, side):
        self.board = board
        self.action_list = action_list
        self.history = history
        self.side = side
        self.step = 0
    def current_state(self):
        _current_state = np.zeros([9, 10, 9])
        # 使用9个平面来表示棋盘状态
        # 0-6个平面表示棋子位置，1代表红方棋子，-1代表黑方棋子, 队列最后一个盘面
        # 第7个平面表示对手player最近一步的落子位置，走子之前的位置为-1，走子之后的位置为1，其余全部是0
        # 第8个平面表示的是当前player是不是先手player，如果是先手player则整个平面全部为1，否则全部为0
        _current_state[:7] = state_list2state_array(self.board).transpose([2, 0, 1])  # [7, 10, 9]

        if len(self.history):
            # 解构self.last_move
            action = self.history[-1]
            old_x, old_y, x, y = action
            _current_state[7][old_x][old_y] = -1
            _current_state[7][x][y] = 1
            
        # 指出当前是哪个玩家走子
        if self.side == 'red':
            _current_state[8][:, :] = 1.0

        return _current_state
    
    
    def get_current_player_id(self):
        return self.side
    
    def game_end(self):
        action_list = get_legal_actions(self.board, self.side, self.history)
        
        if len(action_list) == 0:
            if self.side == "red":
                winner = "black"
            elif self.side == "black":
                winner = "red"
            return True, winner
        if self.step == 120:
            return True, 0
        
        return False , 0
    
    def move(self, action):
        old_x, old_y, x, y = action
        copy_board = copy.deepcopy(self.board)
        action_list = get_legal_actions(copy_board, self.side, self.history)

        # Game Over
        if len(action_list) == 0:
            if round == "red":
                text = "Red loses the game. Black wins!"
                winner = "black"
            elif round == "black":
                text = "Black loses the game. Red wins!"
                winner = "red"

        # Get action
        #if round == "red":
        #    red.update_history(copy.deepcopy(history))
        #    action = get_player_action_with_timeout(copy_board, red) if timeout else red.policy(board)
        #elif round == "black":
        #    black.update_history(copy.deepcopy(history))
        #    action = get_player_action_with_timeout(copy_board, black) if timeout else black.policy(board)

        # Check action
        #if action not in self.action_list:
         #   if round == "red":
          #      text = "Red timeout, Black wins!" if action == "Timed out" else "Red moves illegally, Black wins!"
           #     winner = "black"
                
            
        #    elif round == "black":
        #        text = "Black timeout, Red wins!" if action == "Timed out" else "Black moves illegally, Red wins!"
        #        winner = "red"
                

        # Record game state and change game state
        action_history = (action[0], action[1], action[2], action[3], self.board[action[2]][action[3]])
        self.history.append(action_history)

        # Take action
        self.board[action[2]][action[3]] = self.board[action[0]][action[1]]
        self.board[action[0]][action[1]] = 0

        if action_history[4] != 0: # Refresh the record when some piece is eaten
            self.step = 0
        else:
            self.step += 1
            if self.step == 120:    # Draw
                text = "Both sides have not eaten in sixty rounds, draw!"
                winner = "draw"
                
        self.side = change_round(self.side)
        

# 定义叶子节点
class TreeNode(object):
    """
    mcts树中的节点,树的子节点字典中,键为动作,值为TreeNode。记录当前节点选择的动作,以及选择该动作后会跳转到的下一个子节点。
    每个节点跟踪其自身的Q,先验概率P及其访问次数调整的u
    """

    def __init__(self, parent, prior_p):
        """
        :param parent: 当前节点的父节点
        :param prior_p:  当前节点被选择的先验概率
        """
        self._parent = parent
        self._children = {} # 从动作到TreeNode的映射
        self._n_visits = 0  # 当前当前节点的访问次数
        self._Q = 0         # 当前节点对应动作的平均动作价值
        self._u = 0         # 当前节点的置信上限         # PUCT算法
        self._P = prior_p

    def expand(self, action_priors):    # 这里把不合法的动作概率全部设置为0
        """通过创建新子节点来展开树"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] =  TreeNode(self, prob)

    def select(self, c_puct):
        """
        在子节点中选择能够提供最大的Q+U的节点
        return: (action, next_node)的二元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """
        计算并返回此节点的值, 它是节点评估Q和此节点的先验的组合
        c_puct: 控制相对影响(0, inf)
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """
        从叶节点评估中更新节点值
        leaf_value: 这个子节点的评估值来自当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 使用递归的方法对所有节点（当前节点对应的支线）进行一次更新
    def update_recursive(self, leaf_value):
        """就像调用update()一样，但是对所有直系节点进行更新"""
        # 如果它不是根节点，则应首先更新此节点的父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """检查是否是叶节点，即没有被扩展的节点"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """policy_value_fn: 接收board的盘面状态,返回落子概率和盘面评估得分"""
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        进行一次搜索，根据叶节点的评估值进行反向更新树节点的参数
        注意:state已就地修改,因此必须提供副本
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct)
            state.move(action)   #模拟向前走棋

        # 使用网络评估叶子节点，网络输出（动作，概率）元组p的列表以及当前玩家视角的得分[-1, 1]
        action_probs, leaf_value = self._policy(state)    #_policy is policy_fn
        # 查看游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 对于结束状态，将叶子节点的值换成1或-1
            if winner == 0:    # Tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player_id() else -1.0
                )
        # 在本次遍历中更新节点的值和访问次数
        # 必须添加符号，因为两个玩家共用一个搜索树
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        按顺序运行所有搜索并返回可用的动作及其相应的概率
        state:当前游戏的状态
        temp:介于(0, 1]之间的温度参数
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 跟据根节点处的访问计数来计算移动概率
        act_visits = [(act, node._n_visits)
                     for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        在当前的树上向前一步，保持我们已经直到的关于子树的一切
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


# 基于MCTS的AI玩家
class MCTSPlayer(object):

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    # 重置搜索树
    def reset_player(self):
        self.mcts.update_with_move(-1)

    # 得到行动
    def get_action(self, board, temp=1e-3, return_prob=0):
        # 像alphaGo_Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(2086)

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # 添加Dirichlet Noise进行探索（自我对弈需要）
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(0.2 * np.ones(len(probs)))
            )
            # 更新根节点并重用搜索树
            self.mcts.update_with_move(move)
        else:
            # 使用默认的temp=1e-3，它几乎相当于选择具有最高概率的移动
            move = np.random.choice(acts, p=probs)
            # 重置根节点
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move
