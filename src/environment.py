from copy import deepcopy as dcopy
from math import sqrt, acos, pi
from GameBoard.game_board import Screen
from src.utils import flatten
import numpy as np
import time

ScoreBoardID = 0
AgentID = 1
ConquerID = 2
TreasureID = 3
WallID = 4
dx = [1, -1, 0, 0]
dy = [0, 0, -1, 1]

class Player(object):
    
    def __init__(self, ID):
        self.ID = ID
        self.title_score = 0
        self.area_score = 0
        self.treasure_score = 0
        self.old_score = 0
        
    @property
    def total_score(self):
        """
        Returns the total scores consits of title, area and treasure scores
        """
        return self.title_score + self.area_score + self.treasure_score
    
    def reset(self):
        self.title_score = 0
        self.area_score = 0
        self.treasure_score = 0
        self.old_score = 0
        
    def show_scores(self):
        print("Player " + str(self.ID) + ":")
        print("\tTitle Score: {}".format(self.title_score))
        print("\tTreasure Score: {}".format(self.treasure_score))
        print("\tArea Score {}".format(self.area_score))
        print()
        
class Environment(object):

    def __init__(self, input_data = None, show_screen = False, MAX_SIZE = 20):
        self.MAX_SIZE = MAX_SIZE
        self.show_screen = show_screen
        self.data = dcopy(input_data)
        self.n_actions = 9
        self.punish = 0
        self.n_inputs = 9
        self.num_players = 2
        self.action_dim = self.n_actions
        self.players = [Player(i) for i in range(self.num_players)]
        self.screen = Screen(self)
        self.reset()
    
    
    def reset(self):
        """
        height: height of table
        width: width of table
        score_board: title score in table
        agent_pos: location of agents in table (coord)
        treasure_board: treasures in table
        wall_board: walls in table
        conquer_board: conquered locations of players
        n_turns: number of turns in each game
        n_agents: number of agents

        """
        height, width, score_board, agent_pos, treasure_board, wall_board, \
            conquer_board, n_turns, n_agents = [dcopy(_data) for _data in self.data]
            
        self.score_board = []
        self.agent_board = [[], []]
        self.treasure_board = []
        self.wall_board = []
        self.conquer_board = [[], []]
        self.width = width
        self.height = height
        self.agent_pos = agent_pos
        self.n_turns = n_turns
        self.remaining_turns = n_turns
        self.n_agents = n_agents
        
        for player_ID in range(self.num_players):
            self.players[player_ID].reset()
    
        for _ in range(self.MAX_SIZE):
            self.score_board.append([0] * self.MAX_SIZE)
            self.treasure_board.append([0] * self.MAX_SIZE)
            self.wall_board.append([0] * self.MAX_SIZE)
            for player_ID in range(self.num_players):
                self.agent_board[player_ID].append([0] * self.MAX_SIZE)
                self.conquer_board[player_ID].append([0] * self.MAX_SIZE)

        for i in range(self.height):
            for j in range(self.width):
                self.score_board[i][j] = score_board[i][j]
        
        
        for i in range(self.n_agents):    
            for j in range(self.num_players):
                x, y = self.agent_pos[j][i]
                self.agent_board[j][x][y] = 1
                self.conquer_board[j][x][y] = 1
            
        for x, y in wall_board:
            self.wall_board[x][y] = 1
        
        for i in range(self.MAX_SIZE):
            for j in range(self.MAX_SIZE):
                if i >= self.height or j >= self.width:
                    self.wall_board[i][j] = 1
            
        for x, y, value in treasure_board:
            self.treasure_board[x][y] = value
            
        self.upper_bound_score = np.max(score_board)
        self.lower_bound_score = np.min(score_board)
        self.range_bound = (self.upper_bound_score - self.lower_bound_score)
        # self.score_board = (self.score_board - self.lower_bound_score) \
        #     / self.range_bound
        # self.treasure_board /= self.range_bound
        
        self.observation = self.get_observation(0)
            
        title_scores, treasure_scores, area_scores = \
            self.compute_score(self.observation, self.observation)
        
        for player_ID in range(self.num_players):
            self.players[player_ID].title_score = title_scores[player_ID]
            self.players[player_ID].treasure_score = treasure_scores[player_ID]
            self.players[player_ID].area_score = area_scores[player_ID]
            self.players[player_ID].old_score = self.players[player_ID].total_score
        
        self.old_observation = dcopy(self.observation)
        
        if self.show_screen:
            self.screen.setup(self)
    
    def soft_reset(self):
        
        for player_ID in range(self.num_players):
            self.players[player_ID].reset()
            for x in range(self.height):
                for y in range(self.width):       
                    self.agent_board[player_ID][x][y] = 0
                    self.conquer_board[player_ID][x][y] = 0
        
        for i in range(self.n_agents):    
            for j in range(2):
                x, y = self.agent_pos[j][i]
                if self.show_screen:
                    self.screen.reset_square([x, y], -1, 0)
                    
        height, width, _, agent_pos,  _, _, \
            conquer_board, n_turns, n_agents = [dcopy(_data) for _data in self.data]
            
        self.agent_pos = agent_pos
        self.remaining_turns = self.n_turns
        for player_ID in range(self.num_players):
            for agent_ID in range(self.n_agents):
                x, y = self.agent_pos[player_ID][agent_ID]
                self.agent_board[player_ID][x][y] = 1
                self.conquer_board[player_ID][x][y] = 1
    
        self.observation = self.get_observation(0)
        title_scores, treasure_scores, area_scores = \
            self.compute_score(self.observation, self.observation)
        
        for player_ID in range(self.num_players):
            self.players[player_ID].title_score = title_scores[player_ID]
            self.players[player_ID].treasure_score = treasure_scores[player_ID]
            self.players[player_ID].area_score = area_scores[player_ID]
            self.players[player_ID].old_score = self.players[player_ID].total_score
            
        self.old_observation = dcopy(self.observation)
        
        if self.show_screen:
            self.screen.reset()
        
        
    def render(self):
        """
        display game screen
        """
        self.screen.render()
        
    def get_ub_board_size(self):
        """
        Returns upper bound of board size
        """
        return [self.MAX_SIZE, self.MAX_SIZE]
    
    def get_state(self, player, agent_id):
        state = self.get_observation(player)
        state.append(self.get_agent_state(agent_id, self.agent_pos[player]))
        return state
    
    def get_observation(self, player_ID):
        """
        Returns current observation
        """
        remaining_turns = []
        for _ in range(self.MAX_SIZE):
            remaining_turns.append([self.remaining_turns] * self.MAX_SIZE)
                
        
        state = dcopy([self.score_board, 
                       self.agent_board, 
                       self.conquer_board, 
                       self.treasure_board, 
                       self.wall_board,
                       remaining_turns])
        
        if player_ID == 1:
            temp = dcopy(state[1][0])
            state[1][0] = dcopy(state[1][1])
            state[1][1] = temp
            temp = dcopy(state[2][0])
            state[2][0] = dcopy(state[2][1])
            state[2][1] = temp
            
        return state
    
    def get_obs_for_states(self, states):
        states = np.array(flatten(states), dtype = np.float32)\
            .reshape(-1, self.n_inputs, self.MAX_SIZE, self.MAX_SIZE)
        return states
    
    
    def get_agent_pos(self, player):
        return dcopy(self.agent_pos[player])
    
    def get_agent_state(self, agent_id, agent_pos):
        agent_state = []
        for i in range(self.MAX_SIZE):
            agent_state.append([0] * self.MAX_SIZE)
        x, y = agent_pos[agent_id]
        agent_state[x][y] = 1
        return agent_state
    
    def get_act(act):
        switcher = {
                (0, 0):   0,
                (1, 0):   1,
                (1, 1):   2,
                (0, 1):   3,
                (-1, 1):  4,
                (-1, 0):  5,
                (-1, -1): 6,
                (0, -1):  7,
                (1, -1):  8,
            }
        return switcher.get(act, 0)
    
    def compute_score_area(self, state, player_ID):
        def is_border(x, y):
            return x <= 0 or x >= self.height - 1 or y <= 0 or y >= self.width - 1
        
        def can_move(x, y):
            return x >= 0 and x < self.height and y >= 0 and y < self.width
        
        def dfs(x, y, visited):
            visited[x][y] = True
            is_closed = True
            if is_border(x, y):
                is_closed = False
            temp_score = abs(score_board[x][y])
            if wall_board[x][y] == 1:
                temp_score = 0
            for i in range(4):
                _x = x + dx[i]
                _y = y + dy[i]
                if can_move(_x, _y) and not visited[_x][_y]:
                   _score = dfs(_x, _y, visited)
                   if _score < 0:
                       is_closed = False
                   else:
                       temp_score += _score
            if not is_closed:
                return -1
            return temp_score
        
        visited = []
        score_board = state[ScoreBoardID]
        conquer_board = state[ConquerID]
        wall_board = state[WallID]
        score = 0
        for i in range(self.height):
            visited.append([False] * self.width)
            for j in range(self.width):
                if conquer_board[player_ID][i][j] == 1:
                    visited[i][j] = True

        for i in range(self.height):
            for j in range(self.width):
                if not visited[i][j]:
                    temp = dfs(i, j, visited)
                    score += max(0, temp)
                    # if score > 0:
                    #     print(score)
                    #     print(visited)
        return score
        
    def compute_score(self, state, old_state):
        """
        
        Parameters
        ----------
        state : object
            state of game.
        old_state : object
            prestate of game.

        Returns
        -------
        title_scores : array
            title scores of players.
        treasure_score : TYPE
            treasure scores of players.
        area_scores : TYPE
            area scores of players.

        """
        score_board = state[ScoreBoardID]
        conquer_board = state[ConquerID]
        treasure_board = state[TreasureID]
        title_scores = [0, 0]
        treasure_score = [0, 0]
        area_scores = [0, 0]
        for i in range(self.height):
            for j in range(self.width):
                if conquer_board[0][i][j] == 1:
                    title_scores[0] += score_board[i][j]
                if conquer_board[1][i][j] == 1:
                    title_scores[1] += score_board[i][j]
                if treasure_board[i][j] > 0 and old_state[ConquerID][0][i][j] == 0 \
                        and old_state[ConquerID][1][i][j] == 0:
                    if conquer_board[0][i][j] == 1:
                        treasure_score[0] += treasure_board[i][j]
                    if conquer_board[1][i][j] == 1:
                        treasure_score[1] += treasure_board[i][j]
                        
        for player_ID in range(self.num_players):
            area_scores[player_ID] = self.compute_score_area(state, player_ID)
            
        return title_scores, treasure_score, area_scores
    
    def get_score(self, state, old_state, player_ID):
        state = dcopy(state)
        state.pop()
        title_scores, treasure_scores, area_scores = self.compute_score(state, old_state)
        result = title_scores[0] + treasure_scores[0] + area_scores[0] \
            - title_scores[1] - treasure_scores[1] - area_scores[1]
        return result
            
    def check_next_action(self, _act, id_agent, agent_pos):
        x, y = agent_pos[id_agent][0], agent_pos[id_agent][1]
        x, y = self.next_action(x, y, _act)
        if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
            return False
        
        return self.wall_board[x][y] == 0
    
    def next_action(self, x, y, act):
        def action(x):
            switcher = {
                0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
                4: [-1, 1], 5: [-1, 0], 6: [-1, -1], 7: [0, -1], 8: [1, -1]
            }
            return switcher.get(x, [0, 0])
        _action = action(act)
        return [x + _action[0], y + _action[1]]
    
    def angle(self, a1, b1, a2, b2):
        fi = acos((a1 * a2 + b1 * b2) / (sqrt(a1*a1 + b1*b1) * (sqrt(a2*a2 + b2*b2))))
        return fi
    
    def check(self, x0, y0, x, y, act):
        
        def action(x):
            switcher = {
                0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
                4: [-1, 1], 5: [-1, 0], 6: [-1, -1], 7: [0, -1], 8: [1, -1]
            }
            return switcher.get(x, [0, 0])
        
        a1, b1 = action(act)
        a2, b2 = x - x0, y - y0
        if abs(self.angle(a1, b1, a2, b2)) - 0.0001 <= pi / 3:
            return True
        return False
    
    def predict_spread_scores(self, x, y, state, act):
        score_board, agent_board, conquer_board, treasure_board, wall_board, _ = state
        score = 0
        discount = 0.02
        reduce_negative = 0.02
        p_1 = 1.3
        p_2 = 1
        aux_score = 0
        for i in range(1, min(8, self.remaining_turns)):
            for j in range(max(0, x - i), min(self.height, x + i + 1)):
                new_x = j
                new_y = y - i
                if new_y >= 0:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
                new_x = j
                new_y = y + i
                if new_y  < self.width:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
            for k in range(max(0, y - i), min(self.height, y + i + 1)):
                new_x = x - i
                new_y = k
                if new_x >= 0:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
                new_x = x + i
                new_y = k
                if new_x < self.height:
                    if wall_board[new_x][new_y] == 0: 
                        _sc = treasure_board[new_x][new_y] ** p_1
                        if conquer_board[0][new_x][new_y] != 1:
                            _sc += (max(reduce_negative * score_board[new_x][new_y], score_board[new_x][new_y]) ** p_2)
                        if act == 0 or self.check(x, y, new_x, new_y, act):
                            score += _sc * discount
            discount *= 0.7
        return score
    
    def soft_step(self, agent_id, state, act, agent_pos, predict = False):
        old_state = dcopy(state)
        old_scores, old_treasures_scores, area_scores = self.compute_score(old_state, old_state)
        old_score = old_scores[0] + area_scores[0] - old_scores[1] - area_scores[1]
        score_board, agent_board, conquer_board, treasure_board, wall_board, _ = state
        x, y = agent_pos[agent_id][0], agent_pos[agent_id][1]     
        new_pos = self.next_action(x, y, act)
        _x, _y = new_pos
        valid = True
        aux_score = 0
        reward = 0
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and wall_board[_x][_y] == 0:
            if agent_board[0][_x][_y] == 0 and agent_board[1][_x][_y] == 0:
                if conquer_board[1][_x][_y] == 0:
                    agent_board[0][_x][_y] = 1
                    agent_board[0][x][y] = 0
                    conquer_board[0][_x][_y] = 1
                    agent_pos[agent_id][0] = _x
                    agent_pos[agent_id][1] = _y
                else:
                    conquer_board[1][_x][_y] = 0
        else:
            valid = False
            
        title_scores, treasures_scores, area_scores = self.compute_score(state, old_state)
        if valid:
            if predict:
                aux_score = self.predict_spread_scores(_x, _y, state, act)
            else:
                aux_score = 0
            reward = title_scores[0] + treasures_scores[0] + area_scores [0] + aux_score\
                - title_scores[1] - treasures_scores[1] - area_scores[1] - old_score
        else:
            aux_score = - self.range_bound
            reward = aux_score
        return valid, state, reward
    
    def soft_step_(self, state, action):
        _, agent_board, conquer_board, treasure_board, wall_board, agent, _ = state
        x, y = -1, -1
        for i in range(self.height):
            for j in range(self.width):
                if agent[i][j] == 1:
                    x, y = i, j
        new_pos = (self.next_action(x, y, action))
        _x, _y = new_pos
        if _x >= 0 and _x < self.height and _y >= 0 and _y < self.width and wall_board[_x][_y] == 0:
            if agent_board[0][_x][_y] == 0 and agent_board[1][_x][_y] == 0:
                if conquer_board[1][_x][_y] == 0:
                    agent_board[0][_x][_y] = 1
                    agent_board[0][x][y] = 0
                    conquer_board[0][_x][_y] = 1
                    treasure_board[_x][_y] = 0
                    agent[x][y] = 0
                    agent[_x][_y] = 1
                else:
                    conquer_board[1][_x][_y] = 0
            
        return state        
    
    def get_next_action_pos(self, action_1, action_2):
        point_punish = 30
        punish = 0
        new_pos = [[], []]
        checked = [[False] * self.n_agents, [False] * self.n_agents]
        
        for i in range(self.n_agents):
            x, y = self.agent_pos[0][i][0], self.agent_pos[0][i][1]
            new_pos[0].append(self.next_action(x, y, action_1[i]))
            x, y = self.agent_pos[1][i][0], self.agent_pos[1][i][1]
            new_pos[1].append(self.next_action(x, y, action_2[i]))
        
        for i in range(self.n_agents):
            x, y = new_pos[0][i]
            if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
                checked[0][i] = True
                new_pos[0][i] = dcopy(self.agent_pos[0][i])
                punish += point_punish
            elif self.wall_board[x][y] == 1:
                checked[0][i] = True
                new_pos[0][i] = dcopy(self.agent_pos[0][i])
                punish += point_punish
            
        for i in range(self.n_agents):
            x, y = new_pos[1][i]
            if not (x >= 0 and x < self.height and y >= 0 and y < self.width):
                checked[1][i] = True
                new_pos[1][i] = dcopy(self.agent_pos[1][i])
            elif self.wall_board[x][y] == 1:
                checked[1][i] = True
                new_pos[1][i] = dcopy(self.agent_pos[1][i])
            
        # create connect matrix
        connected_matrix = []
        for j in range(2 * self.n_agents):
            connected_matrix.append([0] * (2 * self.n_agents))
            
        for i in range(2 * self.n_agents):
            X = new_pos[0][i] if i < self.n_agents else new_pos[1][i % self.n_agents]
            for j in range(2 * self.n_agents):
                if i == j:
                    continue
                Y = self.agent_pos[0][j] if j < self.n_agents \
                    else self.agent_pos[1][j % self.n_agents]
                if X[0] == Y[0] and X[1] == Y[1]:
                    connected_matrix[i][j] = 1
                        
        # if conflict action to 1 square
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                for k in range(2):
                    for l in range(k, 2):
                        if i == j and k == l: continue
                        if new_pos[k][i][0] == new_pos[l][j][0] and\
                            new_pos[k][i][1] == new_pos[l][j][1]:
                            checked[k][i] = True
                            checked[l][j] = True
                            if i != j:
                                punish += point_punish
        
        
        for i in range(self.n_agents):
            for j in range(2):
                if checked[j][i]:
                    new_pos[0][i] = dcopy(self.agent_pos[0][i])
                    
        # find the clique
        for i in range(2 * self.n_agents):
            if i < self.n_agents:
                if checked[0][i]:
                    continue
            elif checked[1][i - self.n_agents]:
                continue
            u = i
            stk = [u]
            visited = [False] * (2 * self.n_agents)
            visited[u] = True
            
            for _ in range(2 * self.n_agents):
                for j in range(2 * self.n_agents):
                    if connected_matrix[u][j] == 1 and u != j:
                        stk.append(j)
                        is_clique = False
                        if j < self.n_agents:
                            if checked[0][j]: is_clique = True
                        elif checked[1][j - self.n_agents]:
                            is_clique = True
                        if visited[j]:
                            is_clique = True
                            
                        if is_clique:
                            for id in stk:
                                if id < self.n_agents:
                                    checked[0][id] = 1
                                else:
                                    checked[1][id - self.n_agents] = 1
                            stk = []
                            break
                        u = j
                        visited[j] = True
        
        # find the remove action
        for i in range(2 * self.n_agents):
            u = i
            stk = []
            visited = [False] * (2 * self.n_agents)
            visited[u] = True
            if i < self.n_agents:
                if checked[0][i]:
                    continue
            elif checked[1][i - self.n_agents]:
                continue
            
            for _ in range(2 * self.n_agents):
                for j in range(2 * self.n_agents):
                    if connected_matrix[u][j] == 1 and u != j:
                        congestion = False
                        if j < self.n_agents:
                            x, y = new_pos[0][j]
                            if self.conquer_board[1][x][y] == 1:
                                congestion = True
                        else:
                            x, y = new_pos[1][j - self.n_agents]
                            if self.conquer_board[0][x][y] == 1:
                                congestion = True
                        if visited[j]:
                            congestion = True
                            
                        visited[j] = True
                        
                        if not congestion:
                            for id in stk:
                                if id < self.n_agents:
                                    checked[0][id] = 1
                                else:
                                    checked[1][id - self.n_agents] = 1
                            stk = []
                            break
                        stk.append(j)
                        u = j
                if len(stk) == 0:
                    break
                                
        for i in range(self.n_agents):
            for j in range(2):
                if checked[j][i]:
                    new_pos[j][i] = dcopy(self.agent_pos[j][i])
        
        
        return new_pos, checked, punish
    
    def step(self, action_1, action_2, render = False):
        new_pos, checked, punish = self.get_next_action_pos(action_1, action_2)
        
        # render before action
        for i in range(self.n_agents):
            if checked[0][i] == 0:
                x, y = new_pos[0][i]
                if self.conquer_board[1][x][y] == 0:
                    if self.agent_pos[0][i][0] != new_pos[0][i][0] \
                        or self.agent_pos[0][i][1] != new_pos[0][i][1]:
                        self.agent_board[0][self.agent_pos[0][i][0]][self.agent_pos[0][i][1]] = 0
                        self.agent_board[0][x][y] = 0
                        if render:
                            self.screen.redraw_squares(
                                self.agent_pos[0][i][0], self.agent_pos[0][i][1], 0)
                else:
                    self.agent_board[0][x][y] = self.agent_board[1][x][y] = 0
                                      
            if checked[1][i] == 0:
                x, y = new_pos[1][i]
                if self.conquer_board[0][x][y] == 0 :
                    if self.agent_pos[1][i][0] != new_pos[1][i][0] \
                        or self.agent_pos[1][i][1] != new_pos[1][i][1]:
                        self.agent_board[1][x][y] = 0
                        self.agent_board[1][self.agent_pos[1][i][0]][self.agent_pos[1][i][1]] = 0
                        if render:
                            self.screen.redraw_squares(
                                self.agent_pos[1][i][0], self.agent_pos[1][i][1], 1)
                else:
                    self.agent_board[0][x][y] = self.agent_board[1][x][y] = 0
                        
        # render after action
        for i in range(self.n_agents):
            for j in range(2):
                if checked[j][i] == 0:
                    x, y = new_pos[j][i]
                    if self.conquer_board[1 - j][x][y] == 1:
                        self.conquer_board[1 - j][x][y] = 0
                        if render:
                            self.screen.reset_square([x, y], -1)
                        new_pos[j][i] = dcopy(self.agent_pos[j][i])
                    else:
                        self.conquer_board[j][x][y] = 1
                        self.agent_board[j][x][y] = 1
        
                
        for i in range(self.n_agents):
            self.agent_pos[0][i] = [new_pos[0][i][0], new_pos[0][i][1]]
            self.agent_pos[1][i] = [new_pos[1][i][0], new_pos[1][i][1]]
            
        self.observation = self.get_observation(0)
        
        title_scores, treasure_scores, area_scores = \
            self.compute_score(self.observation, self.old_observation)
            
        if render: self.render()
        for player_ID in range(self.num_players):
            self.players[player_ID].title_score = title_scores[player_ID]
            self.players[player_ID].treasure_score += treasure_scores[player_ID]
            self.players[player_ID].area_score = area_scores[player_ID]
            
        self.old_observation = dcopy(self.observation)
        if render:
            for player_id in range(self.num_players):
                for agent_ID in range(self.n_agents):
                    coord = self.agent_pos[player_id][agent_ID]
                    self.screen.reset_square(coord, player_id, agent_ID)
            self.screen.show_score()
        
        if render: self.render()
        
        reward = self.players[0].total_score - self.players[1].total_score - \
            self.players[0].old_score + self.players[1].old_score
            
        for player_ID in range(self.num_players):
            self.players[player_ID].old_score = self.players[player_ID].total_score
            # self.players[player_ID].show_scores()
            
        # print(self.conquer_board)
        self.remaining_turns -= 1
        terminate = (self.remaining_turns == 0)
        self.punish += punish/1000
        # if not terminate:
        #     reward = 0
        # elif self.players[0].total_score < self.players[1].total_score:
        #     reward = -1
        # else:
        #     reward = 1
        time.sleep(5)
        return [self.observation, reward, terminate, self.remaining_turns]

    def next_state(self, state, action, player_ID, agent_ID):
        state = dcopy(state)
            
        if agent_ID == self.n_agents - 1:
            player_ID = 1 - player_ID
            agent_ID = 0
        else:
            agent_ID += 1
            
        if player_ID == 1:
            temp = dcopy(state[1][0])
            state[1][0] = dcopy(state[1][1])
            state[1][1] = temp
            temp = dcopy(state[2][0])
            state[2][0] = dcopy(state[2][1])
            state[2][1] = temp
            
        state[-1] = self.get_agent_state(agent_ID, self.agent_pos[player_ID])
        return self.soft_step_(state, action), player_ID, agent_ID
            
    def get_return(self, state, old_state, player_ID):
        return 1 if self.get_score(state, old_state, player_ID) >= 0 else -1
    
    def is_done_state(self, state, depth):
        return depth >= 2 * (1 + self.n_turns) * self.n_agents
    