"""
@author: vxfla
"""
from __future__ import print_function
import numpy as np

class Board(object):
    """board for the game"""

    def __init__(self):
        self.width = 8
        self.height = 8
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {27: 1, 28: 2, 35: 2, 36: 1}
        self.last_move = -1
        # init the initial 4 pieces
        self.availables.remove(27)
        self.availables.remove(28)
        self.availables.remove(35)
        self.availables.remove(36)
        self.avail_p1 = self.can_move(1)
        self.avail_p2 = self.can_move(2)

    def can_move(self, current_player):
        """flush avail_p1 & avail_p2"""
        avail = []
        for move in self.availables:
            location = self.move_to_location(move=move)
            row = location[0]
            col = location[1]
            direction = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
            flag = True
            for i in range(8):
                if not flag:
                    continue
                z = 1
                while (0 <= row + direction[i][0] * (z + 1) < 8) and (0 <= col + direction[i][1] * (z + 1) < 8) and \
                        (self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])
                         not in self.availables) \
                        and (self.states[self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])]
                             != current_player):
                    z = z + 1
                    if (self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])
                        not in self.availables) and \
                            (self.states[self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])]
                             == current_player):
                        avail.append(move)
                        flag = False
                        break
        return avail

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # 存储过去走过的自己和对手的棋，展示在棋盘上
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        if move not in self.get_avail(self.current_player):
            print('illegal move')
            exit(0)
        self.states[move] = self.current_player
        self.availables.remove(move)
        location = self.move_to_location(move=move)
        row = location[0]
        col = location[1]
        direction = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        for i in range(8):
            z = 1
            while (0 <= row + direction[i][0] * (z + 1) < 8) and (0 <= col + direction[i][1] * (z + 1) < 8) and \
                        (self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])
                         not in self.availables) \
                        and (self.states[self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])]
                             != self.current_player):
                z = z + 1
                if (self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])
                        not in self.availables) and \
                            (self.states[self.location_to_move([row + direction[i][0] * z, col + direction[i][1] * z])]
                             == self.current_player):
                    for j in range(z):
                        self.states[self.location_to_move([row + j * direction[i][0], col + j * direction[i][1]])] \
                            = self.current_player
                    break
        self.avail_p1 = self.can_move(1)
        self.avail_p2 = self.can_move(2)
        self.last_move = move
        if (self.avail_p1 == [] and self.avail_p2 == []) or (self.availables == []):
            return self.game_end()
        if self.current_player == 1:
            if self.avail_p2:
                self.current_player = 2
        else:
            if self.avail_p1:
                self.current_player = 1
        return False, -1

    def game_end(self):
        black = 0
        white = 0
        for _, value in self.states.items():
            if value == 1:
                black += 1
            else:
                white += 1
        if black > white:
            return True, 1
        elif black == white:
            return True, -1
        else:
            return True, 2

    def get_current_player(self):
        return self.current_player

    def get_avail(self, player):
        if player == self.players[0]:
            return self.avail_p1
        else:
            return self.avail_p2

class Game(object):
    """game server"""
    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height
        avail = board.get_avail(board.get_current_player())
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:5}".format(x), end='')
        print('\r')
        print('_|'+'____|'*width)
        for i in range(height - 1, -1, -1):
            print("{0:1d}".format(i), end='|')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(4), end='|')
                elif p == player2:
                    print("\033[1;31m O  \033[0m".center(4), end='|')
                elif loc in avail:
                    print("\033[1;42m    \033[0m".center(4), end='|')
                else:
                    print(' '.center(4), end='|')
            print('\r')
            print('_|'+'____|'*width)

    def start_play(self, player1, player2, is_shown=1):
        """start a game between two players"""
        start_player = 0
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            end, winner = self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            end, winner = self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)