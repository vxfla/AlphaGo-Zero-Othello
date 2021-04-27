# -*- coding: utf-8 -*-
"""
Start games
If human
Input your move in the format: 2,3

@author: vxfla
"""

from __future__ import print_function
from game import Board, Game
import numpy as np
from mcts_pure import MCTSPlayer
from mcts_alphaZero import MCTSPlayer as MCTS_alphaZero_player
from policy_value_net_pytorch_BN import PolicyValueNet


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("player {} is playing... your input is:".format(self.player))
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.get_avail(self.player):
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

class Random_player(object):
    """
    player who chooses where to put piece randomly
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        avail = board.get_avail(self.player)
        random = np.random.randint(0,len(avail))
        return avail[random]

    def __str__(self):
        return "random player {}".format(self.player)

def run(is_shown=1):
    try:
        board = Board()
        game = Game(board)

        policy_net1 = PolicyValueNet(board_width=8, board_height=8, model_file='BN_policy_900_old.model')
        policy_net2 = PolicyValueNet(board_width=8, board_height=8, model_file='best_policy_old.model')
        player2 = MCTS_alphaZero_player(policy_value_function=policy_net2.policy_value_fn, c_puct=5, n_playout=500)
        player1 = MCTS_alphaZero_player(policy_value_function=policy_net1.policy_value_fn, c_puct=5, n_playout=500)

        game.start_play(player1=player2, player2=player1, is_shown=is_shown)
    except KeyboardInterrupt:
        print('\n\rquit')

def eva():
    for i in range(10):
        run(is_shown=0)
if __name__ == '__main__':
    eva()