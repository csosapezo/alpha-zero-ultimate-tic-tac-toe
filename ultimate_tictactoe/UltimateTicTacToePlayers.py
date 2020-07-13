import random

import numpy as np

from ultimate_tictactoe.UltimateTicTacToeGame import UltimateTicTacToeGame
from ultimate_tictactoe.UltimateTicTacToeLogic import Board


class RandomUltimateTictacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


def ucb(n, C=1.4):
    if n.N == 0:
        return np.inf
    else:
        return (n.U / n.N) + C * np.sqrt(np.log(n.parent.N) / n.N)


class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, parent=None, state=None, U=0, N=0, player=1):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None
        self.player = player


class MonteCarloTreeSearchPlayer:
    def __init__(self, game, n=50):
        self.n = n
        self.player = 1
        self.game = game

    def play(self, state):
        def select(n):
            """select a leaf node in the tree"""
            if n.children:
                return select(max(n.children.keys(), key=ucb))
            else:
                return n

        def expand(n):
            """expand the leaf node by adding all its children states"""
            if not n.children and not self.game.getGameEnded(state, 1):
                n.children = {
                    MCT_Node(state=self.game.getNextState(n.state, n.player, self.game.N * x + y)[0], parent=n,
                             player=self.game.getNextState(n.state, n.player, self.game.N * x + y)[1]):
                        self.game.N * x + y for x, y in n.state.get_legal_moves()}
            return select(n)

        def simulate(state, player):
            """simulate the utility of current state by random picking a step"""
            i = 1
            while not self.game.getGameEnded(state, player):
                action = np.random.randint(self.game.getActionSize())
                valids = self.game.getValidMoves(state, player)
                while valids[action] != 1:
                    action = np.random.randint(self.game.getActionSize())
                state, player = self.game.getNextState(state, player, action)
                i = i + 1
            v = self.game.getGameEnded(state, player)
            return -v

        def backprop(n, utility):
            """passing the utility back to all parent nodes"""
            if utility > 0:
                n.U += utility
            # if utility == 0:
            #     n.U += 0.5
            n.N += 1
            if n.parent:
                backprop(n.parent, -utility)

        root = MCT_Node(state=state)

        for _ in range(self.n):
            leaf = select(root)
            child = expand(leaf)
            result = simulate(child.state, child.player)
            backprop(child, result)

        max_state = max(root.children, key=lambda p: p.N)

        return root.children.get(max_state)


class HumanUltimateTicTacToePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.N), int(i % self.game.N))
        while True:
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x, y = [int(x) for x in a.split(' ')]
            a = self.game.N * x + y if x != -1 else self.game.N ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
