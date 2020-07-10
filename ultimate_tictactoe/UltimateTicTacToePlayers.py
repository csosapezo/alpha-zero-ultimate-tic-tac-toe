from random import random

import numpy as np


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

    def __init__(self, parent=None, state=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None


class MonteCarloTreeSearchPlayer:
    def __init__(self, n=25):
        self.n = n

    def play(self, state, game):
        def select(n):
            """select a leaf node in the tree"""
            if n.children:
                return select(max(n.children.keys(), key=ucb))
            else:
                return n

        def expand(n):
            """expand the leaf node by adding all its children states"""
            if not n.children and not game.getGameEnded(game, 1):
                n.children = {MCT_Node(state=game.getNextState(n.state, action)[0], parent=n): action
                              for action in game.getValidMoves(n.state, 1)}
            return select(n)

        def simulate(game, state):
            """simulate the utility of current state by random picking a step"""
            player = game.to_move(state)
            while not game.terminal_test(state):
                action = random.choice(list(game.actions(state)))
                state = game.result(state, action)
            v = game.utility(state, player)
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
            result = simulate(game, child.state)
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
