import numpy as np

import Arena
from MCTS import MCTS
from othello.OthelloPlayers import HumanOthelloPlayer
from ultimate_tictactoe.UltimateTicTacToeGame import UltimateTicTacToeGame
from ultimate_tictactoe.UltimateTicTacToePlayers import RLUTTTPlayer
from ultimate_tictactoe.UltimateTicTacToePlayers import RandomUltimateTictacToePlayer, MonteCarloTreeSearchPlayer
from ultimate_tictactoe.keras.NNet import NNetWrapper as NNet
from ultimate_tictactoe.learning import NNUltimateLearning
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
# Player 2
random_vs_cpu = False
human_vs_cpu = False
rl_vs_az = True
mc_vs_az = False

# Number of games to play
num_games = 20

# model paths
rl_filepath = './pretrained_models/ultimate_tictactoe/ultimate_player_nn1.h5'
nn_filepath_dir = './pretrained_models/ultimate_tictactoe/keras/'
nn_filepath_p1 = 'ultimate_tictactoe_100_eps_10_epoch_checkpoint_5.h5'
nn_filepath_p2 = 'ultimate_tictactoe_100_eps_10_epoch_checkpoint_3.h5'


g = UltimateTicTacToeGame()

# all players
rp = RandomUltimateTictacToePlayer(g).play
hp = HumanOthelloPlayer(g).play
rlp = RLUTTTPlayer(NNUltimateLearning(filename=rl_filepath)).makeNextMove
mcts = MonteCarloTreeSearchPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint(nn_filepath_dir,nn_filepath_p1)
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
elif random_vs_cpu:
    player2 = rp
elif rl_vs_az:
    player2 = rlp
elif mc_vs_az:
    player2 = mcts
else:
    n2 = NNet(g)
    n2.load_checkpoint(nn_filepath_p1, nn_filepath_p2)
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(player1=n1p, player2=player2, game=g, display=UltimateTicTacToeGame.display)

print(arena.playGames(num_games, verbose=False))
