import torch
from torch.optim import Adam
from utils import torch
from games import Connect4
from models import ResNet
from alphazero import AlphaZeroParallel

if __name__ == '__main__':
    Game = Connect4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(Game , 9,  128,device)
    optimizer = Adam(model.parameters(),lr = 0.001,weight_decay = 1e-4)

    args = {
            'num_iterations': 50,             # number of highest level iterations
            'num_selfPlay_iterations': 500,   # number of self-play games to play within each iteration
            'num_parallel_games': 100,        # number of games to play in parallel
            'num_mcts_searches': 500,         # number of mcts simulations when selecting a move within self-play
            'num_epochs': 2,                  # number of epochs for training on self-play data for each iteration
            'batch_size': 16,                # batch size for training
            'temperature': 1.25,                 # temperature for the softmax selection of moves
            'C': 2,                      # the value of the constant policy
            'augment': False,                 # whether to augment the training data with flipped states
            'dirichlet_alpha': 0.3,           # the value of the dirichlet noise
            'dirichlet_epsilon': 0.25,        # the value of the dirichlet noise
    }
    AlphaZero = AlphaZeroParallel(model,optimizer,Game,args)
    AlphaZero.learn()