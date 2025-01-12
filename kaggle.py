import numpy as np
import torch
import kaggle_environments
from utils import np, torch
from games import Connect4
from models import ResNet
from mcts import AlphaMCTS


class KaggleAgent:
  """
  Implements an agent that can play in the kaggle environment.
  """
  def __init__(self, model: ResNet, game: Connect4, args: dict):
    """
    Initializes the KaggleAgent
    Args:
      model: ResNet model to use.
      game: Game to use
      args: Dictionary of hyperparameters
    """
    self.model = model
    self.game = game
    self.args = args
    if self.args['search']:
      self.mcts = AlphaMCTS(game, args, model)

  def run(self, obs, conf) -> int:
    """
    Runs the agent with the environment data.

    Args:
        obs: observation from the environment
        conf: configurations from the environment

    Returns:
        The action to take
    """
    player = obs['mark'] if obs['mark'] == 1 else -1
    state = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
    state[state==2] = -1
    state = self.game.change_perspective(state, player)

    if self.args['search']:
      policy = self.mcts.search(state)
    else:
      policy, _ = self.model(
        torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
      )

    valid_moves = self.game.get_valid_moves(state)
    policy *= valid_moves
    policy /= np.sum(policy)

    if self.args['temperature'] == 0:
      action = int(np.argmax(policy))
    elif self.args['temperature'] == float('inf'):
      action = np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
    else:
      policy = policy ** (1 / self.args['temperature'])
      policy /= np.sum(policy)
      action = np.random.choice(self.game.action_size, p=policy)
    return action

if __name__ == '__main__':
    game = Connect4()
    args = {
        'C': 2,
        'num_searches': 500,
        'dirichlet_epsilon': 0.1,
        'dirichlet_alpha': 0.3,
        'search': True,
        'temperature': 0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()
    env = kaggle_environments.make("connectx")
    player1 = KaggleAgent(model, game, args)
    player2 = KaggleAgent(model, game, args)
    players = [player1.run, player2.run]
    env.run(players)
    env.render(mode="ipython")
