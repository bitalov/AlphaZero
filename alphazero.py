import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from utils import trange, random, np, torch, F
from games import Connect4
from models import ResNet
from mcts import AlphaMCTS, Node


class AlphaZero:
  """
  Implements the AlphaZero algorithm for self-play and training.
  """
  def __init__(self, model: ResNet, optimizer: torch.optim.Optimizer, game: Connect4, args: dict):
      """
      Initializes the AlphaZero training.

      Args:
          model: The neural network model.
          optimizer: The optimizer for training.
          game: The game instance.
          args: A dictionary of hyperparameters.
      """
      self.model = model
      self.optimizer = optimizer
      self.game = game
      self.args = args
      self.mcts = AlphaMCTS(game, args, model)

  def selfPlay(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Performs a single episode of self-play.

    Returns:
      A list of tuples containing states, action probabilities, and the outcome.
    """
    memory = []
    player = 1
    state = self.game.get_initial_state()
    while True:
      negative_state = self.game.change_perspective(state, player)
      mcts_probs = self.mcts.search(negative_state)
      temperature_mcts_probs = mcts_probs ** (1 / self.args['temperature'])
      action = np.random.choice(self.game.action_size, p=temperature_mcts_probs)
      memory.append((negative_state, mcts_probs, player))
      state = self.game.get_next_state(state, action, player)
      value, is_finished = self.game.get_value_and_terminated(state, action)
      if is_finished:
          return_memory = []
          for hist_negative_state, hist_mcts_probs, hist_player in memory:
            hist_value = value if hist_player == player else self.game.get_opponent_value(value)
            return_memory.append((self.game.get_encoded_state(hist_negative_state), hist_mcts_probs, hist_value))
          return return_memory
      player = self.game.get_opponent(player)

  def train(self, memory: List[Tuple[np.ndarray, np.ndarray, float]]):
    """
    Trains the neural network on a batch of data.

    Args:
        memory: The training data containing the states, action probabilities, and outcomes.
    """
    random.shuffle(memory)
    for batch_idx in range(0,len(memory),self.args['batch_size']):
      sample = memory[batch_idx:min(len(memory) - 1,batch_idx + self.args['batch_size'])]
      state,policy_targets,value_targets = zip(*sample)
      state,policy_targets,value_targets = np.array(state),np.array(policy_targets),np.array(value_targets).reshape(-1,1)
      state = torch.tensor(state,dtype=torch.float32,device=self.model.device)
      policy_targets = torch.tensor(policy_targets,dtype=torch.float32,device=self.model.device)
      value_targets = torch.tensor(value_targets,dtype=torch.float32,device=self.model.device)

      out_policy, out_value = self.model(state)
      policy_loss = F.cross_entropy(out_policy,policy_targets)
      value_loss = F.mse_loss(out_value,value_targets)
      loss = policy_loss + value_loss

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def learn(self):
    """
    Starts the learning process for AlphaZero.
    """
    for iteration in trange(self.args['num_iterations']):
      memory = []
      self.model.eval()
      for _ in trange(self.args['num_selfPlay_iterations']):
          memory += self.selfPlay()

      self.model.train()
      for _ in trange(self.args['num_epochs']):
          self.train(memory)
      torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
      torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")

class AlphaMCTSParallel:
    """
    Implements a parallelized version of the AlphaZero Monte Carlo Tree Search.
    """
    def __init__(self, model: ResNet, game: Connect4, args: dict):
        """
        Initializes the parallelized AlphaMCTS.

        Args:
          model: The neural network model.
          game: An instance of the Connect4 game.
          args: A dictionary of hyperparameters.
        """
        self.model = model
        self.game = game
        self.args = args

    @torch.no_grad()
    def search(self, states: np.ndarray, sp_games: List['SelfPlayGame']):
        """
        Performs parallel MCTS search for given game states.

        Args:
          states: A stack of game states for parallel search.
          sp_games: A list of SelfPlayGame instances for each parallel game.
        """
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        valid_moves = self.game.get_valid_moves(states)
        policy *= valid_moves
        policy /= np.sum(policy, axis=1, keepdims=True)
        for i, g in enumerate(sp_games):
            g.root = Node(self.game, self.args, states[i], visit_count=1)
            g.root.expand(policy[i])
        for _ in range(self.args['num_mcts_searches']):
            for g in sp_games:
                g.node = None
                node = g.root
                while node.is_expanded():
                    node = node.select()
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                if is_terminal:
                    node.backpropagate(value)
                else:
                    g.node = node
            expandable_sp_games = [mapping_idx for mapping_idx in range(len(sp_games)) if sp_games[mapping_idx].node is not None]
            if len(expandable_sp_games) > 0:
              states = np.stack([sp_games[mapping_idx].node.state for mapping_idx in expandable_sp_games])
              policy, value = self.model(
                  torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
              )
              policy = torch.softmax(policy, axis=1).cpu().numpy()
              valid_moves = self.game.get_valid_moves(states)
              policy *= valid_moves
              policy /= np.sum(policy, axis=1, keepdims=True)
              value = value.cpu().numpy()

            for i, mapping_idx in enumerate(expandable_sp_games):
              node = sp_games[mapping_idx].node
              node.expand(policy[i])
              node.backpropagate(value[i])

class AlphaZeroParallel:
    """
    Implements the parallelized version of AlphaZero algorithm for self-play and training.
    """
    def __init__(self, model: ResNet, optimizer: torch.optim.Optimizer, game: Connect4, args: dict):
        """
        Initializes the parallelized AlphaZero training.

        Args:
          model: The neural network model.
          optimizer: The optimizer for training.
          game: The game instance.
          args: A dictionary of hyperparameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTSParallel(model, game, args)

    def selfPlay(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
      """
      Performs self-play using multiple game instances.

      Returns:
        List of training examples containing states, action probabilities and outcomes.
      """
      return_memory = []
      player = 1
      sp_games = [SelfPlayGame(self.game) for _ in range(self.args['num_parallel_games'])]
      while len(sp_games) > 0:
        states = np.stack([g.state for g in sp_games])
        neutral_states = self.game.change_perspective(states, player)
        self.mcts.search(neutral_states, sp_games)
        for i in range(len(sp_games))[::-1]:
          g = sp_games[i]
          action_probs = np.zeros(self.game.action_size)
          for child in g.root.children:
            action_probs[child.action_taken] = child.visit_count
          action_probs /= np.sum(action_probs)
          g.memory.append((g.root.state, action_probs, player))
          temperature_action_probs = action_probs ** (1 / self.args['temperature'])
          temperature_action_probs /= np.sum(temperature_action_probs)
          action = np.random.choice(self.game.action_size, p=temperature_action_probs)
          g.state = self.game.get_next_state(g.state, action, player)
          value, is_terminal = self.game.get_value_and_terminated(g.state, action)
          if is_terminal:
            for hist_neutral_state, hist_action_probs, hist_player in g.memory:
              hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
              return_memory.append((
                  self.game.get_encoded_state(hist_neutral_state),
                  hist_action_probs,
                  hist_outcome
                ))
            del sp_games[i]
        player = self.game.get_opponent(player)
      return return_memory

    def train(self, memory: List[Tuple[np.ndarray, np.ndarray, float]]):
      """
      Trains the neural network using a batch of training samples.

      Args:
          memory: The training data consisting of states, action prob, and outcomes.
      """
      random.shuffle(memory)
      for batch_idx in range(0, len(memory), self.args['batch_size']):
        sample = memory[batch_idx:batch_idx+self.args['batch_size']]
        state, policy_targets, value_targets = zip(*sample)
        state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
        state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
        out_policy, out_value = self.model(state)
        policy_loss = F.cross_entropy(out_policy, policy_targets)
        value_loss = F.mse_loss(out_value, value_targets)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
      """
      Initiates the training loop for the parallelized AlphaZero algorithm.
      """
      for iteration in range(self.args['num_iterations']):
          memory = []
          self.model.eval()
          for _ in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
              memory += self.selfPlay()
          self.model.train()
          for _ in trange(self.args['num_epochs']):
              self.train(memory)
          torch.save(self.model.state_dict(), f"{self.game}/model_{iteration}.pt")
          torch.save(self.optimizer.state_dict(), f"{self.game}/optimizer_{iteration}.pt")
class SelfPlayGame:
  """
  Represents a single game instance during self-play.
  """
  def __init__(self, game: Connect4):
    """
    Initializes the game instance.

    Args:
      game: The Connect4 game instance.
    """
    self.state = game.get_initial_state()
    self.memory: List[Tuple[np.ndarray, np.ndarray, int]] = []
    self.root: Union[Node, None] = None
    self.node: Union[Node, None] = None