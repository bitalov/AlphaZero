import math
import numpy as np
import torch
from typing import List, Union, TYPE_CHECKING
from utils import math, np

if TYPE_CHECKING:
    from games import Connect4

class Node:
  """
  Represents a node in the Monte Carlo Tree Search tree.
  """
  def __init__(self, game: 'Connect4', args: dict, state: np.ndarray, parent = None, action_taken: Union[int, None] = None, prior: float = 0, visit_count: int = 0):
    """
    Initializes a MCTS Node.

    Args:
      game: The game the node belongs to.
      args: Dictionary of hyperparameters.
      state: The game state represented at this node.
      parent: Parent node in the MCTS tree.
      action_taken: The action that led to this node from the parent.
      prior: Prior probability assigned to this node.
      visit_count: Number of times the node has been visited.
    """
    self.game = game
    self.args = args
    self.state = state
    self.parent = parent
    self.action_taken = action_taken
    self.prior = prior
    self.children: List['Node'] = []

    self.visit_count: int = visit_count
    self.value_sum: float = 0

  def is_expanded(self) -> bool:
    """
    Checks if the node has been expanded.

    Returns:
       True if the node has been expanded, false otherwise.
    """
    return len(self.children) > 0

  def select(self) -> 'Node':
    """
    Selects a child node based on the UCB.

    Returns:
      The selected child node.
    """
    best_child = None
    best_ucb = -np.inf

    for child in self.children:
      ucb = self.get_ucb(child)
      if ucb > best_ucb:
          best_child = child
          best_ucb = ucb
    return best_child

  def get_ucb(self, child: 'Node') -> float:
    """
    Calculates the Upper Confidence Bound (UCB) for a child node.

    Args:
      child: The child node.
    Returns:
        UCB value for the child node
    """
    if child.visit_count == 0:
          q_value = 0
    else:
          q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
    return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

  def expand(self, policy: np.ndarray):
    """
    Expands the node by creating child nodes for each possible action.

    Args:
      policy: The policy distribution over the valid actions
    """
    for action, prob in enumerate(policy):
      if np.isclose(prob, 0):
          continue
      child_state = self.state.copy()
      child_state = self.game.get_next_state(child_state, action, 1)
      child_state = self.game.change_perspective(child_state, player=-1)
      child = Node(self.game, self.args, child_state, self, action, prob)
      self.children.append(child)

  def backpropagate(self, value: float):
    """
    Updates the value_sum and visit_count.

    Args:
      value: Value of the state at the current node.
    """
    self.value_sum += value
    self.visit_count += 1
    if self.parent is not None:
        value = self.game.get_opponent_value(value)
        self.parent.backpropagate(value)

class AlphaMCTS:
    """
    Implements the Alpha Zero Monte Carlo Tree Search.
    """
    def __init__(self, game: 'Connect4', args: dict, model):
        """
        Initializes the AlphaMCTS.

        Args:
          game: An instance of the Connect4 game.
          args: A dictionary of hyperparameters.
          model: A neural network model for policy and value prediction.
        """
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state: np.ndarray) -> np.ndarray:
       """
        Performs a single MCTS search from the given state.

        Args:
          state: The current state of the game.
        Returns:
          A numpy array representing the action probabilities
        """
        root = Node(self.game, self.args, state, visit_count=1)
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        for _ in range(self.args['num_searches']):
            node = root
            while node.is_expanded():
                node = node.select()
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            if not is_terminal:
              policy, value = self.model(
                torch.tensor(self.game.get_encoded_state(node.state),device=self.model.device).unsqueeze(0)
              )
              policy = torch.softmax(policy,axis = 1).squeeze(0).cpu().numpy()
              valid_moves = self.game.get_valid_moves(node.state)
              policy *= valid_moves
              policy /= np.sum(policy)
              value = value.item()
              node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

class MCTS:
  """
  Implements Monte Carlo Tree Search.
  """
  def __init__(self,game:'Connect4',args:dict):
    """
    Initializes MCTS

    Args:
      game: An instance of the Connect4 game.
      args: A dictionary of hyperparameters
    """
    self.game = game
    self.args = args

  def search(self,state:np.ndarray):
    """
    Performs a single MCTS search from the given state.

    Args:
      state: The current state of the game.
    Returns:
      A numpy array representing the action probabilities
    """
    root = Node(self.game,self.args,state)
    for _ in range(self.args['num_searches']):
      node = root
      while node.is_expanded():
        node = node.select()
      value, is_terminal = self.game.get_value_and_terminated(node.state,node.action_taken)
      value = self.game.get_opponent_value(value)
      if not is_terminal:
        pass #Simulate the rollout from here if needed
      node.backpropagate(value)

    action_probs = np.zeros(self.game.action_size)
    for child in root.children:
      action_probs[child.action_taken] = child.visit_count
    action_probs /= np.sum(action_probs)
    return action_probs