import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import nn

class ResBlock(nn.Module):
    """
    Represents a Residual Block in the ResNet architecture.
    """
    def __init__(self, num_hidden: int) -> None:
      """
      Initializes a ResBlock

        Args:
          num_hidden: Number of channels in the block
      """
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      Forward call of ResBlock
      """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    """
    Represents the ResNet neural network architecture.
    """
    def __init__(self, game, num_res_blocks: int, num_hidden: int, device: str):
      """
      Initializes the ResNet architecture.

      Args:
        game: Instance of the game to use
        num_res_blocks: Number of resblocks to include
        num_hidden: Number of hidden channels in the network
        device: Whether to use CPU or CUDA
      """
        super().__init__()
        self.device = device

        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.back_bone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh(),
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      """
      Forward call of ResNet
      """
        x = self.start_block(x)
        for res_block in self.back_bone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value