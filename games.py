import numpy as np

class TicTacToe:
    """
    Implements the Tic Tac Toe game logic.
    """
    def __init__(self) -> None:
      """
      Initialize the Tic Tac Toe game
      """
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self) -> str:
        return "TicTacToe"

    def get_initial_state(self) -> np.ndarray:
        """
        Returns the initial state of the game board.
        """
        return np.zeros((self.row_count, self.column_count))

    def play_a_move(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """
        Makes a move on the board and returns the updated state.
        """
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of valid moves from a current board state.
        """
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int) -> bool:
       """
       Checks if the last move resulted in a win
       """
        if action is None:
            return False
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def check_final_state(self, state: np.ndarray, action: int) -> Tuple[int, bool]:
        """
        Checks if the game is over due to a win or draw
        """
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def flip_player(self, player: int) -> int:
      """
      Flips the player value
      """
        return -player

    def flip_value(self, value: int) -> int:
      """
      Flips the value
      """
        return -value

    def flip_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
      """
      Flips the board to respect the player's perspective
      """
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
      """
      Encodes the board state to the neural network
      """
        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(
            np.float32
        )
        return encoded_state

    def get_encoded_state_parallel(self, state: np.ndarray) -> np.ndarray:
      """
      Encodes the board state in a parallel fashion (Used in parallel algorithms)
      """
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state

class Connect4:
    """
    Implements the Connect 4 game logic
    """
    def __init__(self) -> None:
        """
        Initialize the Connect 4 game
        """
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self) -> str:
        return "Connect4"

    def get_initial_state(self) -> np.ndarray:
      """
      Returns the initial state of the game board.
      """
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """
        Makes a move on the board and returns the updated state.
        """
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
       """
       Returns a boolean mask of valid moves from a current board state
       """
        if len(state.shape) == 3:
            return (state[:, 0] == 0).astype(np.uint8)
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int) -> bool:
        """
        Checks if the last move resulted in a win
        """
        if action is None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row: int, offset_column: int) -> int:
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1  # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1  # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1  # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1  # top right diagonal
        )

    def get_value_and_terminated(self, state: np.ndarray, action: int) -> Tuple[int, bool]:
       """
       Checks if the game is over due to a win or draw
       """
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int) -> int:
      """
       Flips the value of the player
      """
        return -player

    def get_opponent_value(self, value: int) -> int:
      """
       Flips the value of the value provided
      """
        return -value

    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
      """
      Flips the board to respect the player's perspective
      """
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
      """
      Encodes the board state to the neural network
      """
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state