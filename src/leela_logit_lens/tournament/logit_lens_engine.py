"""
Implements a logit lens engine based on the given Leela Chess Zero model.
Compatible with the engine protocol defined in the searchless chess paper.
"""


import chess
from searchless_chess.engines import engine  # The Engine protocol
from typing import Mapping, Any

from leela_interp import LeelaBoard


def _ensure_leela_board(board: chess.Board) -> "LeelaBoard":
    """
    Ensure that the provided board is a LeelaBoard. If it is a plain chess.Board,
    then instantiate a new LeelaBoard and replay its move history.
    """
    if isinstance(board, LeelaBoard):
        return board
    lb = LeelaBoard()
    for move in board.move_stack:
        lb.push(move)
    return lb


class LogitLensEngine(engine.Engine):
    def __init__(self, lens, layer_idx: int = 15):
        """
        Args:
            lens: An instance of LeelaLogitLens that implements the forward pass.
            layer_idx: The layer index at which to apply the logit lens.
        """
        self.lens = lens
        self.layer_idx = layer_idx

    def analyse(self, board: chess.Board) -> Mapping[str, Any]:
        """
        Run the lens on the board and return the policy dict.
        Ensures the board is a LeelaBoard before running the forward pass.
        """
        # Ensure we have a LeelaBoard instance.
        board = _ensure_leela_board(board)
        # Run the forward pass with probabilities returned.
        results = self.lens.forward(board, layer_idx=self.layer_idx, output="policy", return_probs=True,
                                    return_policy_as_dict=True)
        return {"results": results[0], "fen": board.fen()}

    def play(self,
             board: chess.Board
             ) -> chess.Move:
        """
        Returns the best move for the given board according to the logit lens.
        Ensures the board is a LeelaBoard before running the forward pass.
        """
        board = _ensure_leela_board(board)
        results = self.lens(board,
                            layer_idx=self.layer_idx,
                            output="policy",
                            return_probs=True,
                            return_policy_as_dict=True,
                            )
        # Extract the tensor (assuming batch dimension 1).
        policy_dict = results[0]["policy_as_dict"]
        # Choose the best move by maximum score.
        best_move_uci = max(policy_dict, key=policy_dict.get)
        return chess.Move.from_uci(best_move_uci)
