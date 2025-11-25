"""
Implements a logit lens engine based on the given Leela Chess Zero model.
Compatible with the engine protocol defined in the searchless chess paper.
"""


import chess
from searchless_chess.engines import engine  # The Engine protocol
from typing import Mapping, Any
import torch

from leela_interp import LeelaBoard, Lc0Model


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
    def __init__(self, model, lens, layer_idx: int = 15, temperature: float = 1.0):
        """
        Args:
            model: Lc0Model instance that provides a method
                   policy_as_dict(board, tensor) to convert a tensor into a dict mapping UCI moves to scores.
            lens: An instance of your AlteredLeelaLogitLens (or similar) that implements the forward pass.
            layer_idx: The layer index at which to apply the logit lens.
            temperature: Sampling temperature. 0 = argmax, 1.0 = sample from raw distribution.
        """
        self.model = model
        self.lens = lens
        self.layer_idx = layer_idx
        self.temperature = temperature

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
        Returns sampled move for the given board according to the logit lens.
        Sampling depends on temperature.
        Ensures the board is a LeelaBoard before running the forward pass.
        """
        board = _ensure_leela_board(board)

        # Get logits (not probabilities)
        results = self.lens.forward(
            board,
            layer_idx=self.layer_idx,
            output="policy",
            return_probs=False,
            return_policy_as_dict=False,
        )

        # Get policy logits tensor of shape (1858,)
        policy_logits = results[0]["policy"]

        # Apply temperature scaling
        if self.temperature == 0:
            scaled_logits = policy_logits
        else:
            scaled_logits = policy_logits / self.temperature

        # Convert to probabilities with legal move masking
        policy_probs = self.model.logits_to_probs(board, scaled_logits)[0]

        # Get legal moves
        legal_indices, legal_uci = self.model.legal_moves(board)

        # Extract probabilities for legal moves only
        legal_probs = policy_probs[legal_indices]

        # Sample or argmax based on temperature
        if self.temperature == 0:
            selected_legal_idx = torch.argmax(legal_probs).item()
        else:
            selected_legal_idx = torch.multinomial(legal_probs, num_samples=1).item()

        # Map back to UCI
        sampled_move_uci = legal_uci[selected_legal_idx]

        return chess.Move.from_uci(sampled_move_uci)
