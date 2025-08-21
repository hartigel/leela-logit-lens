# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements a Leela Chess Zero engine. This file is unaltered from the searchless chess paper."""

import os

import chess.engine

from searchless_chess.engines import engine
from leela_interp import LeelaBoard

class Lc0Engine(engine.Engine):
  """Leela Chess Zero with the biggest available network.

  WARNING: This can only be used with CUDA (i.e. Nvidia GPUs) for now.
  """

  def __init__(
      self,
      limit: chess.engine.Limit,
  ) -> None:
    self._limit = limit
    bin_path = '/opt/leela-logit-lens/searchless_chess/lc0/build/release/lc0'

    # We use the same network as in the searchless chess paper.
    weights_path = os.path.join(
        os.getcwd(),
        '/opt/leela-logit-lens/searchless_chess/lc0/build/release/768x15x24h-t82-swa-7464000.pb',
    )
    options = [f'--weights={weights_path}', '--backend=cuda-auto']
    self._raw_engine = chess.engine.SimpleEngine.popen_uci(
        command=[bin_path] + options,
    )
    self._raw_engine.configure({'Threads': 1})

  def __del__(self) -> None:
    self._raw_engine.close()

  @property
  def limit(self) -> chess.engine.Limit:
    return self._limit

  def analyse(self, board: LeelaBoard) -> engine.AnalysisResult:
    """Returns various analysis results from the Lc0 engine."""
    outcome = board.pc_board.outcome()
    if outcome is not None:
      # The game has now ended.
      if outcome.winner is None:
        score = chess.engine.Cp(0)
      elif outcome.winner == board.pc_board.turn:
        score = -chess.engine.Mate(moves=0)
      else:
        score = chess.engine.Mate(moves=0)
      return {'score': chess.engine.PovScore(score, turn=board.pc_board.turn)}
    return self._raw_engine.analyse(board.pc_board, limit=self._limit)

  def play(self, board: LeelaBoard) -> chess.Move:
    """Returns the best move from the Lc0 engine."""
    best_move = self._raw_engine.play(board.pc_board, limit=self._limit).move
    if best_move is None:
      raise ValueError('No best move found, something went wrong.')
    return best_move


# class AllMovesLc0Engine(Lc0Engine):
#   """A version of Lc0 that evaluates all moves individually."""
#
#   def analyse(self, board: chess.Board) -> engine.AnalysisResult:
#     """Returns analysis results from Lc0."""
#     scores = []
#     sorted_legal_moves = engine.get_ordered_legal_moves(board)
#     for move in sorted_legal_moves:
#       board.push(move)
#       results = super().analyse(board)
#       board.pop()
#       scores.append((move, -results['score'].relative))
#     return {'scores': scores}
#
#   def play(self, board: chess.Board) -> chess.Move:
#     """Returns the best move from Lc0."""
#     scores = self.analyse(board)['scores']
#     sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
#     return sorted_scores[0][0]
