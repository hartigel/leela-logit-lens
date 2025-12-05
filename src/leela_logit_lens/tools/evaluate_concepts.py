"""
concept_evaluation.py

Implementation for evaluating chess concept preferences across neural network layers.

Pipeline:
1. Take a position and evaluate it with Stockfish to get initial concept values
2. Generate all possible moves and evaluate resulting positions with Stockfish
3. Calculate concept deltas (resulting_position - initial_position) for each move
4. For each layer, get move policies and compute weighted deltas
5. Aggregate across all moves to get layer's concept preference deltas

All results are from the perspective of the player to move (mine/opponent naming).
"""

import subprocess
import time
import chess
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from leela_interp import LeelaBoard
from leela_logit_lens import LeelaLogitLens
from leela_logit_lens.tools.concept_spec import (STOCKFISH_CONCEPT_KEYS, MINE_OPPONENT_CONCEPT_KEYS,
                          extract_concept_values, convert_to_mine_opponent_perspective)


class StockfishEvaluator:
    """Stockfish wrapper for position evaluation."""

    def __init__(self, stockfish_path: str):
        self.process = subprocess.Popen(
            [stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        time.sleep(0.1)

        self._send_command("uci")
        self._wait_for_response("uciok", timeout=5.0)
        self._send_command("isready")
        self._wait_for_response("readyok", timeout=5.0)

    def _send_command(self, command: str) -> None:
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()

    def _wait_for_response(self, target: str, timeout: float = 2.0) -> None:
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.process.poll() is not None:
                raise RuntimeError("Stockfish terminated")

            line = self.process.stdout.readline().strip()
            if line and target in line:
                return

            time.sleep(0.01)

        raise TimeoutError(f"Timeout waiting for '{target}'")

    def _read_until_marker(self, end_marker: str, timeout: float = 10.0) -> List[str]:
        start_time = time.time()
        lines = []

        while (time.time() - start_time) < timeout:
            if self.process.poll() is not None:
                raise RuntimeError("Stockfish terminated")

            line = self.process.stdout.readline().strip()
            if line:
                lines.append(line)
                if end_marker in line:
                    return lines

            time.sleep(0.01)

        raise TimeoutError(f"Timeout waiting for '{end_marker}'")

    def evaluate_position(self, fen: str) -> Dict[str, float]:
        self._send_command("ucinewgame")
        self._send_command(f"position fen {fen}")
        self._send_command("isready")
        self._wait_for_response("readyok")
        self._send_command("eval")

        lines = self._read_until_marker("EVAL_END")

        result = {}
        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                try:
                    result[key] = float(value.strip())
                except ValueError:
                    result[key] = value.strip()

        return result

    def close(self) -> None:
        if hasattr(self, 'process') and self.process:
            try:
                self._send_command("quit")
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                self.process.kill()
                self.process.wait()


def evaluate_single_move_worker(args):
    """Worker function for parallel evaluation of a single move."""
    move, initial_fen, stockfish_path, white_to_move = args

    stockfish = StockfishEvaluator(stockfish_path)

    board = chess.Board(initial_fen)
    board.push_uci(move)

    result = stockfish.evaluate_position(board.fen())
    stockfish.close()

    stockfish_concepts = extract_concept_values(result)
    mine_opponent_concepts = convert_to_mine_opponent_perspective(stockfish_concepts, white_to_move)

    return move, mine_opponent_concepts


def evaluate_moves_parallel(board: chess.Board,
                            legal_moves: List[str],
                            stockfish_path: str,
                            white_to_move: bool,
                            max_workers: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all moves in parallel using multiple Stockfish processes.

    Args:
        board: Chess board in initial position
        legal_moves: List of UCI move strings
        stockfish_path: Path to Stockfish executable
        white_to_move: Whether white is to move in original position
        max_workers: Number of parallel workers (default: min(num_moves, cpu_count))

    Returns:
        Dict mapping moves to their concept values
    """
    if max_workers is None:
        max_workers = min(len(legal_moves), mp.cpu_count())

    args_list = [(move, board.fen(), stockfish_path, white_to_move)
                 for move in legal_moves]

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for move, concepts in executor.map(evaluate_single_move_worker, args_list):
            results[move] = concepts

    return results


def calculate_concept_deltas(initial_concepts: Dict[str, float],
                             move_concepts: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate concept deltas for each move.

    Args:
        initial_concepts: Concept values for initial position
        move_concepts: Dict mapping moves to their resulting concept values

    Returns:
        Dict mapping moves to their concept deltas
    """
    deltas = {}

    for move, resulting_concepts in move_concepts.items():
        move_deltas = {}
        for concept_key in MINE_OPPONENT_CONCEPT_KEYS:
            initial_value = initial_concepts[concept_key]
            resulting_value = resulting_concepts[concept_key]
            move_deltas[concept_key] = resulting_value - initial_value
        deltas[move] = move_deltas

    return deltas


def calculate_weighted_deltas(move_policies: Dict[str, float],
                              concept_deltas: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate probability-weighted concept deltas.

    Args:
        move_policies: Dict mapping moves to their probabilities
        concept_deltas: Dict mapping moves to their concept deltas

    Returns:
        Dict of weighted concept deltas
    """
    weighted_deltas = {concept_key: 0.0 for concept_key in MINE_OPPONENT_CONCEPT_KEYS}

    for move, prob in move_policies.items():
        for concept_key in MINE_OPPONENT_CONCEPT_KEYS:
            delta = concept_deltas[move][concept_key]
            weighted_deltas[concept_key] += prob * delta

    return weighted_deltas


def evaluate_positions_by_layer(boards: List[LeelaBoard],
                                lens: LeelaLogitLens,
                                stockfish: StockfishEvaluator,
                                layer_indices: Optional[List[int]] = None,
                                max_workers: Optional[int] = None,
                                stockfish_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Evaluate concept preferences across layers for multiple positions.

    Args:
        boards: List of board positions to evaluate
        lens: LeelaLogitLens instance
        stockfish: StockfishEvaluator instance
        layer_indices: Layer indices to evaluate (default: all layers)
        max_workers: Number of parallel workers for Stockfish (default: min(num_moves, cpu_count))
        stockfish_path: Path to Stockfish executable (required for parallel evaluation)

    Returns:
        List of dictionaries with results for each position
    """
    if layer_indices is None:
        layer_indices = list(range(lens.num_layers + 1))

    if stockfish_path is None:
        raise ValueError("stockfish_path is required for parallel evaluation")

    results = []

    for board in tqdm(boards, desc="Processing boards", mininterval=20.0):
        white_to_move = board.pc_board.turn == chess.WHITE

        initial_stockfish_eval = stockfish.evaluate_position(board.pc_board.fen())
        initial_stockfish_concepts = extract_concept_values(initial_stockfish_eval)
        initial_concepts = convert_to_mine_opponent_perspective(initial_stockfish_concepts, white_to_move)

        legal_moves = [move.uci() for move in board.pc_board.legal_moves]

        move_concepts = evaluate_moves_parallel(
            board=board.pc_board,
            legal_moves=legal_moves,
            stockfish_path=stockfish_path,
            white_to_move=white_to_move,
            max_workers=max_workers
        )

        concept_deltas = calculate_concept_deltas(initial_concepts, move_concepts)

        layer_results = lens.multi_layer_lens(
            boards=[board],
            layer_indices=layer_indices,
            output="policy",
            return_probs=True,
            return_policy_as_dict=True
        )

        board_layer_results = layer_results[0]

        position_result = {
            "fen": board.pc_board.fen(),
            "white_to_move": white_to_move,
            "layer_evaluations": {}
        }

        for layer_idx in layer_indices:
            layer_data = board_layer_results["layers"][layer_idx]
            move_policies = layer_data["policy_as_dict"]
            weighted_deltas = calculate_weighted_deltas(move_policies, concept_deltas)
            position_result["layer_evaluations"][layer_idx] = weighted_deltas

        results.append(position_result)

    return results
