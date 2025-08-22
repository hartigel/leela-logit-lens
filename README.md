# Iterative Inference in a Chess-Playing Neural Network

## Setup

First, install the necessary Python packages:

```bash
pip install -e .
```

Next, download the required data and model files.

> **📦 All-in-One Download**: For convenience, we've compiled all necessary files into a single Figshare repository: https://figshare.com/s/5342980a9ba8b26985a9. This includes models, datasets, and pre-computed results so you can skip directly to analysis if desired.

### Models

Download the Leela Chess Zero models from the "Evidence of Learned Look-Ahead" paper here: https://figshare.com/s/adc80845c00b67c8fce5 (also available in our all-in-one Figshare above).

Place the model files in your root working directory. For our experiments, we primarily used `lc0-original.onnx`, which is not finetuned and uses position history. The code also works with their finetuned model, `lc0.onnx`, with similar results.

### Datasets

- **Evaluation Puzzles & Openings**: These are from the "Amortized Planning with Large-Scale Transformers" paper (also available in our all-in-one Figshare above). Download them into a `data/` subdirectory.
  
  ```bash
  mkdir -p data
  wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv -P data/
  wget https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn -P data/
  ```

- **Augmented Puzzles**: We use an augmented version of the `interesting_puzzles.pkl` file from the "Evidence of Learned Look-Ahead" paper for some plots including the history of the games taken from Lichess. Both `.csv` and `.pkl` versions are available in our all-in-one Figshare above.
  
  > **Note**: You only need the `interesting_puzzles.csv` if you want to execute the puzzle solving analysis for the dataset used in the "Evidence of Learned Look-Ahead" paper. Since they filter the puzzles by different criteria, we used the `puzzles.csv` from the "Amortized Planning" paper for our analysis who didn't filter.
  
  > **Note**: If you wish to regenerate this file yourself, download the original `interesting_puzzles.pkl` and run the `scripts/puzzle_history_augmentation.py` script.

- **CCRL Dataset**: Used for the policy distribution metrics. Download and decompress it into the `data/` directory.
  
  ```bash
  wget http://storage.lczero.org/files/ccrl-pgn.tar.bz2 -P data/
  cd data/
  tar -xjf ccrl-pgn.tar.bz2
  cd ..
  ```

After completing the setup, your project directory should look like this:

    .
    ├── data/
    │   ├── puzzles.csv
    │   ├── eco_openings.pgn
    │   ├── interesting_puzzles_history.pkl (history annotated puzzles for plotting)
    │   └── ccrl/
    │       └── ... (extracted files)
    ├── notebooks/
    │   └── ...
    ├── results/
    │   ├── puzzle_results.csv
    │   └── tournament_games.pgn
    ├── scripts/
    │   └── ..
    ├── lc0-original.onnx
    ├── lc0.onnx (if you want to use the finetuned model)
    ├── Stockfish/
    │   └── ...
    ├── lc0/ (if you want to use the policy anchor in the tournament)
    │   └── ...
    └── 768x15x24h-t82-swa-7464000.pb (if you want to use the policy anchor in the tournament)

---

## Quickstart Demo 🚀

To get a feel for the core functionalities of this codebase, we recommend starting with the demo notebook located at `notebooks/demo.ipynb`. It provides an intuitive walkthrough of how to apply the logit lens to new puzzles or positions.

---

## Reproducing Results

This section details the steps to reproduce the main results from our paper.

### Main Figure

The code for plotting the main figure is available in `notebooks/figure1.ipynb`.

### Puzzle Solving Results

This experiment evaluates the model's ability to solve tactical puzzles.

1. Ensure `data/puzzles.csv` is downloaded and the `lc0-original.onnx` model is in the root directory.

2. Run the evaluation script. This will generate the raw results and save them.
   
   ```bash
   python scripts/evaluate_puzzles.py
   ```
   
   (Alternatively, you can simply invoke the provided `bash_scripts/evaluate_puzzles.sh` script).

3. The results will be stored in `results/puzzle_results.csv`.

4. Use the `notebooks/puzzle_results.ipynb` notebook for evaluation and to generate the final plots.

> **Shortcut**: You can download our pre-computed `puzzle_results.csv` from our all-in-one Figshare repository above and place it in a `results/` directory to skip directly to the analysis notebook.

> **Note**: The plots for the puzzle results are generated in `notebooks/puzzle_results.ipynb`.

### Tournament Elo Results

This experiment runs a round-robin tournament to determine the Elo strength of the model and its logit lens layers.

1. **Build Stockfish**: Follow the instructions in the "Amortized Planning" paper to build the Stockfish engine in the source directory or run this:

   ```bash
   bash_scripts/install_stockfish.sh
   ```

2. **Install BayesElo**: For the calculation of the Elo scores you need BayesElo which you can install as specified in the "Amortized Planning" paper or you run this bash script:
   
   ```bash
   bash_scripts/install_bayeselo.sh
   ```

3. **(Optional) Policy Anchor**: If you want to use the Leela policy anchor, install `lc0` on your system. Note that using the policy net anchor currently only works with CUDA.

4. **(Optional) Download Anchor Model**: Download the protobuf model files used in the reference paper (this is only needed for the policy anchor model):
   
   ```bash
   wget https://storage.lczero.org/files/768x15x24h-t82-swa-7464000.pb.gz
   gunzip 768x15x24h-t82-swa-7464000.pb.gz
   ```

5. Ensure `eco_openings.pgn` is in the `data/` directory.

6. **Run the Tournament**:
   
   ```bash
   python scripts/run_tournament.py
   ```
   
   (Alternatively, you can simply invoke the provided `scripts/run_tournament.sh` script).

7. The played games will be saved to `results/tournament_games.pgn`.

8. Use the `notebooks/tournament_results.ipynb` notebook to calculate the final Elo scores from the PGN file.

> **Shortcut**: You can download our pre-computed `tournament_games.pgn` file from our all-in-one Figshare repository above and place it in `results/` to skip directly to the analysis notebook.

### Policy Distribution Metrics

This analysis evaluates policy metrics on a large dataset of human games.

1. Ensure the CCRL dataset has been downloaded and extracted into the `data/` directory.

2. The entire analysis is performed within the `notebooks/policy_metrics.ipynb` notebook. Simply run the cells in order.

> **Note**: The plots for the policy metrics are generated in `notebooks/policy_metrics.ipynb`.

### Additional Examples

For the plots of the additional examples, see `notebooks/demo.ipynb` which also contains the plotting function for all layers of a given position and the generation of probability tables.

---

## Acknowledgments

This work was heavily inspired by and builds upon the research and public codebase from the paper **"Evidence of Learned Look-Ahead in a Chess-Playing Neural Network"** by Erik Jenner, Shreyas Kapur, Vasil Georgiev, Cameron Allen, Scott Emmons and Stuart Russell. We are very grateful for their foundational work and for making their models and tools publicly available.