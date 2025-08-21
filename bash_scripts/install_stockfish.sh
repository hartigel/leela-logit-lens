#!/bin/bash

# Simple Stockfish installation
set -e

echo "Installing Stockfish..."

# Clone Stockfish
git clone https://github.com/official-stockfish/Stockfish.git

# Build with default settings
cd Stockfish/src
make build

echo "Stockfish installed at: $(pwd)/stockfish"