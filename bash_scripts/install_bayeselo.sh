#!/bin/bash

# Download and install BayesElo
wget https://www.remi-coulom.fr/Bayesian-Elo/bayeselo.tar.bz2
tar -xvjf bayeselo.tar.bz2

# Build with older c++ standard (fixes compilation errors with newer standards)
cd BayesElo
g++ -std=c++98 -o bayeselo -O3 -Wall bayeselo.cpp

# Return to original directory and clean up
cd ..
rm bayeselo.tar.bz2