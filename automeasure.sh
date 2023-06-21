#!/bin/bash

set -xe

rm -f individuals*
rm -f indrun*

mkdir -p ./results

mkdir -p ./results/n_rows/
python3 acrorepeater.py --test-param="--n_rows" --test-values="1,2,4,8" --test-replays 20
mv individuals* ./results/n_rows/
mv indrun* ./results/n_rows/

mkdir -p ./results/levels_back/
python3 acrorepeater.py --test-param="--levels_back" --test-values="0,1,2,4,8,16" --test-replays 20
mv individuals* ./results/levels_back/
mv indrun* ./results/levels_back/

mkdir -p ./results/n_offsprings/
python3 acrorepeater.py --test-param="--n_offsprings" --test-values="1,2,4,8" --test-replays 20
mv individuals* ./results/n_offsprings/
mv indrun* ./results/n_offsprings/

mkdir -p ./results/tournament_size/
python3 acrorepeater.py --test-param="--tournament_size" --test-values="1,2,4,8" --test-replays 20
mv individuals* ./results/tournament_size/
mv indrun* ./results/tournament_size/

mkdir -p ./results/mutation_rate/
python3 acrorepeater.py --test-param="--mutation_rate" --test-values="0.01,0.04,0.08,0.16,0.32" --test-replays 20
mv individuals* ./results/mutation_rate/
mv indrun* ./results/mutation_rate/

mkdir -p ./results/n_parents/
python3 acrorepeater.py --test-param="--n_parents" --test-values="1,2,4,8,16" --test-replays 20
mv individuals* ./results/n_parents/
mv indrun* ./results/n_parents/

mkdir -p ./results/n_columns/
python3 acrorepeater.py --test-param="--n_columns" --test-values="4,8,16,32,64" --test-replays 20
mv individuals* ./results/n_columns/
mv indrun* ./results/n_columns/
