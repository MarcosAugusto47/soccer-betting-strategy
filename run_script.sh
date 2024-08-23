#!/bin/bash

# Define arrays for each parameter
min_games_values=("0" "1")
bets_per_game_values=("2" "5" "10")
weight_values=("0.2" "0.5" "0.8")
probability_mapping_values=("softmax" "sparsemax")

# Loop through all combinations of parameters
for min_games in "${min_games_values[@]}"; do
  for bets_per_game in "${bets_per_game_values[@]}"; do
    for weight in "${weight_values[@]}"; do
      for probability_mapping in "${probability_mapping_values[@]}"; do  
        echo "Running script with --min_games $min_games, --bets_per_game $bets_per_game, --weight $weight, --probability_mapping $probability_mapping"
        python main_callback_parallel_botorch.py --min_games "$min_games" --bets_per_game "$bets_per_game" --weight "$weight" --probability_mapping "$probability_mapping" --aggregator Datetime --n_iterations 20 --n_jobs 10 --save_experiment
      done
    done
  done
done
