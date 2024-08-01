
# Define arrays for each parameter
min_games_values=("0" "1")
bets_per_game_values=("2" "5" "8")
weight_values=("0.1" "0.2" "0.5" "0.8" "0.9")

# Loop through all combinations of parameters
for min_games in "${min_games_values[@]}"; do
  for bets_per_game in "${bets_per_game_values[@]}"; do
    for weight in "${weight_values[@]}"; do
      echo "Running script with --min_games $min_games, --bets_per_game $bets_per_game, --weight $weight, "
      python main_callback_parallel_botorch.py --min_games "$min_games" --bets_per_game "$bets_per_game" --weight "$weight" --aggregator Datetime --n_iterations 50 --n_jobs 10 --save_experiment
    done
  done
done
