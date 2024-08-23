
# Define arrays for each parameter
data_path_values=("data/meanSurface-new.json" "data/medianSurface.json")
min_games_values=("0" "1")
bets_per_game_values=("2" "5" "10")
weight_values=("0.2" "0.5" "0.8")

# Loop through all combinations of parameters
for data_path in "${data_path_values[@]}"; do
  for min_games in "${min_games_values[@]}"; do
    for bets_per_game in "${bets_per_game_values[@]}"; do
      for weight in "${weight_values[@]}"; do
        echo "Running script with --data_path $data_path, --min_games $min_games, --bets_per_game $bets_per_game, --weight $weight, "
        python main_optimizer.py --data_path "$data_path" --date_start 2019-01-01 --date_end 2023-01-01 --min_games "$min_games" --bets_per_game "$bets_per_game" --weight "$weight" --aggregator Datetime --n_iterations 20 --n_jobs 11 --optimizer BoTorchOptimizerVariableStake --save_experiment
      done
    done
  done
done