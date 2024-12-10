# soccer-betting-strategy

This repository forms the core of my master's thesis, where I'll be focusing on developing and applying a soccer match betting strategy. My approach involves a fusion of statistical analysis, machine learning techniques, and numerical optimization, primarily using Python as the main programming language. 

In addition to data-driven modeling, I'll also emphasize risk management. Advanced numerical optimization methods will be used to optimize betting portfolios, considering factors like bankroll management and risk tolerance. The ultimate goal of this project is to provide a well-rounded approach to soccer betting that combines data science, statistical insights, and numerical techniques to make more informed and potentially profitable decisions.


python main_baseline_kreiner.py \
    --aggregator Datetime \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator GameId \
    --do_baseline \
    --save_experiment

python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-08-06 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --optimizer BoTorchOptimizer \
    --probability_mapping softmax \
    --n_iterations 50 \
    --n_jobs 1 \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2019-11-09 \
    --date_end 2019-11-15 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --optimizer BoTorchOptimizer \
    --probability_mapping softmax \
    --n_iterations 100 \
    --n_jobs 1 \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --optimizer BoTorchOptimizerVariableStake \
    --probability_mapping softmax \
    --n_iterations 100 \
    --n_jobs 10 \
    --save_experiment

python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --lambda_param 0.5 \
    --optimizer BoTorchOptimizerVariableStakeLambda \
    --probability_mapping softmax \
    --n_iterations 50 \
    --n_jobs 10 \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.1 \
    --optimizer MOBO \
    --probability_mapping softmax \
    --n_iterations 70 \
    --n_jobs 10 \
    --save_experiment


kernprof -l -v main_botorch.py \
    --date_start 2022-01-01 \
    --date_end 2023-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --bets_per_game 5 \
    --weight 0.5 \
    --optimizer BoTorchOptimizerVariableStake \
    --probability_mapping softmax \
    --n_iterations 150 \
    --n_jobs 1 \
    --save_experiment


python main_optimizer.py \
    --date_start 2022-01-01 \
    --date_end 2023-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --bets_per_game 5 \
    --weight 0.5 \
    --optimizer Optimizer \
    --probability_mapping softmax \
    --n_jobs 1 \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --optimizer LongTermOptimizeSharpeRatio \
    --probability_mapping softmax \
    --n_jobs 10 \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --optimizer LongTermOptimizerProb \
    --probability_mapping sparsemax \
    --n_jobs 10 \
    --save_experiment

python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --optimizer LongTermOptimizerProb \
    --probability_mapping softmax \
    --n_jobs 10 \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --bookmakers 'Stake.com' \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --lambda_param 0.1 \
    --optimizer BoTorchOptimizerLambda \
    --probability_mapping softmax \
    --n_iterations 100 \
    --n_jobs 10 \
    --save_experiment


python main_optimizer.py \
    --data_path data/meanSurface-new.json \
    --date_start 2023-01-01 \
    --date_end 2024-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --bookmakers 'Betobet' \
    --filter_out_bookmakers \
    --max_bets 15 \
    --bets_per_game 5 \
    --weight 0.2 \
    --lambda_param 0.1 \
    --optimizer BoTorchOptimizerLambda \
    --probability_mapping softmax \
    --n_iterations 100 \
    --n_jobs 10 \
    --save_experiment

odds_dt[['GameId', 'Odd']]

## About the code
I have found
2019-11-10 a selected bet with an odd of 41, GameId 2983655

## About the data
- odds-new-correct: dataset that for 2019-2023
- metadata-with-date-new: dataset that contains the metadata for 2019-2023
- meanSurface-new: dataset that contains the mean surface of the 7x7 grid scores for every game id for 2019-2023
- medianSurface-new: dataset that contains the median surface of the 7x7 grid scores for every game id for 2019-2023. The use of the median is to reduce the impact of highly skewed probability estimates for the 7x7 grid scores.

## Considerations
- It seems that a minimum of 100 iterations is necessary for BoTorch optimization to return good results

* If we consider more bookmakers, the betting strategy will have a higher probability of success, because in that case there more opportunities to select. But, to consider too many bookmakers can bring such a high complexity if the system is implemented in a production pipeline that puts real money on the line.

## About conda environment
- The conda environment is stored in the environment.yml file. To create the environment, run the following command:

```bash
conda env create -f environment.yml
```
I have created multiple environments along the way, and I have provided the one that I think is the most up-to-date and enabled better performance, in a general sense. (soccer_betting_strategy_improved_v3_env)
