# soccer-betting-strategy

This repository forms the core of my master's thesis, where I'll be focusing on developing and applying a soccer match betting strategy. My approach involves a fusion of statistical analysis, machine learning techniques, and numerical optimization, primarily using Python as the main programming language. 

In addition to data-driven modeling, I'll also emphasize risk management. Advanced numerical optimization methods will be used to optimize betting portfolios, considering factors like bankroll management and risk tolerance. The ultimate goal of this project is to provide a well-rounded approach to soccer betting that combines data science, statistical insights, and numerical techniques to make more informed and potentially profitable decisions.

python main_callback_parallel_botorch.py \
    --date_start 2022-01-01 \
    --date_end 2023-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --bets_per_game 5 \
    --weight 0.5 \
    --probability_mapping softmax \
    --n_iterations 100 \
    --n_jobs 1 \
    --save_experiment

python main_baseline_kreiner.py \
    --aggregator Datetime \
    --save_experiment

python main_callback_parallel_variable_stake.py \
    --date_start 2019-01-01 \
    --date_end 2020-01-01 \
    --aggregator Datetime \
    --min_games 1 \
    --save_experiment

* If we consider more bookmakers, the betting strategy will have a higher probability of success, because in that case there more opportunities to select. But, to consider too many bookmakers can bring such a high complexity if the system is implemented in a production pipeline that puts real money on the line.
