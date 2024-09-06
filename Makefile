# Define the Python executable and script
PYTHON=python
SCRIPT=main_optimizer.py

# Common arguments shared between all runs
COMMON_ARGS=--data_path data/meanSurface-new.json \
			--date_start 2019-01-01 \
			--date_end 2023-01-01 \
			--aggregator Datetime \
			--min_games 1 \
			--bets_per_game 5 \
			--optimizer BoTorchOptimizer \
			--probability_mapping softmax \
			--n_iterations 10 \
			--n_jobs 10 \
			--save_experiment

# Run-specific arguments
ARGS1=--weight 0.1
ARGS2=--weight 0.2
ARGS3=--weight 0.5
ARGS4=--weight 0.8
ARGS5=--weight 0.9

# Targets to run the script with different combinations
run1:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS) $(ARGS1)

run2:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS) $(ARGS2)

run3:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS) $(ARGS3)

run4:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS) $(ARGS4)

run5:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS) $(ARGS5)

# A target to run all combinations, this is like a hyperparameter tuning job to set the weight parameter with past data
run-all: run1 run2 run3 run4 run5

# Common arguments shared between all runs
COMMON_ARGS_LAMBDA=--data_path data/meanSurface-new.json \
			--date_start 2019-01-01 \
			--date_end 2022-01-01 \
			--aggregator Datetime \
			--min_games 1 \
			--bets_per_game 5 \
			--weight 0.2 \
			--optimizer BoTorchOptimizerLambda \
			--probability_mapping softmax \
			--n_iterations 10 \
			--n_jobs 10 \
			--save_experiment

# Run-specific arguments
ARGS_LAMBDA1=--lambda_param 1
ARGS_LAMBDA2=--lambda_param 5
ARGS_LAMBDA3=--lambda_param 50
ARGS_LAMBDA4=--lambda_param 5000

# Targets to run the script with different combinations
run-lambda1:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LAMBDA) $(ARGS_LAMBDA1)

run-lambda2:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LAMBDA) $(ARGS_LAMBDA2)

run-lambda3:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LAMBDA) $(ARGS_LAMBDA3)

run-lambda4:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LAMBDA) $(ARGS_LAMBDA4)

# A target to run all combinations, this is like a hyperparameter tuning job to set the lambda parameter with past data
run-lambda-all: run-lambda1 run-lambda2 run-lambda3 run-lambda4


# Common arguments shared between all runs
COMMON_ARGS_VARIABLE_STAKE=--data_path data/meanSurface-new.json \
			--date_start 2019-01-01 \
			--date_end 2022-01-01 \
			--aggregator Datetime \
			--min_games 1 \
			--bets_per_game 5 \
			--optimizer BoTorchOptimizerVariableStake \
			--probability_mapping softmax \
			--n_iterations 10 \
			--n_jobs 10 \
			--save_experiment

# Run-specific arguments
ARGS_VARIABLESTAKE1=--weight 0.1
ARGS_VARIABLESTAKE2=--weight 0.2
ARGS_VARIABLESTAKE3=--weight 0.5
ARGS_VARIABLESTAKE4=--weight 0.8
ARGS_VARIABLESTAKE5=--weight 0.9

# Targets to run the script with different combinations
run-variablestake1:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_VARIABLE_STAKE) $(ARGS_VARIABLESTAKE1)

run-variablestake2:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_VARIABLE_STAKE) $(ARGS_VARIABLESTAKE2)

run-variablestake3:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_VARIABLE_STAKE) $(ARGS_VARIABLESTAKE3)

run-variablestake4:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_VARIABLE_STAKE) $(ARGS_VARIABLESTAKE4)

run-variablestake5:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_VARIABLE_STAKE) $(ARGS_VARIABLESTAKE5)

# A target to run all combinations, this is like a hyperparameter tuning job to set the weight parameter with past data
run-variablestake-all: run-variablestake1 run-variablestake2 run-variablestake3 run-variablestake4 run-variablestake5


# Common arguments shared between all runs
COMMON_ARGS_LONG_TERM=--data_path data/meanSurface-new.json \
			--date_start 2022-09-01 \
			--date_end 2023-01-01 \
			--aggregator Datetime \
			--min_games 1 \
			--bets_per_game 5 \
			--optimizer LongTermOptimizer \
			--probability_mapping softmax \
			--n_jobs 10 \
			--save_experiment

# Run-specific arguments
ARGS_LONGTERM1=--weight 0.1
ARGS_LONGTERM2=--weight 0.2
ARGS_LONGTERM3=--weight 0.5
ARGS_LONGTERM4=--weight 0.8
ARGS_LONGTERM5=--weight 0.9


# Targets to run the script with different combinations
run-longterm1:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LONG_TERM) $(ARGS_LONGTERM1)

run-longterm2:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LONG_TERM) $(ARGS_LONGTERM2)

run-longterm3:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LONG_TERM) $(ARGS_LONGTERM3)

run-longterm4:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LONG_TERM) $(ARGS_LONGTERM4)

run-longterm5:
	$(PYTHON) $(SCRIPT) $(COMMON_ARGS_LONG_TERM) $(ARGS_LONGTERM5)

# A target to run all combinations, this is like a hyperparameter tuning job to set the weight parameter with past data
run-longterm-all: run-longterm1 run-longterm2 run-longterm3 run-longterm4 run-longterm5
