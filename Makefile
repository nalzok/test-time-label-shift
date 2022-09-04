.PHONY: run sweep

run:
	pipenv run python3 \
		-m tta.cli \
		--dataset_name MNIST \
		--train_domains 1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-2 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 10 \
		--calibration_steps 200 \
		--calibration_multiplier 1 \
		--test_batch_size 64 --test_batch_size 512 \
		--test_symmetric_dirichlet False --test_symmetric_dirichlet True \
		--test_pseudocount_factor 0 --test_pseudocount_factor 1 \
		--test_fix_marginal False --test_fix_marginal True \
		--seed 360234358 \
		--num_workers 8 \
		--log_dir logs

sweep:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--dataset_name MNIST \
		--train_domains 0 \
		--train_batch_size {1} \
		--train_fraction 0.8 \
		--train_num_layers {2} \
		--train_steps {3} \
		--train_lr {4} \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature {5} \
		--calibration_steps {6} \
		--calibration_multiplier {7} \
		--test_batch_size 512 \
		--test_pseudocount_factor 1 \
		--test_fix_marginal True \
		--seed 360234358 \
		--num_workers 8 \
		--log_dir logs/batch{1}_res{2}_steps{3}_lr{4}_temp{5}_cali{6}_multi{7} \
		::: 64 128 256 \
		::: 18 50 \
		::: 500 1000 2000 \
		::: 1e-2 1e-3 1e-4 \
		::: 0.1 1 10 \
		::: 50 100 200 \
		::: 0.1 1
