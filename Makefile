.PHONY: debug sweep

debug:
	pipenv run python3 \
		-m tta.cli \
		--dataset_name MNIST \
		--train_domains 2 \
		--train_batch_size 512 \
		--train_fraction 1.0 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-2 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.0 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_multiplier 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_fix_marginal True \
		--seed 360234358 \
		--num_workers 8 \
		--log_path logs/debug.txt \
		--plot_path plots/debug.png \
		--accuracy_path accuracy/debug.npz \
		--norm_path norm/debug.npz

sweep:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--dataset_name MNIST \
		--train_domains {1} \
		--train_batch_size {2} \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps {3} \
		--train_lr {4} \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature {5} \
		--calibration_steps {6} \
		--calibration_multiplier 1 \
		--test_batch_size 64 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 8 \
		--test_fix_marginal False \
		--test_fix_marginal True \
		--seed 360234358 \
		--num_workers 8 \
		--log_path logs/source{1}_batch{2}_steps{3}_lr{4}_temp{5}_cali{6}.txt \
		--plot_path plots/source{1}_batch{2}_steps{3}_lr{4}_temp{5}_cali{6}.png \
		--accuracy_path accuracy/source{1}_batch{2}_steps{3}_lr{4}_temp{5}_cali{6}.npz \
		--norm_path norm/source{1}_batch{2}_steps{3}_lr{4}_temp{5}_cali{6}.npz \
		::: 1 \
		::: 64 512 \
		::: 1000 2000 \
		::: 1e-2 1e-3 \
		::: 0.1 1 10 \
		::: 0 100 200
