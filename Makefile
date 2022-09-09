.PHONY: debug sweep mnist

debug:
	pipenv run python3 \
		-m tta.cli \
		--config_name debug \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 100 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_multiplier 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet True \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--seed 360234358 \
		--num_workers 48

sweep:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name source{1}_batch{2}_steps{3}_lr{4}_temp{5}_cali{6} \
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
		--num_workers 48 \
		::: 1 \
		::: 64 512 \
		::: 1000 2000 \
		::: 1e-2 1e-3 \
		::: 0.1 1 10 \
		::: 0 100 200


mnist: mnist-default mnist-calibrated mnist-marginal-false mnist-small-batch mnist-no-rotation mnist-unconfounded-source

mnist-default:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_multiplier 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title '' \
		--seed 360234358 \
		--num_workers 48

mnist-calibrated:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_apply_rotation True \
		--train_domains 9 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 10 \
		--calibration_multiplier 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title '' \
		--seed 360234358 \
		--num_workers 48

mnist-marginal-false:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_apply_rotation True \
		--train_domains 9 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_multiplier 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal False \
		--plot_title '' \
		--seed 360234358 \
		--num_workers 48

mnist-small-batch:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_apply_rotation True \
		--train_domains 9 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_multiplier 1 \
		--test_batch_size 64 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title '' \
		--seed 360234358 \
		--num_workers 48

mnist-no-rotation:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_apply_rotation False \
		--train_domains 9 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_multiplier 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title '' \
		--seed 360234358 \
		--num_workers 48

mnist-unconfounded-source:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_apply_rotation True \
		--train_domains 5 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps 1000 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_multiplier 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title '' \
		--seed 360234358 \
		--num_workers 48
