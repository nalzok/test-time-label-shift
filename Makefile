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
		--train_steps 1000 \
		--train_lr 1e-4 \
		--source_prior_estimation induce \
		--calibration_batch_size 64 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_lr 1e-4 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
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
		--config_name sweep_Tbatch{1}_Tsteps{2}_Tlr{3}_Cbatch{4}_Ctemp{5}_Csteps{6}_Clr{7} \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_batch_size {1} \
		--train_fraction 0.8 \
		--train_num_layers 18 \
		--train_steps {2} \
		--train_lr {3} \
		--source_prior_estimation induce \
		--calibration_batch_size {4} \
		--calibration_fraction 0.1 \
		--calibration_temperature {5} \
		--calibration_steps {6} \
		--calibration_lr {7} \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--seed 360234358 \
		--num_workers 48 \
		::: 64 512 \
		::: 1000 5000 \
		::: 1e-4 1e-3 \
		::: 64 512 \
		::: 2 \
		::: 0 5 20 100 \
		::: 1e-4 1e-3


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
		--calibration_lr 1 \
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
		--calibration_lr 1 \
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
		--calibration_lr 1 \
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
		--calibration_lr 1 \
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
		--calibration_lr 1 \
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
		--calibration_lr 1 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title '' \
		--seed 360234358 \
		--num_workers 48
