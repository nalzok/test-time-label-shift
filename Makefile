.PHONY: debug sweep mnist

debug:
	pipenv run python3 \
		-m tta.cli \
		--config_name debug \
		--dataset_name MNIST \
		--train_domains 5 \
		--train_apply_rotation True \
		--train_label_noise 0.1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_model LeNet \
		--train_steps 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
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
		--train_label_noise 0.1 \
		--train_batch_size {1} \
		--train_fraction 0.8 \
		--train_model LeNet \
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
		::: 2048 16384 \
		::: 1e-4 1e-3 \
		::: 64 512 \
		::: 1 \
		::: 0 10 100 \
		::: 1e-4 1e-3


mnist: mnist-default mnist-unconfounded-source mnist-no-calibration mnist-no-fixed-marginal mnist-small-batch mnist-no-rotation

mnist-default:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_label_noise 0.1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_model LeNet \
		--train_steps 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Default Setting' \
		--seed 360234358 \
		--num_workers 48

mnist-unconfounded-source:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_domains 5 \
		--train_apply_rotation True \
		--train_label_noise 0.1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_model LeNet \
		--train_steps 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Training on unconfounded source' \
		--seed 360234358 \
		--num_workers 48

mnist-no-calibration:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_label_noise 0.1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_model LeNet \
		--train_steps 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0 \
		--calibration_temperature 1 \
		--calibration_steps 0 \
		--calibration_lr 0 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Without calibration' \
		--seed 360234358 \
		--num_workers 48

mnist-no-fixed-marginal:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_label_noise 0.1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_model LeNet \
		--train_steps 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Without fixing p_t(y)' \
		--seed 360234358 \
		--num_workers 48

mnist-small-batch:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_label_noise 0.1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_model LeNet \
		--train_steps 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 64 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Using a small batch size of 8' \
		--seed 360234358 \
		--num_workers 48

mnist-no-rotation:
	pipenv run python3 \
		-m tta.cli \
		--config_name $@ \
		--dataset_name MNIST \
		--train_domains 9 \
		--train_apply_rotation True \
		--train_label_noise 0.1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_model LeNet \
		--train_steps 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Disabling rotation' \
		--seed 360234358 \
		--num_workers 48
