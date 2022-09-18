.PHONY: debug sweep mnist

debug:
	pipenv run python3 \
		-m tta.cli \
		--config_name debug \
		--dataset_name Waterbirds \
		--dataset_apply_rotation False \
		--dataset_label_noise 0 \
		--train_model ResNet18 \
		--train_domains 0 \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.0 \
		--train_batch_size 512 \
		--train_steps 2000 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_temperature 1 \
		--calibration_domains 1 \
		--calibration_fraction 0.8 \
		--calibration_batch_size 512 \
		--calibration_steps 100 \
		--calibration_lr 1e-3 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--test_batch_size 512 \
		--seed 360234358 \
		--num_workers 48

sweep:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name sweep_{1}_{2}_Tbatch{3}_Tsteps{4}_Tlr{5}_temp{6}_Cbatch{7}_Csteps{8}_Clr{9} \
		--dataset_name {1} \
		--dataset_apply_rotation False \
		--dataset_label_noise 0 \
		--train_model {2} \
		--train_domains 0 \
		--train_fraction 1.0 \
		--train_calibration_fraction 0.0 \
		--train_batch_size {3} \
		--train_steps {4} \
		--train_lr {5} \
		--source_prior_estimation induce \
		--calibration_temperature {6} \
		--calibration_domains 1 \
		--calibration_fraction 1.0 \
		--calibration_batch_size {7} \
		--calibration_steps {8} \
		--calibration_lr {9} \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--test_batch_size 512 \
		--seed 360234358 \
		--num_workers 48 \
		::: Waterbirds \
		::: ResNet18 \
		::: 512 \
		::: 50000 \
		::: 1e-3 \
		::: 1 \
		::: 512 \
		::: 0 \
		::: 0


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
