.PHONY: debug sweep mnist

debug:
	pipenv run python3 \
		-m tta.cli \
		--config_name debug \
		--dataset_name CheXpert \
		--dataset_use_embedding True \
		--dataset_apply_rotation False \
		--dataset_label_noise 0 \
		--train_model Linear \
		--train_domains 9 \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs 50 \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs 10 \
		--calibration_lr 1e-3 \
		--test_prior_strength 1 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_fix_marginal True \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48

sweep:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name sweep_{1}_{2}_Tbatch{3}_Tepochs{4}_Tlr{5}_temp{6}_Cbatch{7}_Cepochs{8}_Clr{9}_seed{10} \
		--dataset_name {1} \
		--dataset_apply_rotation False \
		--dataset_label_noise 0 \
		--train_model {2} \
		--train_checkpoint_path pretrained/ResNet50_ImageNet1k \
		--train_domains 0 \
		--train_fraction 1.0 \
		--train_calibration_fraction 0.1 \
		--train_batch_size {3} \
		--train_epochs {4} \
		--train_lr {5} \
		--source_prior_estimation induce \
		--calibration_temperature {6} \
		--calibration_batch_size {7} \
		--calibration_epochs {8} \
		--calibration_lr {9} \
		--test_batch_size 512 \
		--seed {10} \
		--num_workers 48 \
		::: Waterbirds \
		::: ResNet50 \
		::: 512 \
		::: 100 200 \
		::: 1e-3 \
		::: 1 \
		::: 512 \
		::: 0 \
		::: 0 \
		::: 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031


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
		--train_epochs 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_epochs 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Default Setting' \
		--seed 2022 \
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
		--train_epochs 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_epochs 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Training on unconfounded source' \
		--seed 2022 \
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
		--train_epochs 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0 \
		--calibration_temperature 1 \
		--calibration_epochs 0 \
		--calibration_lr 0 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Without calibration' \
		--seed 2022 \
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
		--train_epochs 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_epochs 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Without fixing p_t(y)' \
		--seed 2022 \
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
		--train_epochs 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_epochs 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 64 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Using a small batch size of 8' \
		--seed 2022 \
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
		--train_epochs 16384 \
		--train_lr 1e-3 \
		--source_prior_estimation induce \
		--calibration_batch_size 512 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_epochs 100 \
		--calibration_lr 1e-3 \
		--test_batch_size 512 \
		--test_symmetric_dirichlet False \
		--test_symmetric_dirichlet True \
		--test_prior_strength 1 \
		--test_prior_strength 4 \
		--test_fix_marginal True \
		--plot_title 'Disabling rotation' \
		--seed 2022 \
		--num_workers 48
