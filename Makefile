.PHONY: debug sweep mnist

small:
	pipenv run python3 \
		-m tta.cli \
		--config_name pneumonia_small \
		--dataset_name CheXpert \
		--dataset_Y_column PNEUMONIA \
		--dataset_Z_column GENDER \
		--dataset_use_embedding True \
		--dataset_apply_rotation False \
		--dataset_label_noise 0 \
		--train_joint True \
		--train_model Linear \
		--train_domains 9 \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs 1 \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs 1 \
		--calibration_lr 1e-3 \
		--adapt_prior_strength 1 \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint True \
		--test_argmax_joint False \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48

tree:
	pipenv run python3 -m tree

paper-mnist:
	pipenv run python3 \
		-m tta.cli \
		--config_name mnist_rotated_0.1_18 \
		--dataset_name MNIST \
		--dataset_apply_rotation False \
		--dataset_label_noise 0.1 \
		--train_joint True \
		--train_model LeNet \
		--train_domains 18 \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs 10 \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs 2 \
		--calibration_lr 1e-3 \
		--adapt_prior_strength 1 \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_only False
	pipenv run python3 \
		-m tta.cli \
		--config_name mnist_rotated_0.1_18_more \
		--dataset_name MNIST \
		--dataset_apply_rotation False \
		--dataset_label_noise 0.1 \
		--train_joint True \
		--train_model LeNet \
		--train_domains 18 \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs 10 \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs 2 \
		--calibration_lr 1e-3 \
		--adapt_gmtl_alpha -1 \
		--adapt_gmtl_alpha 0.5 \
		--adapt_gmtl_alpha 2 \
		--adapt_prior_strength 1 \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_only False

debug:
	pipenv run python3 \
		-m tta.cli \
		--config_name mimic_pneumonia_edema_100_20_domain2 \
		--dataset_name MIMIC \
		--dataset_Y_column Pneumonia \
		--dataset_Z_column Edema \
		--dataset_use_embedding True \
		--dataset_label_noise 0 \
		--train_joint True \
		--train_model Linear \
		--train_domains 2 \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs 100 \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs 20 \
		--calibration_lr 1e-3 \
		--adapt_gmtl_alpha -1 \
		--adapt_gmtl_alpha 0.5 \
		--adapt_gmtl_alpha 2 \
		--adapt_prior_strength 1 \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_only True
	pipenv run python3 \
		-m tta.cli \
		--config_name mimic_effusion_edema_100_20_domain2 \
		--dataset_name MIMIC \
		--dataset_Y_column "Pleural Effusion" \
		--dataset_Z_column Edema \
		--dataset_use_embedding True \
		--dataset_label_noise 0 \
		--train_joint True \
		--train_model Linear \
		--train_domains 2 \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs 100 \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs 20 \
		--calibration_lr 1e-3 \
		--adapt_gmtl_alpha -1 \
		--adapt_gmtl_alpha 0.5 \
		--adapt_gmtl_alpha 2 \
		--adapt_prior_strength 1 \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_only True

sweep-chexpert:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name chexpert_{1}_{2}_domain{3}_train{4}_cali{5} \
		--dataset_name CheXpert \
		--dataset_Y_column {1} \
		--dataset_Z_column {2} \
		--dataset_use_embedding True \
		--dataset_label_noise 0 \
		--train_joint True \
		--train_model Linear \
		--train_domains {3} \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs {4} \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs {5} \
		--calibration_lr 1e-3 \
		--adapt_gmtl_alpha -1 \
		--adapt_gmtl_alpha 0.5 \
		--adapt_gmtl_alpha 2 \
		--adapt_prior_strength 0 \
		--adapt_prior_strength 1 \
		--adapt_prior_strength 5 \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_only False \
		::: PNEUMONIA EFFUSION \
		::: GENDER \
		::: 1 2 4 10 16 18 19 \
		::: 100 250 \
		::: 20 50

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
		--train_pretrained_path pretrained/ResNet50_ImageNet1k \
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
