.PHONY: paper paper-chexpert paper-mnist paper-chexpert-embedding paper-chexpert-pixel baseline manova tree

paper: paper-mnist paper-chexpert

paper-chexpert: paper-chexpert-embedding paper-chexpert-pixel

paper-mnist:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog-$@.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name mnist_rot{1}_noise{2}_domain{3}_train{4}_cali{5}_prior{6} \
		--dataset_name MNIST \
		--dataset_apply_rotation {1} \
		--dataset_label_noise {2} \
		--train_fit_joint True \
		--train_model LeNet \
		--train_domains {3} \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs {4} \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs {5} \
		--calibration_lr 1e-3 \
		--adapt_prior_strength {6} \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 64 \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_title ColoredMNIST \
		--plot_only False \
		::: False \
		::: 0.1 \
		::: 10 9 8 7 6 5 4 3 2 1 \
		::: 500 \
		::: 100 \
		::: 1
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain9_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain8_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain7_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain6_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain5_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain4_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain3_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain2_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mnist_rotFalse_noise0.1_domain10_train500_cali100_prior1.npz \
		--target npz/mnist_rotFalse_noise0.1_domain1_train500_cali100_prior1.npz

paper-chexpert-embedding:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog-$@.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name chexpert-embedding_{1}_{2}_domain{3}_size{4}_sub{5}_tau{6}_train{7}_cali{8}_prior{9} \
		--dataset_name CheXpert \
		--dataset_Y_column {1} \
		--dataset_Z_column {2} \
		--dataset_target_domain_count 512 \
		--dataset_source_domain_count {4} \
		--dataset_subsample_what {5} \
		--dataset_use_embedding True \
		--dataset_label_noise 0 \
		--train_fit_joint True \
		--train_model Linear \
		--train_domains {3} \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs {7} \
		--train_patience 5 \
		--train_tau {6} \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs {8} \
		--calibration_patience 5 \
		--calibration_tau {6} \
		--calibration_lr 1e-3 \
		--adapt_prior_strength {9} \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 64 \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_title CheXpert-embedding \
		--plot_only False \
		::: EFFUSION \
		::: GENDER \
		::: 1 2 4 10 \
		::: 65536 16384 4096 \
		::: none groups classes \
		::: 0 1 \
		::: 5000 \
		::: 1000 \
		::: 1
	for domain in 1 2 3 4 5 6 7 8 9; do \
		for size in 65536 16384 4096; do \
			for sub in none groups classes; do \
				for tau in 0 1 ; do \
					pipenv run python3 -m scripts.superpose \
						--source npz/chexpert-embedding_EFFUSION_GENDER_domain10_size$${size}_sub$${sub}_tau$${tau}_train5000_cali1000_prior1.npz \
						--target npz/chexpert-embedding_EFFUSION_GENDER_domain$${domain}_size$${size}_sub$${sub}_tau$${tau}_train5000_cali1000_prior1.npz; \
				done \
			done \
		done \
    done

paper-chexpert-pixel:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog-$@.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name chexpert-pixel_{1}_{2}_domain{3}_train{4}_cali{5}_prior{6} \
		--dataset_name CheXpert \
		--dataset_Y_column {1} \
		--dataset_Z_column {2} \
		--dataset_target_domain_count 512 \
		--dataset_source_domain_count 85267 \
		--dataset_use_embedding False \
		--dataset_label_noise 0 \
		--train_fit_joint True \
		--train_model ResNet50 \
		--train_pretrained_path pretrained/ResNet50_ImageNet1k \
		--train_domains {3} \
		--train_fraction 0.9 \
		--train_calibration_fraction 0.1 \
		--train_batch_size 64 \
		--train_epochs {4} \
		--train_lr 1e-3 \
		--calibration_batch_size 64 \
		--calibration_epochs {5} \
		--calibration_lr 1e-3 \
		--adapt_gmtl_alpha 0.5 \
		--adapt_gmtl_alpha 1 \
		--adapt_gmtl_alpha 2 \
		--adapt_prior_strength {6} \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 64 \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_title CheXpert-pixel \
		--plot_only True \
		::: EFFUSION \
		::: GENDER \
		::: 2 10 \
		::: 100 \
		::: 20 \
		::: 1
	pipenv run python3 -m scripts.superpose \
		--source npz/chexpert-pixel_EFFUSION_GENDER_domain10_train100_cali20_prior1.npz \
		--target npz/chexpert-pixel_EFFUSION_GENDER_domain2_train100_cali20_prior1.npz

paper-mimic-embedding:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog-$@.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name mimic-embedding_{1}_{2}_domain{3}_train{4}_cali{5}_prior{6} \
		--dataset_name MIMIC \
		--dataset_Y_column {1} \
		--dataset_Z_column {2} \
		--dataset_target_domain_count 512 \
		--dataset_source_domain_count 24584 \
		--dataset_use_embedding True \
		--dataset_label_noise 0 \
		--train_fit_joint True \
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
		--adapt_gmtl_alpha 0.5 \
		--adapt_gmtl_alpha 1 \
		--adapt_gmtl_alpha 2 \
		--adapt_prior_strength {6} \
		--adapt_symmetric_dirichlet False \
		--adapt_fix_marginal False \
		--test_argmax_joint False \
		--test_batch_size 64 \
		--test_batch_size 512 \
		--seed 2022 \
		--num_workers 48 \
		--plot_title MIMIC-embedding \
		--plot_only True \
		::: Pneumonia \
		::: gender \
		::: 10 9 8 7 6 5 4 3 2 \
		::: 500 \
		::: 100 \
		::: 1
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain9_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain8_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain7_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain6_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain5_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain4_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain3_train500_cali100_prior1.npz
	pipenv run python3 -m scripts.superpose \
		--source npz/mimic-embedding_Pneumonia_gender_domain10_train500_cali100_prior1.npz \
		--target npz/mimic-embedding_Pneumonia_gender_domain2_train500_cali100_prior1.npz

data/CheXpert/data_matrix.npz:
	pipenv run python3 -m scripts.matching

baseline: data/CheXpert/data_matrix.npz
	pipenv run python3 -m scripts.baseline

manova:
	pipenv run python3 -m scripts.manova

tree:
	pipenv run python3 -m scripts.tree
