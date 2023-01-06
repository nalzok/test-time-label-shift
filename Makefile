.PHONY: paper paper-chexpert paper-mnist paper-chexpert-embedding paper-chexpert-pixel baseline manova tree

paper: paper-mnist paper-chexpert


paper-chexpert: paper-chexpert-embedding paper-chexpert-pixel


paper-mnist:
	for rot in False; do \
		for noise in 0.025 0.1 0.4; do \
			for domain in 10 4 2 1; do \
				for sub in none groups classes; do \
					for tau in 0 1; do \
						for train in 5000; do \
							for cali in 1000; do \
								for prior in 1; do \
									pipenv run python3 \
										-m tta.cli \
										--config_name mnist_rot$${rot}_noise$${noise}_domain$${domain}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior} \
										--dataset_name MNIST \
										--dataset_apply_rotation $${rot} \
										--dataset_subsample_what $${sub} \
										--dataset_label_noise $${noise} \
										--train_fit_joint True \
										--train_model LeNet \
										--train_domains $${domain} \
										--train_fraction 0.9 \
										--train_calibration_fraction 0.1 \
										--train_batch_size 64 \
										--train_epochs $${train} \
										--train_patience 5 \
										--train_tau $${tau} \
										--train_lr 1e-3 \
										--calibration_batch_size 64 \
										--calibration_epochs $${cali} \
										--calibration_patience 5 \
										--calibration_tau $${tau} \
										--calibration_lr 1e-3 \
										--adapt_prior_strength $${prior} \
										--adapt_symmetric_dirichlet False \
										--adapt_fix_marginal False \
										--test_argmax_joint False \
										--test_batch_size 64 \
										--test_batch_size 512 \
										--seed 2022 \
										--num_workers 48 \
										--plot_title ColoredMNIST \
										--plot_only True; \
								pipenv run python3 -m scripts.superpose \
									--source npz/mnist_rot$${rot}_noise$${noise}_domain10_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior}.npz \
									--target npz/mnist_rot$${rot}_noise$${noise}_domain$${domain}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior}.npz; \
								done \
							done \
						done \
					done \
				done \
			done \
		done \
    done


paper-chexpert-embedding:
	for Y_column in EFFUSION; do \
		for Z_column in GENDER; do \
			for domain in 10 4 2 1; do \
				for size in 65536 16384 4096; do \
					for sub in none groups classes; do \
						for tau in 0 1; do \
							for train in 5000; do \
								for cali in 1000; do \
									for prior in 1; do \
										pipenv run python3 \
											-m tta.cli \
											--config_name chexpert-embedding_$${Y_column}_$${Z_column}_domain$${domain}_size$${size}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior} \
											--dataset_name CheXpert \
											--dataset_Y_column $${Y_column} \
											--dataset_Z_column $${Z_column} \
											--dataset_target_domain_count 512 \
											--dataset_source_domain_count $${size} \
											--dataset_subsample_what $${sub} \
											--dataset_use_embedding True \
											--dataset_label_noise 0 \
											--train_fit_joint True \
											--train_model Linear \
											--train_domains $${domain} \
											--train_fraction 0.9 \
											--train_calibration_fraction 0.1 \
											--train_batch_size 64 \
											--train_epochs $${train} \
											--train_patience 5 \
											--train_tau $${tau} \
											--train_lr 1e-3 \
											--calibration_batch_size 64 \
											--calibration_epochs $${cali} \
											--calibration_patience 5 \
											--calibration_tau $${tau} \
											--calibration_lr 1e-3 \
											--adapt_prior_strength $${prior} \
											--adapt_symmetric_dirichlet False \
											--adapt_fix_marginal False \
											--test_argmax_joint False \
											--test_batch_size 64 \
											--test_batch_size 512 \
											--seed 2022 \
											--num_workers 48 \
											--plot_title CheXpert-embedding \
											--plot_only True; \
										pipenv run python3 -m scripts.superpose \
											--source npz/chexpert-embedding_EFFUSION_GENDER_domain10_size$${size}_sub$${sub}_tau$${tau}_train5000_cali1000_prior1.npz \
											--target npz/chexpert-embedding_EFFUSION_GENDER_domain$${domain}_size$${size}_sub$${sub}_tau$${tau}_train5000_cali1000_prior1.npz; \
									done \
								done \
							done \
						done \
					done \
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
