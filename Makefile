.PHONY: paper paper-chexpert paper-mnist paper-chexpert-embedding paper-chexpert-pixel baseline manova tree merge

paper: paper-mnist paper-chexpert


paper-chexpert: paper-chexpert-embedding paper-chexpert-pixel


paper-mnist:
	for rot in False; do \
		for noise in 0; do \
			for domain in 1 2 4 10; do \
				for sub in none groups classes; do \
					for tau in 0 1; do \
						for train in 5000; do \
							for cali in 0 1000; do \
								for prior in 0; do \
									pipenv run python3 \
										-m tta.cli \
										--config_name mnist_rot$${rot}_noise$${noise}_domain$${domain}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior} \
										--dataset_name MNIST \
										--dataset_apply_rotation $${rot} \
										--dataset_subsample_what $${sub} \
										--dataset_feature_noise $${noise} \
										--dataset_label_noise 0 \
										--train_fit_joint True \
										--train_model LeNet \
										--train_domains $${domain} \
										--train_fraction 0.9 \
										--train_calibration_fraction 0.1 \
										--train_batch_size 64 \
										--train_epochs $${train} \
										--train_decay 0.1 \
										--train_patience 5 \
										--train_tau $${tau} \
										--train_lr 1e-3 \
										--calibration_batch_size 64 \
										--calibration_epochs $${cali} \
										--calibration_decay 0.1 \
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
										--plot_only False; \
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
			for domain in 1 2 4 10; do \
				for size in 65536; do \
					for sub in none groups classes; do \
						for tau in 0 1; do \
							for train in 5000; do \
								for cali in 0 1000; do \
									for prior in 0; do \
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
											--dataset_feature_noise 0 \
											--dataset_label_noise 0 \
											--train_fit_joint True \
											--train_model Linear \
											--train_domains $${domain} \
											--train_fraction 0.9 \
											--train_calibration_fraction 0.1 \
											--train_batch_size 64 \
											--train_epochs $${train} \
											--train_decay 0.1 \
											--train_patience 5 \
											--train_tau $${tau} \
											--train_lr 1e-3 \
											--calibration_batch_size 64 \
											--calibration_epochs $${cali} \
											--calibration_decay 0.1 \
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
											--plot_title "" \
											--plot_only True; \
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
	for Y_column in EFFUSION; do \
		for Z_column in GENDER; do \
			for domain in 1 2 4 10; do \
				for size in 65536; do \
					for sub in none groups classes; do \
						for tau in 0 1; do \
							for train in 5000; do \
								for cali in 0 1000; do \
									for prior in 0; do \
										pipenv run python3 \
											-m tta.cli \
											--config_name chexpert-pixel_$${Y_column}_$${Z_column}_domain$${domain}_size$${size}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior} \
											--dataset_name CheXpert \
											--dataset_Y_column $${Y_column} \
											--dataset_Z_column $${Z_column} \
											--dataset_target_domain_count 512 \
											--dataset_source_domain_count $${size} \
											--dataset_subsample_what $${sub} \
											--dataset_use_embedding False \
											--dataset_feature_noise 0 \
											--dataset_label_noise 0 \
											--train_fit_joint True \
											--train_model ResNet50 \
											--train_pretrained_path pretrained/ResNet50_ImageNet1k \
											--train_domains $${domain} \
											--train_fraction 0.9 \
											--train_calibration_fraction 0.1 \
											--train_batch_size 64 \
											--train_epochs $${train} \
											--train_decay 0.1 \
											--train_patience 5 \
											--train_tau $${tau} \
											--train_lr 1e-3 \
											--calibration_batch_size 64 \
											--calibration_epochs $${cali} \
											--calibration_decay 0.1 \
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
											--plot_title CheXpert-pixel \
											--plot_only False; \
									done \
								done \
							done \
						done \
					done \
				done \
			done \
		done \
    done


paper-mimic-embedding:
	for Y_column in Pneumonia; do \
		for Z_column in gender; do \
			for domain in 1 2 4 10; do \
				for size in 23290; do \
					for sub in none groups classes; do \
						for tau in 0 1; do \
							for train in 5000; do \
								for cali in 0 1000; do \
									for prior in 0; do \
										pipenv run python3 \
											-m tta.cli \
											--config_name mimic-embedding_$${Y_column}_$${Z_column}_domain$${domain}_size$${size}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior} \
											--dataset_name MIMIC \
											--dataset_Y_column $${Y_column} \
											--dataset_Z_column $${Z_column} \
											--dataset_target_domain_count 512 \
											--dataset_source_domain_count $${size} \
											--dataset_subsample_what $${sub} \
											--dataset_use_embedding True \
											--dataset_feature_noise 0 \
											--dataset_label_noise 0 \
											--train_fit_joint True \
											--train_model Linear \
											--train_domains $${domain} \
											--train_fraction 0.9 \
											--train_calibration_fraction 0.1 \
											--train_batch_size 64 \
											--train_epochs $${train} \
											--train_decay 0.1 \
											--train_patience 5 \
											--train_tau $${tau} \
											--train_lr 1e-3 \
											--calibration_batch_size 64 \
											--calibration_epochs $${cali} \
											--calibration_decay 0.1 \
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
											--plot_title MIMIC-embedding \
											--plot_only False; \
									done \
								done \
							done \
						done \
					done \
				done \
			done \
		done \
    done


data/CheXpert/data_matrix.npz:
	pipenv run python3 -m scripts.matching


baseline: data/CheXpert/data_matrix.npz
	pipenv run python3 -m scripts.baseline


manova:
	pipenv run python3 -m scripts.manova


tree:
	pipenv run python3 -m scripts.tree


merge:
	for noise in 0 0.5 1; do \
		for domain in 1; do \
			for cali in 0 1000; do \
				env JAX_PLATFORMS="" \
					pipenv run python3 \
						-m scripts.merge \
						--npz_pattern "mnist_rotFalse_noise$${noise}_domain$${domain}_*train5000_cali$${cali}_prior0.npz" \
						--merged_title "" \
						--merged_name "mnist-domain$${domain}-noise$${noise}-cali$${cali}"; \
			done \
		done \
	done
	for domain in 1; do \
		for cali in 0 1000; do \
			env JAX_PLATFORMS="" \
				pipenv run python3 \
					-m scripts.merge \
					--npz_pattern "chexpert-embedding_EFFUSION_GENDER_domain$${domain}_size65536_*train5000_cali$${cali}_prior0.npz" \
					--merged_title "" \
					--merged_name "chexpert-embedding-domain$${domain}-cali$${cali}"; \
		done \
    done
	for domain in 1; do \
		for cali in 0 1000; do \
			env JAX_PLATFORMS="" \
				pipenv run python3 \
					-m scripts.merge \
					--npz_pattern "chexpert-pixel_EFFUSION_GENDER_domain$${domain}_size65536_*train5000_cali$${cali}_prior0.npz" \
					--merged_title "" \
					--merged_name "chexpert-pixel-domain$${domain}-cali$${cali}"; \
		done \
    done
	for domain in 1; do \
		for cali in 0 1000; do \
			env JAX_PLATFORMS="" \
				pipenv run python3 \
					-m scripts.merge \
					--npz_pattern "mimic-embedding_Pneumonia_gender_domain$${domain}_size23290_*train5000_cali$${cali}_prior0.npz" \
					--merged_title "" \
					--merged_name "mimic-embedding-domain$${domain}-cali$${cali}"; \
		done \
    done
