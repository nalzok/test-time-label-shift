.PHONY: paper paper-chexpert paper-mnist paper-chexpert-embedding paper-chexpert-pixel baseline manova tree merge


paper: paper-mnist paper-chexpert


paper-chexpert: paper-chexpert-embedding paper-chexpert-pixel


paper-mnist:
	for seed in $$(seq 2022 2025); do \
		for rot in False; do \
			for noise in 0; do \
				for domain in 1; do \
					for sub in none groups; do \
						for tau in 0 1; do \
							for train in 5000; do \
								for cali in 0 1000; do \
									for prior in 1; do \
										pipenv run python3 \
											-m tta.cli \
											--config_name mnist_rot$${rot}_noise$${noise}_domain$${domain}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior}_seed$${seed} \
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
											--adapt_gmtl_alpha 0.5 \
											--adapt_gmtl_alpha 1 \
											--adapt_prior_strength $${prior} \
											--adapt_symmetric_dirichlet False \
											--adapt_fix_marginal False \
											--test_argmax_joint False \
											--test_batch_size 64 \
											--test_batch_size 512 \
											--seed $${seed} \
											--num_workers 48 \
											--plot_title "" \
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


paper-chexpert-embedding:
	for seed in $$(seq 2022 2025); do \
		for Y_column in EFFUSION; do \
			for Z_column in GENDER; do \
				for domain in 1; do \
					for size in 65536; do \
						for sub in none groups; do \
							for tau in 0 1; do \
								for train in 5000; do \
									for cali in 0 1000; do \
										for prior in 1; do \
											pipenv run python3 \
												-m tta.cli \
												--config_name chexpert-embedding_$${Y_column}_$${Z_column}_domain$${domain}_size$${size}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior}_seed$${seed} \
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
												--adapt_gmtl_alpha 0.5 \
												--adapt_gmtl_alpha 1 \
												--adapt_prior_strength $${prior} \
												--adapt_symmetric_dirichlet False \
												--adapt_fix_marginal False \
												--test_argmax_joint False \
												--test_batch_size 64 \
												--test_batch_size 512 \
												--seed $${seed} \
												--num_workers 48 \
												--plot_title "" \
												--plot_only False; \
										done \
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
	for seed in $$(seq 2022 2025); do \
		for Y_column in EFFUSION; do \
			for Z_column in GENDER; do \
				for domain in 1; do \
					for size in 65536; do \
						for sub in none groups; do \
							for tau in 0 1; do \
								for train in 5000; do \
									for cali in 0 1000; do \
										for prior in 1; do \
											pipenv run python3 \
												-m tta.cli \
												--config_name chexpert-pixel_$${Y_column}_$${Z_column}_domain$${domain}_size$${size}_sub$${sub}_tau$${tau}_train$${train}_cali$${cali}_prior$${prior}_seed$${seed} \
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
												--adapt_gmtl_alpha 0.5 \
												--adapt_gmtl_alpha 1 \
												--adapt_prior_strength $${prior} \
												--adapt_symmetric_dirichlet False \
												--adapt_fix_marginal False \
												--test_argmax_joint False \
												--test_batch_size 64 \
												--test_batch_size 512 \
												--seed $${seed} \
												--num_workers 48 \
												--plot_title "" \
												--plot_only False; \
										done \
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
	env JAX_PLATFORMS="cpu" pipenv run python3 -m scripts.tree


merge:
	for noise in 0; do \
		for domain in 1; do \
			for cali in 0 1000; do \
				env JAX_PLATFORMS="cpu" \
					pipenv run python3 \
						-m scripts.merge \
						--npz_pattern "mnist_rotFalse_noise$${noise}_domain$${domain}_sub*_tau*_train5000_cali$${cali}_prior1_seed????.npz" \
						--merged_title "" \
						--merged_name "mnist-domain$${domain}-noise$${noise}-cali$${cali}"; \
			done \
		done \
	done
	# for domain in 1; do \
	# 	for cali in 1000; do \
	# 		env JAX_PLATFORMS="cpu" \
	# 			pipenv run python3 \
	# 				-m scripts.merge \
	# 				--npz_pattern "chexpert-embedding_EFFUSION_GENDER_domain$${domain}_size65536_sub*_tau*_train5000_cali$${cali}_prior1_seed????.npz" \
	# 				--merged_title "" \
	# 				--merged_name "chexpert-embedding-domain$${domain}-cali$${cali}"; \
	# 	done \
	# done
	# for domain in 1; do \
	# 	for cali in 0 1000; do \
	# 		env JAX_PLATFORMS="cpu" \
	# 			pipenv run python3 \
	# 				-m scripts.merge \
	# 				--npz_pattern "chexpert-pixel_EFFUSION_GENDER_domain$${domain}_size65536_sub*_tau*_train5000_cali$${cali}_prior1_seed????.npz" \
	# 				--merged_title "" \
	# 				--merged_name "chexpert-pixel-domain$${domain}-cali$${cali}"; \
	# 	done \
	# done
