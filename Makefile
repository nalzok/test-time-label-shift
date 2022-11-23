.PHONY: debug sweep mnist

sweep-mnist:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name mnist_rot{1}_noise{2}_domain{3}_train{4}_cali{5}_prior{6} \
		--dataset_name MNIST \
		--dataset_apply_rotation {1} \
		--dataset_label_noise {2} \
		--train_joint True \
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
		--plot_title CheXpert \
		--plot_only False \
		::: False \
		::: 0.1 \
		::: 2 10 \
		::: 500 \
		::: 50 \
		::: 0 1

sweep-chexpert:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name chexpert_{1}_{2}_domain{3}_train{4}_cali{5}_prior{6} \
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
		--plot_title CheXpert \
		--plot_only False \
		::: EFFUSION \
		::: GENDER \
		::: 2 10 \
		::: 500 \
		::: 50 \
		::: 0 1

sweep-chexpert-resnet:
	parallel \
		--eta \
		--jobs 1 \
		--joblog joblog-$@.txt \
		pipenv run python3 \
		-m tta.cli \
		--config_name rawChexpert_{1}_{2}_domain{3}_train{4}_cali{5}_prior{6} \
		--dataset_name CheXpert \
		--dataset_Y_column {1} \
		--dataset_Z_column {2} \
		--dataset_use_embedding False \
		--dataset_label_noise 0 \
		--train_joint True \
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
		--plot_title CheXpert \
		--plot_only False \
		::: EFFUSION \
		::: GENDER \
		::: 2 10 \
		::: 500 \
		::: 50 \
		::: 0 1

baseline:
	pipenv run python3 -m scripts.baseline

manova:
	pipenv run python3 -m scripts.manova

tree:
	pipenv run python3 -m scripts.tree
