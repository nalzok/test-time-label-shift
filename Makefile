.PHONY: run

run:
	pipenv run python3 \
		-m tta.cli \
		--dataset_name CCOCO \
		--train_domains 0,1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_steps 1000 \
		--train_lr 1e-4 \
		--source_prior_estimation induce \
		--calibration_batch_size 8 \
		--calibration_fraction 0.1 \
		--calibration_temperature 10 \
		--calibration_steps 100 \
		--calibration_multiplier 1 \
		--test_batch_size 64 \
		--seed 360234358 \
		--num_workers 8 \
		--log_dir logs
