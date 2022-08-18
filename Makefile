.PHONY: run

run:
	pipenv run python3 -m tta.cli \
		--dataset_name CMNIST \
		--train_domains 0,1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_steps 5000 \
		--train_lr 1e-3 \
		--calibration_fraction 0.1 \
		--calibration_temperature 1 \
		--calibration_steps 500 \
		--test_batch_size 256 \
		--seed 360234358 \
		--num_workers 8 \
		--log_dir logs
