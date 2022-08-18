.PHONY: run

run:
	pipenv run python3 -m tta.cli \
		--dataset_name CMNIST \
		--test_envs 2 \
		--train_fraction 0.8 \
		--calibration_fraction 0.1 \
		--batch_size 64 \
		--num_workers 8 \
		--train_steps 5000 \
		--lr 1e-3 \
		--temperature 1 \
		--calibration_steps 500 \
		--seed 360234358 \
		--log_dir logs
