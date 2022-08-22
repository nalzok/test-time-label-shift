.PHONY: run

run:
	parallel \
		--eta \
		--jobs 1 \
		pipenv run python3 \
		-m tta.cli \
		--dataset_name CMNIST \
		--train_domains 0,1 \
		--train_batch_size 64 \
		--train_fraction 0.8 \
		--train_steps {1} \
		--train_lr {2} \
		--source_prior_estimation induce \
		--calibration_fraction 0.1 \
		--calibration_temperature {3} \
		--calibration_steps {4} \
		--calibration_multiplier {5} \
		--test_batch_size 256 \
		--seed 360234358 \
		--num_workers 8 \
		--log_dir logs \
		'|' tee logs/steps{1}_lr{2}_temp{3}_cali{4}_multi{5}.txt \
		::: 10 100 1000 10000 \
		::: 1e-2 1e-3 1e-4 1e-5 \
		::: 0.1 1 10 \
		::: 10 100 1000 \
		::: 0.1 1
