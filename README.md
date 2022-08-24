# Test time label shift adaptation

## CMNIST

Hyper-parameters

```
pipenv run python3 \
    -m tta.cli \
    --dataset_name CMNIST \
    --train_domains 0,1 \
    --train_batch_size 64 \
    --train_fraction 0.8 \
    --train_steps 1000 \
    --train_lr 1e-3 \
    --source_prior_estimation induce \
    --calibration_batch_size 64 \
    --calibration_fraction 0.1 \
    --calibration_temperature 10 \
    --calibration_steps 100 \
    --calibration_multiplier 1 \
    --test_batch_size 512 \
    --seed 360234358 \
    --num_workers 8 \
    --log_dir logs
```

Results

```
Environment 0: test accuracy 0.9025 (source), 0.5646 (independent), 0.5637 (uniform), 0.9014 (adapted)
Environment 1: test accuracy 0.8140 (source), 0.5910 (independent), 0.5895 (uniform), 0.8123 (adapted)
Environment 2: test accuracy 0.0896 (source), 0.7574 (independent), 0.7577 (uniform), 0.9098 (adapted)
```
