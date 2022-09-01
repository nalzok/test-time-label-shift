# Test time label shift adaptation

## MNIST

Hyper-parameters

```bash
pipenv run python3 \
    -m tta.cli \
    --dataset_name MNIST \
    --train_domains 0 \
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
    --test_fix_marginal True \
    --seed 360234358 \
    --num_workers 8 \
    --log_dir logs
```

Results

```
===> Adapting & Evaluating
---> Environment 0 (seen)
Test accuracy 0.9022927284240723 (source), 0.8545104265213013 (adapted)
Test norm 0.0003403940354473889 (source), 0.0005156156257726252 (adapted)
---> Environment 1 (unseen)
Test accuracy 0.7885151505470276 (source), 0.7443754076957703 (adapted)
Test norm 0.00019217537192162126 (source), 0.0005369287682697177 (adapted)
---> Environment 2 (unseen)
Test accuracy 0.09942147135734558 (source), 0.8902935981750488 (adapted)
Test norm 0.0008778728661127388 (source), 0.00040434510447084904 (adapted)
```
