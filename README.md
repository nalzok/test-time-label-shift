# Beyond Invariance: Test-Time Label-Shift Adaptation for Distributions with "Spurious" Correlations

Available: https://arxiv.org/abs/2211.15646

To cite our paper

```
@misc{sun2022invariance,
      title={Beyond Invariance: Test-Time Label-Shift Adaptation for Distributions with "Spurious" Correlations}, 
      author={Qingyao Sun and Kevin Murphy and Sayna Ebrahimi and Alexander D'Amour},
      year={2022},
      eprint={2211.15646},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## How to Reproduce

1. Install [Pipenv](https://pipenv.pypa.io/en/latest/) and [Pyenv](https://github.com/pyenv/pyenv#installation).
2. Install dependence with `PIP_FIND_LINKS=https://storage.googleapis.com/jax-releases/libtpu_releases.html pipenv install --deploy`.
3. Run experiments with `make paper-mnist`, `make paper-chexpert-embedding`, `make paper-chexpert-pixel`, and `make tree`.
4. Aggregate experimental results and generate figures with `make merge`.
