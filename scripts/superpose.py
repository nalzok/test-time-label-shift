from pathlib import Path

import numpy as np
import click


@click.command()
@click.option("--source", type=click.Path(path_type=Path), required=True)
@click.option("--target", type=click.Path(path_type=Path), required=True)
def cmd(source, target):
    all_source_sweeps = np.load(source, allow_pickle=True)
    all_target_sweeps = dict(**np.load(target, allow_pickle=True))

    for sweep_type, (source_sweeps, source_ylabel) in all_source_sweeps.items():
        target_sweeps, target_ylabel = all_target_sweeps[sweep_type]
        assert source_ylabel == target_ylabel
        for ((algo, *_), argmax_joint, batch_size), sweep in source_sweeps.items():
            if algo == "Null":
                key = ("Null-unconfounded",), argmax_joint, batch_size
                target_sweeps[key] = sweep
        all_target_sweeps[sweep_type] = (target_sweeps, target_ylabel)

    np.savez(target, **all_target_sweeps)


if __name__ == "__main__":
    cmd()
