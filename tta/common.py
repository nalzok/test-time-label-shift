from typing import Tuple, Dict, Optional

import jax.numpy as jnp


Scheme = Tuple[float, bool, bool]

Curves = Dict[
    Tuple[str, Optional[Scheme], int],
    jnp.ndarray,
]

Sweeps = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
