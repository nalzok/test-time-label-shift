from typing import Tuple, Dict, Optional

import jax.numpy as jnp


Curves = Dict[
    Tuple[str, Optional[Tuple[float, bool, bool]], int],
    jnp.ndarray,
]

Sweeps = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
