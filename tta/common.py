from typing import Tuple, Dict, Union, Literal

import jax.numpy as jnp


AdaptationNull = Tuple[Literal["Null"]]
AdaptationOracle = Tuple[Literal["Oracle"]]
AdaptationGMTL = Tuple[Literal["GMTL"], float]
AdaptationEM = Tuple[Literal["EM"], float, bool, bool]
Adaptation = Union[AdaptationNull, AdaptationOracle, AdaptationGMTL, AdaptationEM]

Curves = Dict[
    Tuple[Adaptation, bool, int],
    jnp.ndarray,
]

Sweeps = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
