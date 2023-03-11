import jax.numpy as jnp


def waveFunction(x, t):
    timeIndependent = jnp.sqrt(2) / jnp.cosh(x - 1 / jnp.sqrt(2)) * jnp.exp(1j * x - 1 / jnp.sqrt(2))

    return timeIndependent


def V(x, t):
    return jnp.zeros_like(x)
