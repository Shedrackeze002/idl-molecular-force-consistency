from dataclasses import dataclass
from functools import partial
import e3x
from flax import linen as nn
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64',True)


@dataclass(frozen=True)
class ModelConfig:
    features: int=16
    max_degree: int=1
    iterations: int=2
    radial_basis: int=8
    cutoff_A: float=5.0


class Backbone(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self,atomic_numbers,positions,dst_idx,src_idx):
        if not jax.config.jax_enable_x64 or positions.dtype!=jnp.float64: raise ValueError('Model requires float64')
        c=self.config
        basis=e3x.nn.basis(positions[src_idx]-positions[dst_idx],max_degree=c.max_degree,num=c.radial_basis,
                           radial_fn=e3x.nn.reciprocal_bernstein,cutoff_fn=partial(e3x.nn.smooth_cutoff,cutoff=c.cutoff_A))
        dtype={'dtype':jnp.float64,'param_dtype':jnp.float64}
        x=e3x.nn.Embed(num_embeddings=119,features=c.features,**dtype)(atomic_numbers)
        for i in range(c.iterations):
            message=e3x.nn.MessagePass(max_degree=c.max_degree,include_pseudotensors=False,**dtype,name=f'message_{i}')(x,basis,dst_idx=dst_idx,src_idx=src_idx)
            x=e3x.nn.add(x,message)
            y=e3x.nn.Dense(c.features,**dtype,name=f'dense_in_{i}')(x)
            y=e3x.nn.silu(y)
            y=e3x.nn.Dense(c.features,**dtype,name=f'dense_out_{i}')(y)
            x=e3x.nn.add(x,y)
        return x
