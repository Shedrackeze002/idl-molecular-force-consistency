"""Shared E3x backbone with scalar energy and polar degree-one force heads."""
from flax import linen as nn
import e3x
import jax
import jax.numpy as jnp

from .backbone import Backbone, ModelConfig


class DualHeadModel(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        if self.config.max_degree < 1:
            raise ValueError('Direct force head requires degree one or higher')
        if not e3x.Config.cartesian_order:
            raise ValueError('Direct force head requires Cartesian degree-one basis ordering')
        x = Backbone(self.config)(atomic_numbers, positions, dst_idx, src_idx)
        scalar = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
        if self.config.energy_readout_init not in ('zero', 'lecun_normal'):
            raise ValueError('Unknown energy readout initialization')
        kwargs = {'kernel_init': jax.nn.initializers.zeros} if self.config.energy_readout_init == 'zero' else {}
        atom_energy = e3x.nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64,
                                   name='energy_head', **kwargs)(scalar).reshape(-1)
        polar = e3x.nn.change_max_degree_or_type(x, max_degree=1, include_pseudotensors=False)
        readout = e3x.nn.Dense(1, use_bias=False, dtype=jnp.float64, param_dtype=jnp.float64,
                               kernel_init=jax.nn.initializers.zeros, name='force_head')(polar)
        self.sow('intermediates', 'force_readout', readout)
        # Locked E3x Cartesian order has natural-parity l=1 channels [x,y,z].
        forces = readout[:, 0, 1:4, 0]
        return jax.ops.segment_sum(atom_energy, batch_segments, num_segments=batch_size), forces


def direct_energy_and_forces(params, batch, *, model):
    if any(v.dtype != jnp.float64 for v in jax.tree.leaves(params)):
        raise ValueError('Parameters must be float64')
    return model.apply({'params': params}, batch['atomic_numbers'], batch['positions'],
                       batch['dst_idx'], batch['src_idx'], batch['batch_segments'], batch['energy'].shape[0])
