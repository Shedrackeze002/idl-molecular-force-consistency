import numpy as np
from flax import linen as nn
import e3x
import jax
import jax.numpy as jnp

from .backbone import Backbone,ModelConfig


class EnergyModel(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self,atomic_numbers,positions,dst_idx,src_idx,batch_segments,batch_size):
        x=Backbone(self.config)(atomic_numbers,positions,dst_idx,src_idx)
        self.sow('intermediates','features',x)
        # Natural parity: degree zero is even; degree one is a polar vector.
        scalar=e3x.nn.change_max_degree_or_type(x,max_degree=0,include_pseudotensors=False)
        if self.config.energy_readout_init not in ('zero','lecun_normal'): raise ValueError('Unknown energy readout initialization')
        kwargs={'kernel_init':jax.nn.initializers.zeros} if self.config.energy_readout_init=='zero' else {}
        atom_energy=e3x.nn.Dense(1,dtype=jnp.float64,param_dtype=jnp.float64,name='energy_head',**kwargs)(scalar).reshape(-1)
        return jax.ops.segment_sum(atom_energy,batch_segments,num_segments=batch_size)


def make_batch(positions,atomic_numbers,energy=None,forces=None):
    positions=np.asarray(positions);z=np.asarray(atomic_numbers)
    if positions.dtype!=np.float64: raise ValueError('Coordinates must be float64')
    if positions.ndim!=3 or positions.shape[-1]!=3 or z.shape!=positions.shape[:2]: raise ValueError('Invalid batch shape')
    if not np.isfinite(positions).all(): raise ValueError('Nonfinite coordinates')
    b,n=positions.shape[:2];dst,src=np.where(~np.eye(n,dtype=bool))
    offsets=np.arange(b)[:,None]*n
    return {'positions':jnp.asarray(positions.reshape(-1,3)), 'atomic_numbers':jnp.asarray(z.reshape(-1),dtype=jnp.int64),
            'dst_idx':jnp.asarray((dst[None]+offsets).reshape(-1)), 'src_idx':jnp.asarray((src[None]+offsets).reshape(-1)),
            'batch_segments':jnp.repeat(jnp.arange(b),n),'energy':jnp.asarray(np.zeros(b) if energy is None else energy,dtype=jnp.float64),
            'forces':jnp.asarray(np.zeros_like(positions).reshape(-1,3) if forces is None else np.asarray(forces).reshape(-1,3),dtype=jnp.float64)}


def energy_and_forces(params,batch,*,model):
    if any(v.dtype!=jnp.float64 for v in jax.tree.leaves(params)): raise ValueError('Parameters must be float64')
    def energies(p):
        return model.apply({'params':params},batch['atomic_numbers'],p,batch['dst_idx'],batch['src_idx'],batch['batch_segments'],batch['energy'].shape[0])
    def total(p):
        e=energies(p);return jnp.sum(e),e
    (_,e),gradient=jax.value_and_grad(total,has_aux=True)(batch['positions'])
    return e,-gradient
