"""Fixed supervised weighting and an end-to-end energy/force consistency loss."""
import math
import jax
import jax.numpy as jnp
from .models.direct import direct_energy_and_forces


def validate_loss_weights(weights):
    if not isinstance(weights, dict) or set(weights) != {'energy', 'force'}:
        raise ValueError('Loss weights require exactly energy and force')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Loss weights must be finite nonnegative numbers')
    if sum(weights.values()) <= 0:
        raise ValueError('Loss weights cannot both be zero')
    return weights


def supervised_loss(energy, forces, batch, normalization, weights):
    weights = validate_loss_weights(weights)
    emse = jnp.mean((energy - batch['energy']) ** 2)
    fmse = jnp.mean((forces - batch['forces']) ** 2)
    e_scaled = emse / normalization.energy_loss_scale_eV ** 2
    f_scaled = fmse / normalization.force_loss_scale_eV_per_A ** 2
    return weights['energy'] * e_scaled + weights['force'] * f_scaled, {
        'energy_mse_eV2': emse, 'force_mse_eV2_per_A2': fmse,
        'normalized_energy_mse': e_scaled, 'normalized_force_mse': f_scaled,
        'weighted_energy_loss': weights['energy'] * e_scaled,
        'weighted_force_loss': weights['force'] * f_scaled,
    }


def direct_loss(params, batch, model, normalization, loss_weights):
    e, f = direct_energy_and_forces(params, batch, model=model)
    return supervised_loss(e, f, batch, normalization, loss_weights)


def energy_derived_forces(params, batch, *, model):
    def total(positions):
        e, _ = model.apply({'params': params}, batch['atomic_numbers'], positions,
                           batch['dst_idx'], batch['src_idx'], batch['batch_segments'], batch['energy'].shape[0])
        return jnp.sum(e)
    return -jax.grad(total)(batch['positions'])


def consistency_loss(params, batch, model, normalization, strength, loss_weights):
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength < 0:
        raise ValueError('Consistency strength must be finite and nonnegative')
    if strength == 0:
        return direct_loss(params, batch, model, normalization, loss_weights)
    e, f = direct_energy_and_forces(params, batch, model=model)
    supervised, metrics = supervised_loss(e, f, batch, normalization, loss_weights)
    residual = jnp.mean((f - energy_derived_forces(params, batch, model=model)) ** 2)
    scaled = residual / normalization.force_loss_scale_eV_per_A ** 2
    return supervised + strength * scaled, {**metrics, 'consistency_mse_eV2_per_A2': residual,
                                           'normalized_consistency_mse': scaled,
                                           'weighted_consistency_loss': strength * scaled}
