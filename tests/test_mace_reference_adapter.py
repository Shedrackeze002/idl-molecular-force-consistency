"""Calculator bridge contract tests; synthetic ASE calculations only."""
import copy
import unittest

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from evaluation.adapter import (ArmConfigurationError, ForceShapeMismatch,
                                MalformedForceReturn, NonfiniteValueError)
from evaluation.relax import relax


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, *, offset=7., force_result=None, energy_result=None,
                 error=None):
        super().__init__()
        self.offset = offset
        self.force_result = force_result
        self.energy_result = energy_result
        self.error = error
        self.calls = []
        self.requests = []

    def get_property(self, name, atoms=None, allow_calculation=True):
        self.requests.append(name)
        return super().get_property(name, atoms, allow_calculation)

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls.append(atoms.copy())
        if self.error is not None:
            raise self.error
        p = atoms.get_positions()
        self.results = {
            'energy': self.offset + np.sum(p*p) if self.energy_result is None
                      else self.energy_result,
            'forces': -2*p if self.force_result is None else self.force_result,
        }


class CalculatorBridgeTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[1., 2., 3.], [4., 5., 6.]])
        self.identity = {'role': 'synthetic-only', 'nested': {'tags': ['original']}}
        self.calc = Harmonic()

    def arm(self, calc=None, **kwargs):
        from mace_reference_adapter import calculator_arm
        return calculator_arm(self.calc if calc is None else calc,
                              kwargs.pop('species', ('O', 'H')),
                              arm_id=kwargs.pop('arm_id', 'synthetic-harmonic'),
                              checkpoint_identity=kwargs.pop('identity', self.identity),
                              **kwargs)

    def test_shape_units_atom_order_and_no_energy_accessor(self):
        before = self.positions.copy()
        arm = self.arm()
        forces, info = arm['forces'](self.positions, ('O', 'H'), 'fixture/1')
        np.testing.assert_array_equal(forces, [[-2., -4., -6.], [-8., -10., -12.]])
        np.testing.assert_array_equal(self.positions, before)
        self.assertEqual(forces.dtype, np.dtype('float64'))
        self.assertEqual(self.calc.requests, ['forces'])
        np.testing.assert_array_equal(self.calc.calls[0].numbers, [8, 1])
        self.assertFalse(self.calc.calls[0].pbc.any())
        self.assertEqual(info['record_id'], 'fixture/1')
        self.assertEqual(info['species'], ('O', 'H'))
        self.assertTrue(info['atom_order_preserved'])
        self.assertEqual(info['shape'], (2, 3))
        self.assertEqual(info['units'], {'positions': 'angstrom',
                         'forces': 'eV/angstrom', 'energy': 'eV'})

    def test_batch_order_shape_dtype_and_fresh_structures(self):
        p = np.stack([self.positions, self.positions + 1])
        f, info = self.arm()['forces'](p)
        np.testing.assert_array_equal(f, -2*p)
        self.assertEqual(f.dtype, np.dtype('float64'))
        self.assertEqual(info['shape'], (2, 2, 3))
        self.assertEqual(len(self.calc.calls), 2)
        for actual, expected in zip(self.calc.calls, p):
            np.testing.assert_array_equal(actual.positions, expected)

    def test_energy_preserves_constant_gauge_and_is_scalar(self):
        arm = self.arm()
        energy, info = arm['energy'](self.positions, record_id='fixture/energy')
        self.assertEqual(energy, 98.)  # 7 + (1+4+9+16+25+36)
        self.assertIsInstance(energy, float)
        self.assertEqual(self.calc.requests, ['energy'])
        self.assertEqual(info['record_id'], 'fixture/energy')
        self.assertTrue(arm['provides_energy'])

    def test_nested_identity_cannot_be_mutated_by_caller_or_outputs(self):
        original = copy.deepcopy(self.identity)
        arm = self.arm()
        self.identity['nested']['tags'][0] = 'caller mutation'
        arm['checkpoint_identity']['nested']['tags'][0] = 'arm mutation'
        _, info = arm['forces'](self.positions)
        self.assertEqual(info['checkpoint_identity'], original)
        self.assertFalse(info['checkpoint_identity_verified'])
        info['checkpoint_identity']['nested']['tags'][0] = 'info mutation'
        _, next_info = arm['energy'](self.positions)
        self.assertEqual(next_info['checkpoint_identity'], original)
        self.assertFalse(arm['checkpoint_identity_verified'])

    def test_invalid_whole_batch_and_species_refused_before_calculator(self):
        arm = self.arm()
        cases = [np.ones((2, 2)), np.ones((3, 3)), np.ones(6),
                 np.empty((0, 2, 3)), np.full((2, 3), np.nan),
                 np.stack([self.positions, np.full((2, 3), np.inf)])]
        for p in cases:
            with self.subTest(shape=p.shape), self.assertRaises(ArmConfigurationError):
                arm['forces'](p)
        with self.assertRaises(ArmConfigurationError):
            arm['forces'](self.positions, ('H', 'O'))
        self.assertEqual(self.calc.requests, [])

    def test_all_batched_energy_refused_before_calculator(self):
        arm = self.arm()
        for p in (self.positions[None], np.stack([self.positions]*2)):
            with self.assertRaises(ArmConfigurationError):
                arm['energy'](p)
        self.assertEqual(self.calc.requests, [])

    def test_invalid_constructor_contract_refused_without_calculation(self):
        for kwargs in ({'species': ()}, {'species': ('NotElement',)},
                       {'species': ('X',)},
                       {'arm_id': ''}, {'identity': {}},
                       {'identity': {'bad': float('nan')}},
                       {'identity': {'bad': object()}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ArmConfigurationError):
                self.arm(**kwargs)
        force_only = Harmonic()
        force_only.implemented_properties = ['forces']
        with self.assertRaises(ArmConfigurationError):
            self.arm(force_only)
        self.assertEqual(self.calc.requests, [])

    def test_bad_force_returns_use_declared_refusals(self):
        for value, error in [(np.array(['bad']), MalformedForceReturn),
                             (np.ones((1, 3)), ForceShapeMismatch),
                             (np.full((2, 3), np.nan), NonfiniteValueError)]:
            with self.subTest(error=error.__name__):
                calc = Harmonic(force_result=value)
                with self.assertRaises(error) as caught:
                    self.arm(calc)['forces'](self.positions, record_id='bad/force')
                self.assertEqual(calc.requests, ['forces'])
                if error is NonfiniteValueError:
                    self.assertEqual(caught.exception.info['record_id'], 'bad/force')

    def test_nonfinite_and_nonscalar_energy_refused(self):
        for value, error in [(float('inf'), NonfiniteValueError),
                             (np.array([1.]), ValueError), ('bad', ValueError)]:
            with self.subTest(value=value), self.assertRaises(error):
                self.arm(Harmonic(energy_result=value))['energy'](self.positions)

    def test_internal_calculator_error_propagates_without_retry(self):
        failure = ValueError('calculator operational failure')
        calc = Harmonic(error=failure)
        with self.assertRaises(ValueError) as caught:
            self.arm(calc)['forces'](self.positions)
        self.assertIs(caught.exception, failure)
        self.assertEqual(calc.requests, ['forces'])
        self.assertEqual(len(calc.calls), 1)

    def test_frozen_relaxation_matches_plain_analytic_supplier(self):
        procedure = dict(max_force_evaluations=12, initial_step_size=.2,
                         backtracking_factor=.5, min_backtracking_floor=1e-8,
                         convergence_tolerance_eV_per_A=.01,
                         max_backtracking_attempts_per_step=3)
        arm = self.arm()
        plain = relax(self.positions, lambda p: -2*p, procedure)
        wrapped = relax(self.positions, lambda p: arm['forces'](p)[0], procedure)
        self.assertEqual(plain.keys(), wrapped.keys())
        for key in plain:
            if isinstance(plain[key], np.ndarray):
                self.assertEqual(plain[key].tobytes(), wrapped[key].tobytes(), key)
            else:
                self.assertEqual(plain[key], wrapped[key], key)


if __name__ == '__main__':
    unittest.main()
