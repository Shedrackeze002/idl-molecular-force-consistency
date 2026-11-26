import numpy as np
import pytest
from force_consistency.representation import rigid_projection, partition_constraints, cutoff_graph
from force_consistency.data.prepare import PreparedPartition

P = np.array([[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
R = np.array([[2/3,-2.,0.],[2/3,2.,0.],[-4/3,0.,0.]])
H = -np.array([[-1.,-1/3,0.],[1.,-1/3,0.],[0.,2/3,0.]])

@pytest.mark.parametrize('forces,floor,compatible', [
    (np.tile([.3,-.4,.1],(3,1)),.26/3,0.),(R,32/27,0.),(H,0.,8/27),
    (R+H+[.3,-.4,.1],32/27+.26/3,8/27)])
def test_analytic_rigid_and_harmonic_components(forces,floor,compatible):
    d=rigid_projection(P,forces)
    assert d['rank']==6
    assert d['floor_mse']==pytest.approx(floor,abs=1e-14)
    assert d['compatible_mse']==pytest.approx(compatible,abs=1e-14)
    assert d['raw_mse']==pytest.approx(floor+compatible)
    np.testing.assert_allclose(d['parallel']+d['compatible'],forces,atol=1e-14)
    assert max(abs(v) for v in d['checks'].values())<1e-13
    assert d['translation_mse']==pytest.approx(np.sum(np.sum(forces,axis=0)**2)/27)


def test_projection_matches_independent_least_squares_and_covariance():
    f=np.random.default_rng(7).normal(size=(3,3));r=P-P.mean(0)
    q=np.column_stack([np.tile(e,(3,1)).ravel() for e in np.eye(3)]+[np.cross(e,r).ravel() for e in np.eye(3)])
    want=(q@np.linalg.lstsq(q,f.ravel(),rcond=None)[0]).reshape(3,3)
    d=rigid_projection(P,f);np.testing.assert_allclose(d['parallel'],want,atol=1e-14)
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    for transformed,force in [(P+17,f),(P*1e-6,f),(P*1e6,f),(P@rotation.T,f@rotation.T)]:
        assert rigid_projection(transformed,force)['floor_mse']==pytest.approx(d['floor_mse'],abs=1e-12)

@pytest.mark.parametrize('positions,rank',[(np.array([[-1.,0,0],[0.,0,0],[1.,0,0]]),5),(np.zeros((1,3)),3)])
def test_degenerate_rigid_columns(positions,rank):
    d=rigid_projection(positions,np.ones_like(positions))
    assert d['rank']==rank and d['floor_mse']==pytest.approx(1.)
    assert np.max(abs(d['compatible']))<1e-14

@pytest.mark.parametrize('positions,forces',[(np.zeros((0,3)),np.zeros((0,3))),(P,np.zeros((2,3))),(P*float('nan'),P),(np.zeros(3),np.zeros(3))])
def test_invalid_projection_rejected(positions,forces):
    with pytest.raises(ValueError):rigid_projection(positions,forces)


def test_partition_pooling_groups_metadata_and_prediction_constraint():
    positions=np.tile(P,(3,1,1));forces=np.stack([R,H,2*H]);ids=tuple(f'Landscape17/aspirin/SteepestDescent/{g}/coordsminus_{i}.xyz' for i,g in enumerate(['01','01','23']))
    p=PreparedPartition(positions,np.zeros(3),forces,np.ones((3,3),int),ids,{r:{'role':'path','groups':[g]} for r,g in zip(ids,['01','01','23'])})
    d=partition_constraints(p,np.zeros_like(forces))
    assert d['record_ids']==list(ids) and d['force_component_count']==27
    assert d['pooled']['floor_mse']==pytest.approx(32/81)
    assert d['equal_group']['floor_mse']==pytest.approx(8/27)
    assert d['records'][0]['branch']=='minus' and d['records'][0]['groups']==['01']
    assert d['prediction_checks']['valid'] and d['prediction_checks']['max_projected_component']==0
    assert not partition_constraints(p,np.tile(R,(3,1,1)))['prediction_checks']['valid']


def test_graph_strict_cutoff_two_hops_and_components():
    graph=cutoff_graph(np.array([[0.,0,0],[1.,0,0],[2.,0,0],[10.,0,0]]),cutoff_A=1.5)
    assert graph['degree_min']==0 and graph['degree_max']==2
    assert graph['isolated_atoms']==1 and graph['connected_components']==2
    assert graph['directed_edges']==4 and graph['directed_pair_density']==pytest.approx(1/3)
    assert graph['reachable_pairs']==6 and graph['two_hop_fraction']==pytest.approx(.5)
    boundary=cutoff_graph(np.array([[0.,0,0],[5.,0,0]]))
    assert boundary['directed_edges']==0 and boundary['pairs_near_cutoff']==2
    one=cutoff_graph(np.zeros((1,3)))
    assert one['connected_components']==1 and one['two_hop_fraction'] is None
    with pytest.raises(ValueError):cutoff_graph(np.zeros((0,3)))
