from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from force_consistency.data.types import FrameRecord, PreparationPolicy


def frame(name, positions=None, species=("C","H","O"), force=None, energy=0.0):
    p=np.array(positions if positions is not None else [[0,0,0],[1,0,0],[0,1.2,0]],dtype=float)
    f=-(p-p.mean(0)) if force is None else np.array(force,dtype=float)
    return FrameRecord(name,"synthetic","path",name,"00","plus",0,species,p,energy,f,"0"*64)


def test_alignment_handles_orthogonal_transforms_and_species_order():
    from force_consistency.data.geometry import align_records
    a=frame('a');q=np.array([[0,1,0],[1,0,0],[0,0,1.]])
    order=np.array([2,0,1])
    b=frame('b',(a.positions@q+3)[order],tuple(a.species[i] for i in order),a.forces[order]@q)
    r=align_records(a,b,PreparationPolicy())
    assert r.search_complete and r.rmsd<1e-12
    np.testing.assert_allclose(a.positions@r.rotation+r.translation,b.positions[r.permutation],atol=1e-12)
    np.testing.assert_allclose(a.forces@r.rotation,b.forces[r.permutation],atol=1e-12)


def test_illegal_graph_mapping_and_search_limit_are_unresolved():
    from force_consistency.data.geometry import align_records
    a=frame('a');b=frame('b',[[0,0,0],[8,0,0],[0,9,0]])
    assert not align_records(a,b,PreparationPolicy()).search_complete
    c=frame('c',[[0,0,0],[1,0,0],[0,1,0]],("C","C","C"))
    assert not align_records(c,c,replace(PreparationPolicy(),search_limit=1)).search_complete


def test_near_duplicates_do_not_require_equal_force_labels():
    from force_consistency.data.matching import audit_duplicates
    a=frame('a');p=a.positions.copy();p[1,0]+=1e-6
    b=frame('b',p,force=-10000*p)
    out=audit_duplicates([a,b],PreparationPolicy())
    assert out.pairs and not out.label_conflicts and not out.unresolved


def test_exact_same_positions_reject_conflicting_labels():
    from force_consistency.data.matching import audit_duplicates
    a=frame('a');b=replace(a,record_id='b',energy=1.0)
    assert audit_duplicates([a,b],PreparationPolicy()).label_conflicts


def test_symmetric_exact_positions_do_not_hide_conflicting_labels():
    from force_consistency.data.matching import audit_duplicates
    positions=.63*np.array([[0,0,0],[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])+1
    a=frame('methane',positions,('C','H','H','H','H'))
    b=replace(a,record_id='conflict',energy=1.)
    out=audit_duplicates([a,b],PreparationPolicy())
    assert out.diagnostics[0]['exact_stored_positions'] and out.label_conflicts
    # An exact row reordering also requires the exact coordinate mapping.
    order=np.array([3,0,4,1,2]);forces=a.forces[order].copy();forces[0,0]+=1.
    c=replace(a,record_id='force-conflict',species=tuple(a.species[i] for i in order),positions=a.positions[order],forces=forces)
    assert audit_duplicates([a,c],PreparationPolicy()).label_conflicts


def test_descriptor_screen_includes_boundary_and_reordered_pairs():
    from force_consistency.data.geometry import candidate_pairs, align_records
    a=frame('a');p=a.positions.copy();p[1,0]+=1e-5
    b=frame('b',p)
    r=align_records(a,b,PreparationPolicy())
    assert r.rmsd<1e-5
    assert (0,1) in candidate_pairs([a,b],1e-5)


def test_aspirin_real_permutation_is_chemically_valid():
    from force_consistency.data.xyz import parse_xyz
    from force_consistency.data.geometry import align_records
    root=Path(__file__).resolve().parents[1]/'data/landscape17_sample/aspirin'
    ps=[root/'SteepestDescent/01/coordsminus_122.xyz',root/'Conformations/coordsmin_1.xyz']
    rs=[parse_xyz(p.read_text(),source_path='Landscape17/aspirin/'+str(p.relative_to(root)),molecule='aspirin') for p in ps]
    r=align_records(*rs,PreparationPolicy())
    assert r.search_complete and r.rmsd<0.01
    assert r.permutation[7]==9 and r.permutation[9]==7


def test_endpoint_classes_and_ambiguity():
    from force_consistency.data.matching import classify_endpoint
    a=frame('a');b=replace(a,record_id='b');end=replace(a,record_id='end')
    r=classify_endpoint(end,{'a':a,'b':b},PreparationPolicy())
    assert r['status']=='matched' and len(r['accepted_classes'])==1
    p=a.positions.copy();p[1,0]+=0.04
    c=frame('c',p)
    r=classify_endpoint(end,{'a':a,'c':c},PreparationPolicy())
    assert r['status']=='ambiguous'
