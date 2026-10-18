"""Validation-only input access and genuine model-free geometric controls."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from evaluation.start_set import build_start_set
from force_consistency.data.types import PreparationPolicy


def controls():
    assert importlib.util.find_spec('evaluation.validation_controls'), 'validation controls missing'
    return __import__('evaluation.validation_controls',fromlist=['geometry_controls'])


def test_validation_selector_uses_true_manifest_without_test_records(synthetic_landscape,frozen_synthetic_manifest):
    land=synthetic_landscape;manifest=copy.deepcopy(frozen_synthetic_manifest)
    allowed=set(manifest['groups']['aspirin']['validation'])
    records=[r for r in land['records'] if r.connection_id in allowed or r.kind=='minimum']
    selected=build_start_set(manifest,records,land['policy'],{'kind':'first_stored_frame'},partitions=('validation',))
    assert len(selected)==6 and all(x['connection_id'] in allowed and x['frame_index']==0 for x in selected.values())
    assert manifest==frozen_synthetic_manifest
    default=build_start_set(manifest,land['records'],land['policy'],{'kind':'first_stored_frame'})
    explicit=build_start_set(manifest,land['records'],land['policy'],{'kind':'first_stored_frame'},partitions=('validation','test'))
    assert len(default)==10 and {k:x['record_id'] for k,x in default.items()}=={k:x['record_id'] for k,x in explicit.items()}


@pytest.mark.parametrize('parts',[(),('train',),('validation','validation'),'validation',('unknown',)])
def test_invalid_partition_selector_refused(parts,synthetic_landscape,frozen_synthetic_manifest):
    with pytest.raises(ValueError,match='partition'):
        build_start_set(frozen_synthetic_manifest,synthetic_landscape['records'],synthetic_landscape['policy'],{'kind':'first_stored_frame'},partitions=parts)


def geometry_case(land,classification):
    refs={r.record_id:r for r in land['records'] if r.kind=='minimum'}
    key=next(iter(classification));own=refs[classification[key]['reference_class']]
    other=next(v for k,v in refs.items() if k!=own.record_id)
    structure=replace(own,record_id='unrelaxed-start',kind='path',connection_id=key.split('/')[1],branch=key.split('/')[2],frame_index=0)
    entry={'record_id':structure.record_id,'positions':structure.positions,'species':structure.species,'structure':structure,'frame_index':0,'connection_id':structure.connection_id,'branch':structure.branch,'molecule':'aspirin'}
    return key,entry,own,other,refs


def test_no_relaxation_counts_geometry_without_convergence_or_force_calls(synthetic_landscape,synthetic_endpoint_classification):
    m=controls();key,entry,own,other,refs=geometry_case(synthetic_landscape,synthetic_endpoint_classification)
    out=m.geometry_controls({key:entry},refs,synthetic_endpoint_classification,synthetic_landscape['policy'])
    assert out['own_geometry_status']=='verified' and out['own_geometry_count']==1
    assert out['nearest_context_status']=='verified' and out['nearest_own_class_count']==1
    assert out['force_evaluation_count']==0 and out['step_count']==0
    assert 'converged' not in out['branches'][key]
    assert out['branches'][key]['own_geometry_match'] is True
    wrong={**entry,'positions':other.positions,'structure':replace(entry['structure'],positions=other.positions)}
    out=m.geometry_controls({key:wrong},refs,synthetic_endpoint_classification,synthetic_landscape['policy'])
    assert out['own_geometry_count']==0 and out['nearest_own_class_count']==0


def test_ambiguous_geometry_and_distinct_class_nearest_ties_retained(synthetic_landscape,synthetic_endpoint_classification):
    m=controls();key,entry,own,other,refs=geometry_case(synthetic_landscape,synthetic_endpoint_classification)
    mid=(own.positions+other.positions)/2
    entry={**entry,'positions':mid,'structure':replace(entry['structure'],positions=mid)}
    # Looser synthetic geometry tolerance makes both distinct classes accepted.
    out=m.geometry_controls({key:entry},refs,synthetic_endpoint_classification,replace(synthetic_landscape['policy'],endpoint_tolerance=.1))
    assert out['own_geometry_count']==0 and out['branches'][key]['pool_match']['status']=='ambiguous'
    # Pin exact nearest tie independently of floating alignment asymmetry.
    from types import SimpleNamespace
    def tied(*args):return SimpleNamespace(search_complete=True,rmsd=.03,reason=None)
    original=m.align_records;m.align_records=tied
    try:out=m.geometry_controls({key:entry},refs,synthetic_endpoint_classification,synthetic_landscape['policy'])
    finally:m.align_records=original
    assert out['nearest_context_status']=='unavailable' and out['nearest_own_class_count'] is None
    assert out['branches'][key]['nearest']['status']=='ambiguous'


def test_equivalent_reference_member_ties_collapse_to_one_class(synthetic_landscape,synthetic_endpoint_classification):
    m=controls();key,entry,own,other,refs=geometry_case(synthetic_landscape,synthetic_endpoint_classification)
    refs={own.record_id:own,'equivalent':replace(own,record_id='equivalent',positions=own.positions+np.array([2.,3.,4.]))}
    out=m.geometry_controls({key:entry},refs,synthetic_endpoint_classification,synthetic_landscape['policy'])
    assert out['own_geometry_count']==1 and out['nearest_own_class_count']==1
    assert out['branches'][key]['nearest']['status']=='matched'


def test_unresolved_nearest_does_not_invalidate_verified_own_geometry(monkeypatch,synthetic_landscape,synthetic_endpoint_classification):
    from types import SimpleNamespace
    m=controls();key,entry,own,other,refs=geometry_case(synthetic_landscape,synthetic_endpoint_classification)
    monkeypatch.setattr(m,'align_records',lambda *args:SimpleNamespace(search_complete=False,rmsd=float('inf'),reason='synthetic limit'))
    out=m.geometry_controls({key:entry},refs,synthetic_endpoint_classification,synthetic_landscape['policy'])
    assert out['own_geometry_status']=='verified' and out['own_geometry_count']==1
    assert out['nearest_context_status']=='unavailable' and out['nearest_own_class_count'] is None


def test_unresolved_own_control_count_withheld(monkeypatch,synthetic_landscape,synthetic_endpoint_classification):
    m=controls();key,entry,own,other,refs=geometry_case(synthetic_landscape,synthetic_endpoint_classification)
    monkeypatch.setattr(m,'classify_endpoint',lambda *args,**kw:{'status':'unresolved','accepted_classes':[],'reason':'synthetic limit'})
    out=m.geometry_controls({key:entry},refs,synthetic_endpoint_classification,synthetic_landscape['policy'])
    assert out['own_geometry_status']=='unavailable' and out['own_geometry_count'] is None


def test_real_validation_loader_never_reads_test_members(monkeypatch):
    m=controls();root=Path(__file__).resolve().parents[1];private=root
    manifest=json.loads((root/'manifests/landscape17-aspirin-split.json').read_text());classification=json.loads((root/'reports/landscape17-endpoints.json').read_text())
    validation=set(manifest['groups']['aspirin']['validation']);allowed={rid for c in validation for ids in manifest['connections']['aspirin'][c]['branches'].values() for rid in ids}
    actual=zipfile.ZipFile.read;opened=[]
    def guarded(z,name,*args,**kw):
        assert name in allowed or name.startswith('Landscape17/aspirin/Conformations/coordsmin_'), 'forbidden ZIP payload '+name
        opened.append(name);return actual(z,name,*args,**kw)
    monkeypatch.setattr(zipfile.ZipFile,'read',guarded)
    starts,refs,policy,evidence=m.validation_inputs(private/'data/Landscape17.zip',manifest,classification)
    assert len(starts)==10 and all(x['frame_index']==0 and x['connection_id'] in validation for x in starts.values())
    assert evidence['opened_members']==opened and len(opened)==len(allowed)+len(refs)
    assert evidence['full_heldout_id_map_sha256']=='039893949b9b37f3505962815932327bf013e84a3915d91d1967b3daa3c78cc0'
    assert evidence['test_branch_members_read']==0
    assert all(x['structure'].record_id==x['record_id'] for x in starts.values())
