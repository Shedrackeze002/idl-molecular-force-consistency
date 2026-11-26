"""Validation-only archive access and geometric controls with no force calls."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from force_consistency.data.geometry import align_records
from force_consistency.data.matching import classify_endpoint, _reference_classes
from force_consistency.data.source import verify_archive
from force_consistency.data.types import PreparationPolicy
from force_consistency.data.xyz import parse_xyz
from evaluation.runner import _bind_split_manifest, _own_reference
from evaluation.start_set import build_start_set


def validation_inputs(archive, manifest, classification):
    """Read true validation paths and minimum geometries; never test paths."""
    if _bind_split_manifest(manifest, synthetic_declaration=None) != 'approved':
        raise ValueError('Production validation manifest required')
    verify_archive(Path(archive))
    policy=PreparationPolicy(**manifest['policy'])
    groups=manifest['groups']['aspirin']['validation']
    if len(groups)!=5:raise ValueError('Five validation connections required')
    branch_members={rid for c in groups for ids in manifest['connections']['aspirin'][c]['branches'].values() for rid in ids}
    records=[];opened=[]
    with zipfile.ZipFile(archive) as z:
        minima={n for n in z.namelist() if re.fullmatch(r'Landscape17/aspirin/Conformations/coordsmin_\d+\.xyz',n)}
        if not minima:raise ValueError('Minimum reference pool missing')
        for name in sorted(branch_members|minima):
            records.append(parse_xyz(z.read(name).decode('utf-8'),source_path=name,molecule='aspirin'))
            opened.append(name)
    refs={r.record_id:r for r in records if r.kind=='minimum'}
    starts=build_start_set(manifest,records,policy,{'kind':'first_stored_frame'},partitions=('validation',))
    if len(starts)!=10:raise ValueError('Ten validation starts required')
    for key,entry in starts.items():
        _,own=_own_reference(key,classification,refs)
        if entry['species']!=own.species:raise ValueError('Reference atom order/species mismatch')
    # This namespace is the historical reporter's ID map, not start_set_hash.
    full_ids={f'aspirin/{c}/{side}':ids[0] for part in ('validation','test')
              for c in manifest['groups']['aspirin'][part]
              for side,ids in manifest['connections']['aspirin'][c]['branches'].items()}
    full_hash=hashlib.sha256(json.dumps(full_ids,sort_keys=True).encode()).hexdigest()
    if full_hash!='039893949b9b37f3505962815932327bf013e84a3915d91d1967b3daa3c78cc0':
        raise ValueError('Frozen held-out start ID map changed')
    payload={k:{'record_id':e['record_id'],'frame_index':e['frame_index'],
                'positions':np.asarray(e['positions']).tolist()} for k,e in starts.items()}
    species={k:list(e['species']) for k,e in starts.items()}
    evidence={'opened_members':opened,'test_branch_members_read':0,'full_heldout_id_map_sha256':full_hash,
        'validation_payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
        'validation_species_sha256':hashlib.sha256(json.dumps(species,sort_keys=True).encode()).hexdigest(),
        'reference_ids':sorted(refs),'reference_scope':'All aspirin minimum geometries; no test path payloads',
        'historical_full22_coordinate_payload_reverified':False}
    return starts,refs,policy,evidence


def _nearest(structure, refs, classes, class_unresolved, own_key, policy):
    scores={};incomplete=[]
    for key,ref in sorted(refs.items()):
        alignment=align_records(structure,ref,policy)
        if not alignment.search_complete or not np.isfinite(alignment.rmsd):
            incomplete.append({'reference':key,'reason':alignment.reason});scores[key]=None
        else:scores[key]=float(alignment.rmsd)
    if incomplete or class_unresolved:
        return {'status':'unresolved','own_class_nearest':None,'scores_A':scores,'incomplete':incomplete}
    best=min(scores.values())
    nearest_keys={k for k,v in scores.items() if abs(v-best)<=1e-10}
    accepted=[members for members in classes if nearest_keys.intersection(members)]
    if len(accepted)!=1:
        return {'status':'ambiguous','own_class_nearest':None,'scores_A':scores,'nearest_classes':accepted}
    return {'status':'matched','own_class_nearest':own_key in accepted[0],
            'scores_A':scores,'nearest_classes':accepted,'rmsd_A':best}


def geometry_controls(starts, references, classification, policy):
    """Geometric own-match count and nearest context, without convergence facts."""
    if not starts or not references:raise ValueError('Starts and reference pool required')
    classes,class_unresolved=_reference_classes(references,policy)
    rows={};own_verified=True;nearest_verified=True
    for key,entry in sorted(starts.items()):
        own_key,own=_own_reference(key,classification,references)
        structure=entry['structure']
        if (structure.record_id!=entry['record_id'] or structure.species!=tuple(entry['species'])
                or not np.array_equal(structure.positions,entry['positions'])
                or not np.isfinite(structure.positions).all()):raise ValueError('Control start structure changed')
        pool=classify_endpoint(structure,references,policy)
        individual=classify_endpoint(structure,{own_key:own},policy)
        known=pool['status']!='unresolved' and individual['status']!='unresolved'
        correct=(pool['status']=='matched' and individual['status']=='matched') if known else None
        nearest=_nearest(structure,references,classes,class_unresolved,own_key,policy)
        own_verified &= known;nearest_verified &= nearest['status']=='matched'
        rows[key]={'start_record_id':entry['record_id'],'reference_id':own.record_id,
            'own_geometry_match':correct,'pool_match':pool,'own_match':individual,'nearest':nearest,
            'force_evaluation_count':0,'step_count':0}
    return {'schema':1,'branch_count':len(rows),'branches':rows,'force_evaluation_count':0,'step_count':0,
        'own_geometry_status':'verified' if own_verified else 'unavailable',
        'own_geometry_count':sum(r['own_geometry_match'] for r in rows.values()) if own_verified else None,
        'nearest_context_status':'verified' if nearest_verified else 'unavailable',
        'nearest_own_class_count':sum(r['nearest']['own_class_nearest'] for r in rows.values()) if nearest_verified else None,
        'interpretation':'Unmodified-start geometry only; no model force convergence measured; nearest assignment is context, not a gate'}
