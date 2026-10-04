"""Classified branch targets with immutable source binding.

Historical endpoint_pair values describe filename tokens. They cannot select
plus/minus targets. reference_class is canonical; reference_record is separate
matching-representative metadata. This module performs no geometry matching.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


class EndpointReferenceError(ValueError):
    pass


def canonical_hash(value):
    try:
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    except (TypeError,ValueError) as exc:
        raise EndpointReferenceError('Reference data is not finite JSON') from exc
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise EndpointReferenceError('Duplicate JSON key: '+key)
        result[key]=value
    return result


def _read(path):
    try:
        raw=Path(path).read_bytes()
        return json.loads(raw,object_pairs_hook=_pairs),hashlib.sha256(raw).hexdigest()
    except (OSError,ValueError,TypeError) as exc:
        raise EndpointReferenceError('Cannot read reference evidence: '+str(path)) from exc


def _minimum(value,molecule):
    if not isinstance(value,str) or not re.fullmatch(
            r'Landscape17/'+re.escape(molecule)+r'/Conformations/coordsmin_[0-9]+\.xyz',value):
        raise EndpointReferenceError('Invalid same-molecule minimum reference')
    return value


def _references(split,classes):
    if not isinstance(split,dict) or not isinstance(classes,dict):raise EndpointReferenceError('Expected source dictionaries')
    payload={k:v for k,v in split.items() if k!='identity'}
    if split.get('identity')!=canonical_hash(payload):raise EndpointReferenceError('Stale split identity')
    references={};filename_pairs={}
    try:
        connections=split['connections']
        if not isinstance(connections,dict) or not connections:raise EndpointReferenceError('Missing selected connections')
        for molecule,groups in sorted(connections.items()):
            if not isinstance(molecule,str) or not isinstance(groups,dict) or not groups:raise EndpointReferenceError('Invalid selected connections')
            for group,connection in sorted(groups.items()):
                branches=connection['branches']
                if set(branches)!= {'plus','minus'}:raise EndpointReferenceError('Missing or extra branch')
                pair=connection['endpoint_pair']
                if not isinstance(pair,list) or len(pair)!=2 or any(not isinstance(v,str) or not v.isdigit() for v in pair):
                    raise EndpointReferenceError('Invalid historical filename metadata')
                filename_pairs[molecule+'/'+group]=list(pair)
                for branch,records in sorted(branches.items()):
                    if not isinstance(records,list) or not records or any(not isinstance(v,str) or not v for v in records):
                        raise EndpointReferenceError('Missing branch record sequence')
                    key=molecule+'/'+group+'/'+branch;entry=classes.get(key)
                    if not isinstance(entry,dict) or entry.get('status')!='matched':raise EndpointReferenceError('Unmatched or missing endpoint: '+key)
                    target=_minimum(entry.get('reference_class'),molecule)
                    representative=_minimum(entry.get('reference_record'),molecule)
                    if entry.get('accepted_classes')!=[target] or entry.get('incomplete')!=[] or entry.get('reference_unresolved')!=[]:
                        raise EndpointReferenceError('Ambiguous or uncertified endpoint: '+key)
                    if entry.get('record_id')!=records[-1]:raise EndpointReferenceError('Classification is not the branch terminal: '+key)
                    references[key]={'status':'matched','reference_class':target,'reference_record':representative,'endpoint_record_id':records[-1]}
    except (KeyError,TypeError,AttributeError) as exc:
        raise EndpointReferenceError('Malformed split/classification source') from exc
    return references,filename_pairs


def build_reference_manifest(split_path,classification_path):
    """Build a companion, leaving frozen sources and their identities unchanged."""
    split,split_sha=_read(split_path);classes,class_sha=_read(classification_path)
    references,pairs=_references(split,classes)
    payload={'schema':1,'split_identity':split['identity'],
        'source_sha256':{'split_manifest':split_sha,'endpoint_classification':class_sha},
        'target_authority':'committed per-branch reference_class; no filename fallback',
        'filename_metadata_semantics':'historical transition-state filename tokens; no guaranteed plus/minus orientation',
        'filename_minimum_pairs':pairs,'branch_references':references}
    return {**payload,'identity':canonical_hash(payload)}


@dataclass(frozen=True)
class VerifiedBranchReferences:
    identity: str
    split_identity: str
    references: Mapping[str,str]
    representatives: Mapping[str,str]
    endpoint_records: Mapping[str,str]


def load_reference_manifest(path,split_path,classification_path):
    """Verify source files, companion hash and every canonical target before use.

    Source files must be the evaluation protocol's frozen, trusted evidence.
    This checks consistency and integrity, not the physical classification anew.
    """
    actual,_=_read(path)
    if not isinstance(actual,dict) or type(actual.get('schema')) is not int or actual['schema']!=1:
        raise EndpointReferenceError('Unsupported companion schema')
    payload={k:v for k,v in actual.items() if k!='identity'}
    if actual.get('identity')!=canonical_hash(payload):
        raise EndpointReferenceError('Stale companion identity')
    expected=build_reference_manifest(split_path,classification_path)
    if actual!=expected:raise EndpointReferenceError('Companion differs from frozen source evidence')
    rows=actual['branch_references']
    return VerifiedBranchReferences(actual['identity'],actual['split_identity'],
        MappingProxyType({k:v['reference_class'] for k,v in rows.items()}),
        MappingProxyType({k:v['reference_record'] for k,v in rows.items()}),
        MappingProxyType({k:v['endpoint_record_id'] for k,v in rows.items()}))


def reference_for_branch(verified,key):
    if not isinstance(verified,VerifiedBranchReferences):raise EndpointReferenceError('Load and verify the companion before lookup')
    try:return verified.references[key]
    except (KeyError,TypeError) as exc:raise EndpointReferenceError('Unknown branch: '+str(key)) from exc
