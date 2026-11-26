"""Keep the notebook readable and end to end, with exactly four code cells."""
import json
from pathlib import Path


def test_four_cell_notebook_has_no_operational_barriers():
    p=Path(__file__).resolve().parents[1]/'notebooks/representation_comparison.ipynb'
    doc=json.loads(p.read_text());assert len(doc['cells'])==4 and all(c['cell_type']=='code' for c in doc['cells'])
    source='\n'.join(''.join(c['source']) for c in doc['cells'])
    assert 'run_queue' in source and 'verify_execution' in source and 'paired' in source
    for forbidden in ['READY_FOR_REVIEW','HUMAN_RELEASE','drive.mount','SESSION_STARTED_UTC','owner_recovery','segment_hours']:
        assert forbidden not in source
    for c in doc['cells']:compile(''.join(c['source']),str(p),'exec')
