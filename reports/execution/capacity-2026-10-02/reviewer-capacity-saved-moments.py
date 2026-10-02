"""Pure inspection of stored moment arrays; no gradient or optimizer computation."""
import json
from pathlib import Path
import msgpack
import numpy as np
from force_consistency import capacity_contract as cc

def leaves(value):
    if isinstance(value, dict):
        return [a for v in value.values() for a in leaves(v)]
    shape, dtype, raw = msgpack.unpackb(value.data, raw=False)
    assert dtype == 'float64'
    return [np.frombuffer(raw, dtype=np.float64).reshape(shape)]

result = {}
for case in ('width32','width64'):
    root = Path('runs/aspirin-capacity-2026-10-02') / case
    provenance = cc.read(root/'provenance.json')
    result[case] = {}
    for update in (0,3000):
        state = cc._envelope(root/f'snapshots/{update}.msgpack', provenance)
        count_shape,count_dtype,count_raw = msgpack.unpackb(state['optimizer']['0']['count'].data,raw=False)
        assert count_shape == [] and count_dtype == 'int32' and int(np.frombuffer(count_raw,dtype=np.int32)[0]) == update
        moments = {}
        for key in ('mu','nu'):
            arrays = leaves(state['optimizer']['0'][key])
            assert all(np.isfinite(a).all() for a in arrays)
            if key == 'nu':assert all((a>=0).all() for a in arrays)
            nonzero = sum(int(np.count_nonzero(a)) for a in arrays)
            assert (nonzero == 0) == (update == 0)
            moments[key] = {'elements':sum(int(a.size) for a in arrays),'nonzero_elements':nonzero}
        result[case][str(update)] = {'adam_count':update,'moments':moments,'snapshot_sha256':cc.digest(root/f'snapshots/{update}.msgpack')}
out = {'status':'passed','scope':'Read/decode stored initial0 and incoming3000 moments only; no gradient/proposal/model computation.',
       'cases':result}
path = Path('reports/execution/capacity-2026-10-02/reviewer-capacity-saved-moments.json')
path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,sort_keys=True))
