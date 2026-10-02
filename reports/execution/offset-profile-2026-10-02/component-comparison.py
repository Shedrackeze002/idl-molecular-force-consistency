"""Saved-only decomposition on four pre-existing historical spike states."""
import json
from pathlib import Path

def build(report):
    if not report['execution_valid']:return {'status':'unendorsed','reason':report.get('reason')}
    spike=json.loads(Path('reports/capacity-spike-diagnostics.json').read_text())
    rows={r['update']:r for r in report['audit']['history']};out={}
    keys={'energy_mse_eV2':'energy_mse_eV2','force_mse_eV2_per_A2':'force_mse_eV2_per_A2',
        'signed_mean_energy_error_eV':'signed_mean_eV','mean_energy_error_squared_eV2':'squared_mean_eV2',
        'centered_energy_error_variance_eV2':'centered_population_variance_eV2'}
    for u in (3600,3700,3800,6000):
        old=spike['cases']['width32-u'+str(u)]['observations'];new=rows[u]['metrics']
        if not old['verification_valid']:raise ValueError('Historical decomposition not verified')
        out[str(u)]={}
        for role in ('fitting','anchor'):
            if old['record_ids'][role]!=new[role]['record_ids']:raise ValueError('Historical ordered support differs')
            def pair(a,b):
                return {source:{'historical':a[source],'intervention':b[target]} for source,target in keys.items()}
            values=pair(old['residuals'][role],new[role]);groups={}
            if set(old['residuals'][role]['per_group'])!=set(new[role]['per_group']):raise ValueError('Historical group membership differs')
            for group,a in old['residuals'][role]['per_group'].items():
                b=new[role]['per_group'][group]
                if a['record_count']!=b['record_count']:raise ValueError('Group support count differs')
                groups[group]=pair(a,b)
            out[str(u)][role]={'pooled':values,'same_role_groups':groups}
    return {'status':'verified_saved_data','matched_updates':[3600,3700,3800,6000],
        'scope':'Four pre-existing selected width32 historical spike states, separate from62-point raw comparison.',
        'method':'Population variance and squared pooled/group means; no per-group recalibration.',
        'limits':'Fitting pooled zero mean is enforced. Group mean and variance and force errors remain empirical. No dense historical trajectory or held-out claims.',
        'comparisons':out}
