"""Configuration and shape-only parameter matching; no data loading."""
from dataclasses import asdict
from pathlib import Path
import jax
import numpy as np
from force_consistency.models.backbone import ModelConfig
from force_consistency.models.energy import EnergyModel,make_batch
from force_consistency.train import RunConfig

SPLIT='73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a'
WEIGHTS={'energy':.014584365670390719,'force':1.9854156343296092}


def make_configs(data_directory,output_root,scalar_width):
    if type(scalar_width) is not int or not 8<=scalar_width<=256:
        raise ValueError('Scalar width outside8–256')
    result={}
    for name,width,degree in [('E64',64,1),('SP',scalar_width,0)]+([] if scalar_width==64 else [('S64',64,0)]):
        for seed in (0,1):
            key=f'{name}-{seed}'
            result[key]=asdict(RunConfig(data_directory=str(data_directory),expected_split_hash=SPLIT,
                run_directory=str(Path(output_root)/key),mode='integration',backend='cpu',seed=seed,
                batch_size=4,learning_rate=.003,validation_interval=500,max_updates=200000,
                wall_seconds=86400,model=asdict(ModelConfig(features=width,max_degree=degree,
                    iterations=2,radial_basis=8,cutoff_A=5.,energy_readout_init='zero')),
                arm='energy',loss_weights=dict(WEIGHTS)))
    return result


def choose_scalar_width(counts,target):
    width=min(counts,key=lambda w:(abs(counts[w]-target),w))
    if abs(counts[width]-target)>target*.01:
        raise ValueError('No scalar parameter match within1%')
    return width


def count_parameters(config):
    model=EnergyModel(ModelConfig(**config['model']))
    b=make_batch(np.array([[[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3],[-.2,.3,1.4]]]),np.array([[6,1,8,1]]))
    shapes=jax.eval_shape(lambda key:model.init(key,b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1),jax.random.PRNGKey(0))
    return sum(x.size for x in jax.tree.leaves(shapes['params']))


def parameter_counts():
    configs=make_configs('unused','unused',64);reference=count_parameters(configs['E64-0'])
    counts={w:count_parameters({**configs['SP-0'],'model':{**configs['SP-0']['model'],'features':w}}) for w in range(8,257)}
    if reference!=43521:raise ValueError('Historical E64 parameter count differs')
    width=choose_scalar_width(counts,reference)
    return {'reference':reference,'scalar_counts':counts,'scalar_width':width,
            'relative_parameter_gap':abs(counts[width]-reference)/reference,'shape_only':True}
