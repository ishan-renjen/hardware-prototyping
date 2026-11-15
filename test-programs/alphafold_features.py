"""Turns a cached MSA into AlphaFold 2 model inputs, using AlphaFold's own pipeline.

python alphafold_features.py ubiquitin [model_name]

Reads alphafold_inputs/<name>/ (written by alphafold_msa.py) and writes
alphafold_inputs/<name>/features_<model_name>.npz. Feature processing is
AlphaFold's TensorFlow code, so this runs as a separate step. The default model,
model_3, does not use templates, so no template search or PDB database is needed.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'alphafold'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
from alphafold.data import parsers, pipeline
from alphafold.model import config, features
from alphafold_msa import A3M_FILES, INPUTS


def make_features(name, model_name='model_3', random_seed=0):
    in_dir = os.path.join(INPUTS, name)
    with open(os.path.join(in_dir, 'query.fasta')) as f:
        (sequence,), _ = parsers.parse_fasta(f.read())
    msas = []
    for a3m in A3M_FILES:
        with open(os.path.join(in_dir, a3m)) as f:
            msas.append(parsers.parse_a3m(f.read()))

    cfg = config.model_config(model_name)
    if cfg.data.common.use_templates:
        sys.exit(f'{model_name} uses templates, which need a template search; use model_3, 4 or 5')
    raw = {
        **pipeline.make_sequence_features(sequence=sequence, description=name, num_res=len(sequence)),
        **pipeline.make_msa_features(msas),
    }
    print(f'{name}: {len(sequence)} residues, {int(raw["num_alignments"][0])} unique sequences in the MSA')

    processed = features.np_example_to_features(np_example=raw, config=cfg, random_seed=random_seed)
    out = os.path.join(in_dir, f'features_{model_name}.npz')
    np.savez_compressed(out, **processed)
    print(f'wrote {out}')
    for key in sorted(processed):
        print(f'  {key:32s} {processed[key].dtype}{list(processed[key].shape)}')
    return out


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    make_features(*sys.argv[1:])
