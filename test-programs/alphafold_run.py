"""Runs AlphaFold 2 on a real protein twice, once on the host and once with the
Evoformer blocks on the 'alphafold' device, and records what the device saw.

python alphafold_run.py ubiquitin [--num-recycle N] [--no-check] [--trace-layers 0,23,47]

Needs alphafold_inputs/<name>/features_model_3.npz (alphafold_features.py), the real
weights in alphafold_params/, and the gateway running. Writes to alphafold_outputs/<name>/:
  host.pdb, device.pdb     predicted structures, pLDDT in the B-factor column
  ranges.csv               min/max/mean/abs-max of every operand and result of every device call
  traces/*.npz             full operands, device result and CPU reference for the traced layers
"""
import argparse
import collections
import csv
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'alphafold'))

import alphafold_compat  # noqa: F401  (must come before AlphaFold runs)
import haiku as hk
import jax
import numpy as np
from alphafold.common import confidence, protein
from alphafold.model import config, data, modules
from offload_haiku import offload_haiku_modules

# Full float32 matmuls if the model itself runs on a GPU (see README).
jax.config.update('jax_default_matmul_precision', 'float32')

MODEL = 'model_3'  # no templates, so no template search is needed
MAIN_STACK = '/evoformer/evoformer_iteration/'  # not the extra-MSA or template stacks
NUM_LAYERS = 48
# Haiku module name -> program on the device
BLOCKS = {
    'msa_transition': 'transition',
    'outer_product_mean': 'outer_product_mean',
    'triangle_multiplication_outgoing': 'triangle_multiplication_outgoing',
    'triangle_multiplication_incoming': 'triangle_multiplication_incoming',
    'pair_transition': 'transition',
}


def on_device(context):
    if context.method_name != '__call__':  # __init__ is intercepted too, before the name exists
        return None
    path = context.module.module_name
    name = path.rsplit('/', 1)[-1]
    if MAIN_STACK in path + '/' and name in BLOCKS:
        return BLOCKS[name]


class Recorder:
    """trace callback for offload(): value ranges for every call, full tensors for some."""

    def __init__(self, out_dir, trace_layers, last_pass):
        self.counts = collections.Counter()
        self.trace_layers, self.last_pass = set(trace_layers), last_pass
        self.trace_dir = os.path.join(out_dir, 'traces')
        os.makedirs(self.trace_dir, exist_ok=True)
        self.file = open(os.path.join(out_dir, 'ranges.csv'), 'w', newline='')
        self.csv = csv.writer(self.file)
        self.csv.writerow(['pass', 'layer', 'module', 'tensor', 'shape', 'min', 'max', 'mean', 'abs_max'])

    def __call__(self, program, args, result, reference):
        weights, *inputs = args
        # The module is named in its weights' full Haiku names.
        module = next(iter(weights)).split(MAIN_STACK)[1].split('/')[0]
        n = self.counts[module]
        self.counts[module] += 1
        recycle_pass, layer = divmod(n, NUM_LAYERS)
        tensors = {name.split(module + '/')[1]: v for name, v in sorted(weights.items())}
        tensors.update(act=inputs[0], mask=inputs[1], result=np.asarray(result))
        for name, v in tensors.items():
            self.csv.writerow([recycle_pass, layer, module, name, 'x'.join(map(str, v.shape)),
                               f'{v.min():.6g}', f'{v.max():.6g}', f'{v.mean():.6g}', f'{np.abs(v).max():.6g}'])
        if recycle_pass == self.last_pass and layer in self.trace_layers:
            path = os.path.join(self.trace_dir, f'pass{recycle_pass}_layer{layer:02d}_{module}.npz')
            extra = {} if reference is None else {'reference': np.asarray(reference)}
            # Operand order on the wire: weights sorted by name, then act, then mask.
            np.savez(path, **{name.replace('/', '__'): v for name, v in tensors.items()}, **extra)

    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('name')
    parser.add_argument('--num-recycle', type=int, default=3, help='recycling passes after the first (AlphaFold uses 3)')
    parser.add_argument('--no-check', action='store_true', help='skip the per-call CPU reference check')
    parser.add_argument('--trace-layers', default='0,23,47', help='layers of the last pass to save in full')
    opts = parser.parse_args()

    feature_path = os.path.join(REPO, 'alphafold_inputs', opts.name, f'features_{MODEL}.npz')
    out_dir = os.path.join(REPO, 'alphafold_outputs', opts.name)
    os.makedirs(out_dir, exist_ok=True)
    with np.load(feature_path) as f:
        feat = {k: f[k] for k in f.files}
    available = feat['aatype'].shape[0] - 1
    if opts.num_recycle > available:
        sys.exit(f'features were made for at most {available} recycling passes')
    # The features hold one resampled MSA per pass; keep the ones this run uses.
    feat = {k: v[:opts.num_recycle + 1] for k, v in feat.items()}

    cfg = config.model_config(MODEL)
    cfg.model.num_recycle = opts.num_recycle
    params = data.get_model_haiku_params(MODEL, os.path.join(REPO, 'alphafold_params'))

    def forward(batch):
        return modules.AlphaFold(cfg.model)(batch, is_training=False, compute_loss=False,
                                            ensemble_representations=True)

    recorder = Recorder(out_dir, [int(x) for x in opts.trace_layers.split(',') if x], opts.num_recycle)

    def forward_on_device(batch):
        with offload_haiku_modules(on_device, kind='alphafold', check=not opts.no_check,
                                   rtol=1e-3, atol=1e-3, trace=recorder):
            return forward(batch)

    rng = jax.random.key(0)
    num_res = feat['aatype'].shape[1]
    print(f'{opts.name}: {num_res} residues, {MODEL}, {opts.num_recycle + 1} passes of {NUM_LAYERS} '
          f'Evoformer layers, {len(BLOCKS)} blocks per layer on the device')
    results = {}
    for label, fn in (('host', forward), ('device', forward_on_device)):
        start = time.time()
        out = jax.jit(hk.transform(fn).apply)(params, rng, feat)
        out = jax.tree.map(np.asarray, out)
        results[label] = out
        plddt = confidence.compute_plddt(out['predicted_lddt']['logits'])
        prot = protein.from_prediction(feat, out, b_factors=np.repeat(plddt[:, None], 37, axis=-1))
        with open(os.path.join(out_dir, f'{label}.pdb'), 'w') as f:
            f.write(protein.to_pdb(prot))
        print(f'{label} run: {time.time() - start:.0f}s, mean pLDDT {plddt.mean():.1f}')
    recorder.close()

    host, device = results['host'], results['device']
    mask = host['structure_module']['final_atom_mask'] > 0
    diff = np.linalg.norm(host['structure_module']['final_atom_positions']
                          - device['structure_module']['final_atom_positions'], axis=-1)
    ca = 1  # atom37 index of C-alpha
    print('device run vs host run:')
    print(f'  atom positions: max deviation {diff[mask].max():.2e} A, C-alpha RMSD '
          f'{np.sqrt(np.mean(diff[:, ca] ** 2)):.2e} A (same frame, no superposition)')
    plddt = {k: confidence.compute_plddt(r['predicted_lddt']['logits']) for k, r in results.items()}
    print(f'  pLDDT: max difference {np.abs(plddt["host"] - plddt["device"]).max():.2e} '
          f'(scale 0-100)')
    print(f'  distogram logits: max difference '
          f'{np.abs(host["distogram"]["logits"] - device["distogram"]["logits"]).max():.2e}')
    print(f'device calls: {dict(recorder.counts)}')
    print(f'wrote {out_dir}/: host.pdb, device.pdb, ranges.csv, traces/ ({len(os.listdir(recorder.trace_dir))} files)')


if __name__ == '__main__':
    main()
