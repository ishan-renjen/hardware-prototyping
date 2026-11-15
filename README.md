 to set up conda env:
1. conda create -n fpga-jax python=3.12
2. conda activate fpga-jax
3. pip install "jax[cuda12]==0.11.2"   (NVIDIA GPU, driver 525 or newer; CUDA libraries come from pip)
   or, for CPU only: pip install "jax==0.11.2" "jaxlib==0.11.2"
4. optional, for the framework adapters in test-programs (offload_haiku.py, offload_torch.py):
   pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cpu
   pip install "dm-haiku==0.0.17" "torchax==0.0.13"
   (torchax 0.0.13 does not import with torch 2.15; the CPU build of torch avoids a second set of CUDA libraries)
5. AlphaFold 2 (git submodule in alphafold/, used by test-programs/alphafold_blocks_test.py):
   git submodule update --init
   pip install ml-collections
   optional real weights (5.6 GB download, 5.3 GB on disk, git-ignored):
   mkdir -p alphafold_params/params && cd alphafold_params/params
   curl -O https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
   tar -xf alphafold_params_2022-12-06.tar && rm alphafold_params_2022-12-06.tar
   The AlphaFold blocks the gateway models are specified in grpc-server/alphafold_blocks.py.

6. running a real protein through AlphaFold with the blocks on the device (needs step 5 with real weights):
   pip install tensorflow-cpu==2.21.0 biopython==1.88   (TensorFlow is only used for AlphaFold's feature processing)
   cd test-programs
   python alphafold_msa.py ubiquitin MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
       sends the sequence to the public ColabFold MSA server (api.colabfold.com): public proteins only.
       Set COLABFOLD_USER_AGENT to identify yourself to that service. Cached in alphafold_inputs/<name>/.
   python alphafold_features.py ubiquitin
   python alphafold_run.py ubiquitin [--num-recycle 0] [--no-check] [--trace-layers 0,23,47]   (gateway running)
       writes alphafold_outputs/<name>/: host.pdb, device.pdb, ranges.csv, traces/*.npz
   The full run makes 960 device calls (4 passes x 48 layers x 5 blocks); --num-recycle 0 makes 240.

notes for running on a GPU:
- JAX reserves most of the GPU's memory at start-up; on a shared GPU set XLA_PYTHON_CLIENT_PREALLOCATE=false
  and pick a card with CUDA_VISIBLE_DEVICES.
- JAX's default matmul precision on the GPU is reduced (about 1e-3 relative error on an A100). For results
  comparable with the hardware, call jax.config.update("jax_default_matmul_precision", "float32").