# LMTK SLURM Multi-Node Tutorial

This guide shows how to launch a multi-node training job on SLURM without editing existing configs. It assumes you submit through `slurm/submit_job.sh` and use the Conda environment set up by `scripts/set_environment.sh`.

## Preconditions

- Your config uses a distributed strategy: `parallelization_strategy: fsdp`, `ddp`, or `deep_speed`.
- The dataset path in the config is accessible from all nodes.
- You will launch via SLURM (this is required for multi-node runs).

## Choose a Partition

Run these commands and pick a partition with at least two GPU nodes available:

```bash
sinfo -o "%P %a %l %D %N"
scontrol show partition postiguet1
scontrol show partition allen
scontrol show partition lovelace
```

Selection rule: prefer `postiguet1`, otherwise `allen`, then `lovelace`. Record the chosen partition in your issue notes.

## Submit a Multi-Node Job (No Config Changes)

You do **not** need to change existing configs (for example, `config/experiments/aitana-s2b/c0dc17.yaml`) as long as:
- it already uses `fsdp`/`ddp`/`deep_speed`, and
- you launch with `--nodes` and `--ntasks-per-node`.

Example (2 nodes, 4 GPUs per node):

```bash
./slurm/submit_job.sh \
  --config config/experiments/aitana-s2b/c0dc17.yaml \
  --nodes 2 \
  --ntasks-per-node 4 \
  --gpus 4 \
  --partition postiguet1
```

Notes:
- `--ntasks-per-node` must match the GPUs per node you want to use (same number as `--gpus`).
- Total tasks is `nodes * ntasks-per-node`; `submit_job.sh` passes `--ntasks` automatically to keep SLURM aligned.
- Multi-node runs automatically use `srun` inside `slurm/p.slurm`.

## Optional: Pin `num_nodes` and `devices_per_node` in the Config

If you want the config to enforce the allocation, add:

```yaml
num_nodes: 2
devices_per_node: 4
```

When these are set, they **must** match `--nodes` and `--ntasks-per-node`, or the run will fail fast with a clear error.

## Verify the Run

Check the SLURM logs for:
- `ENV` lines (python/torch/lightning versions).
- `Distributed settings` and `SLURM env` lines from the orchestrator.
- `fabric.world_size == num_nodes * devices_per_node`.

Default log filenames come from SLURM (e.g. `<job_id>_lmtk.out`).

## Smoke Test (Multi-Node)

Once you have a multi-node partition ready, run the dedicated smoke config:

```bash
./slurm/tests/run_tests.sh \
  --config config/tests/clm_training_multinode_smoke.yaml \
  --nodes 2 \
  --ntasks-per-node 1 \
  --gpus 1 \
  --partition postiguet1
```

## Common Errors

- `SLURM_NTASKS_PER_NODE is required...`: pass `--ntasks-per-node`.
- `devices_per_node must be set when num_nodes > 1`: add `devices_per_node` or remove `num_nodes`.
- `SLURM_NTASKS_PER_NODE exceeds available CUDA devices`: request fewer GPUs per node.
