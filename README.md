# TomoGNN: DETR + GNN for Cryo-ET Organelles & Particles

[![DOI](https://zenodo.org/badge/863390386.svg)](https://doi.org/10.5281/zenodo.21502804)

TomoGNN is a pipeline for slice-wise detection and mask refinement in cryo-electron tomography (cryo-ET) volumes. It combines a DETR-based detector with graph neural network (GNN) context, optional mask heads, and a lightweight 3D CNN scorer to identify organelles (e.g., mitochondria, nucleus) and particles (e.g., ribosome, hsp60).

## Highlights
- DETR-based slice-wise detection with configurable stages (mask vs detection).
- GNN features and spatial post-processing to cluster and consolidate detections.
- Per-class mask refinement using a mask head and morphological smoothing.
- Optional 3D CNN scoring for improved candidate ranking.
- Reproducible training via PyTorch Lightning with TensorBoard logging and checkpoints.

## Repository Layout
- `src/` – Core modules and utilities
  - `data.py` – Dataset loaders (`MrcDataset`, `MrcDataModule`, particle 3D crops)
  - `modules.py` – Model definitions (DETR variants, mask head, 3D CNN)
  - `postprocess.py` – Thresholding, NMS, DBSCAN clustering, mask refinement, metrics
  - `utils.py` – I/O helpers, visualization (`drawannotation`), model utilities
- `scripts/` – Training entry points
  - `train.py` – Single-stage training driven by `<path>/config.json`
  - `train_full.py` – Two-stage training (Stage 1 → Stage 2) with best-checkpoint handoff
  - `result/` – Saved experiment outputs/checkpoints from prior runs
- `notebooks/` – End-to-end examples and analyses
  - `buildDataset.ipynb` – Construct dataset pickle from tomograms + label volumes
  - `train3DCNN.ipynb` – Train the optional 3D CNN scorer on particle crops
  - `trainModel.ipynb` – Minimal training walkthrough with Lightning
  - `scan_particle_with3DCNN_pipeline.ipynb` – End-to-end particle scan pipeline with optional 3D CNN rescoring
  - `scan_particles.ipynb` – Particle scanning, post-processing, 3D CNN scoring
  - `scan_organelle.ipynb` – Organelle detection and mask refinement
- `datasets/` – Place dataset pickle files here (not included)

## Hugging Face Assets
Example data and trained models used by notebooks in this repository are available on Hugging Face, with the data stored separately from the pretrained checkpoints:

- Dataset: [tyfei216/TomoGNN_data](https://huggingface.co/datasets/tyfei216/TomoGNN_data)
- Model checkpoints: [tyfei216/TomoGNN](https://huggingface.co/tyfei216/TomoGNN)
- Contents: notebook-ready example tomograms/labels (dataset repo) and pretrained checkpoints (model repo) for workflows under `notebooks/`

Use the separate data repository for example tomograms and labels, and use the model repository for pretrained checkpoints when running the notebook pipelines.

## Installation
Requirements (typical): Python ≥ 3.9, CUDA-capable GPU (optional but recommended), PyTorch, PyTorch Lightning.

```bash
# Create and activate conda environment
conda create -n pytorch python=3.12 -y
conda activate pytorch

# Install PyTorch (adjust CUDA build as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

## Data Format (Dataset Pickle)
Datasets are stored as Python pickle dictionaries consumed by `data.MrcDataset`.

- `mapclass`: `{class_name: int}` mapping (e.g., `{"ribosome": 0}` or `{ "mitochondria": 0, "nucleus": 1 }`).
- `annotations`: `{class_name: {slice_index: [label_ids...]}}` listing instance IDs per slice.
- `masks`: `{class_name: {slice_index: scipy.sparse.csr_matrix(H, W)}}` integer label map per slice (0 = background).
- `bboxes`: `{class_name: {slice_index: {label_id: [x_min, y_min, width, height]}}}` from mask extents.
- `mrc_path`: path to source `.mrc` volume; `mrc_shape`: full volume shape.

See `notebooks/buildDataset.ipynb` for a complete construction walkthrough.

## Quick Start
### 1) Build a Dataset
Use `notebooks/buildDataset.ipynb` to convert tomogram + label volumes into a dataset pickle with `annotations`, `masks`, and `bboxes`.

### 2) Train (Single-Stage)
Create an experiment folder with `config.json`, then run:
```bash
python scripts/train.py -p /path/to/experiment -d 0 
```
- Saves checkpoints and TensorBoard logs under the experiment path and `training.logger_path`.
- Configure data, model, and training options in `/path/to/experiment/config.json`.

### 3) Train (Two-Stage)
Stage 1 (mask) → pick best → Stage 2 (detection):
```bash
python scripts/train_full.py -p /path/to/experiment -d 0
```
- Stage 1 typically uses `data.require_mask=True` and trains representation with mask supervision.
- Stage 2 uses `data.require_mask=False` and focuses on detection; best Stage 1 checkpoint is loaded automatically.
- Checkpoints are written to `/path/to/experiment/stage2` and `/path/to/experiment/stage3` for the two training phases.

### 4) Inference & Post-processing
- Particles: `notebooks/scan_particles.ipynb`
  - Build `MrcDataset`, run detector to produce a candidates DataFrame `df`.
  - Post-process with class-specific thresholds, NMS, and DBSCAN.
  - Optionally re-score candidates with 3D CNN crops (`Particle3DDataset`), then recompute metrics.
- End-to-end particle + rescoring pipeline: `notebooks/scan_particle_with3DCNN_pipeline.ipynb`
  - Runs particle detection, post-processing, and optional 3D CNN rescoring in one workflow.
- Organelles: `notebooks/scan_organelle.ipynb`
  - Detect nucleus/mitochondria, visualize per-slice detections.
  - Switch to mask stage and refine masks with per-class morphology + thresholds.

## Configuration Notes
- `model.stage` controls the training and evaluation flow (e.g., `"stage 1 mask"`, `"stage 1 + 2"`, `"stage mask"`).
- Monitor metrics like `total_validate_loss` via TensorBoard.
- Device selection: pass GPU indices with `-d`, strategy via `--strategy`.
- Adjust per-class post-process parameters (`min_prob`, `nms`, DBSCAN settings) in notebooks or pipeline scripts.

## Visualizations
- Use `utils.drawannotation(img, labels)` to overlay bounding boxes and class labels per slice.
- Notebook cells demonstrate per-slice visualization and mask previews (raw sigmoid vs refined binary).

<!-- ## Tips & Troubleshooting
- Ensure the label volume aligns spatially with the raw tomogram when building datasets.
- Sparse masks (`csr_matrix`) reduce memory usage for large volumes.
- Tune DBSCAN (`eps`, `min_samples`, `dis_penalty_coef`) per dataset scale.
- For CUDA issues, verify `nvidia-smi` and match your PyTorch CUDA version. -->

<!-- ## Citation
If you use tomoGNN in your research, please cite this repository. A formal citation will be added when a manuscript/preprint is available. -->

<!-- ## License
Please consult the repository owner regarding licensing if you plan to redistribute or modify substantial portions of this codebase. -->
