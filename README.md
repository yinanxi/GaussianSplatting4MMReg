GaussianDIR (PyTorch) – Paper-faithful Refactor
================================================


Implements pairwise deformable image registration using sparsely distributed Gaussian primitives with:
- Multi-scale Gaussian primitives (coarse→fine)
- Adaptive density control (clone / prune by gradient norm)
- NCC similarity loss + mini-batch TV regularization
- Mini-batch gradient descent with cosine LR schedule


Key paper hyperparameters reflected here (config overrides available):
- Voxel mini-batch M = 20000
- Adam optimizer, total iters = 2000 (warm-up then cosine decay)
- Two-stage training: Stage-1 max Gaussians = 1/8*|Ω|; Stage-2 max Gaussians = 1/4*|Ω|
- Adaptive density thresholds: τ_max = 2e-3 (clone), τ_min = 1e-7 (prune)
- TV weight λ = 15


Run (MINC .mnc/.mnc.gz supported)
---
Install NiBabel first:


pip install nibabel


Example (register T2 → T1):


python main.py \
--fixed /path/t1_icbm_normal_1mm_pn3_rf20.mnc.gz \
--moving /path/t2_icbm_normal_1mm_pn3_rf20.mnc.gz \
--config configs/default.yaml \
--resample


Or PD → T1:


python main.py \
--fixed /path/t1_icbm_normal_1mm_pn3_rf20.mnc.gz \
--moving /path/pd_icbm_normal_1mm_pn3_rf20.mnc.gz \
--config configs/default.yaml \
--resample


`--resample` will resize the moving volume to match the fixed shape (trilinear).


Outputs
---
- Saves checkpoints in ./outputs/exp_xxxxx/
- Exports final DVF φ and warped image I_w (to be added in exporters)


Notes
---
- KNN backend defaults to torch.cdist (simple & differentiable). For speed, swap in FAISS or a voxel-hash grid.
- Initialization places Gaussian centers on a 3D grid in canonical space [-1,1]^3; local SE(3) = identity.
- Σ_i is built from scale s_i and rotation q_i as Σ_i = Q_i S_i S_i^T Q_i^T with Q_i from quaternion.
- If your MINC volumes differ in voxel spacing/orientation, resampling is recommended before training.
---
- KNN backend defaults to torch.cdist (simple & differentiable). For speed, swap in FAISS or a voxel-hash grid.
- Initialization places Gaussian centers on a 3D grid in canonical space [-1,1]^3; local SE(3) = identity.
- Σ_i is built from scale s_i and rotation q_i as Σ_i = Q_i S_i S_i^T Q_i^T with Q_i from quaternion.