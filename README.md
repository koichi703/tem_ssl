# TEM Self-Supervised Learning Pilot

A model-case pipeline for turning DM4/TEM images into a reproducible
self-supervised-learning dataset.

## Scientific design

The pipeline does **not** mix all magnifications blindly.

1. Read DM4 physical pixel size with HyperSpy.
2. Assign each source image to a physical-scale group.
3. Crop patches by **physical field of view in nm**, not simply by pixel count.
4. Resample patches within each group to 224×224.
5. Split train/validation/test by **original source DM4**, never by random patch.
6. Train a SimCLR-style ResNet-18 encoder separately for each eligible scale group.
7. Extract learned 512-dimensional embeddings.
8. Extract handcrafted intensity / GLCM / LBP / edge / FFT descriptors.
9. Perform correlation-based feature selection for handcrafted descriptors.
10. Run PCA / exploratory KMeans and cross-source nearest-neighbour checks.

The default configuration was chosen as a pilot for the uploaded
DM4 series:

- `lattice`: ≤ 0.0125 nm/pixel, 4 nm field-of-view patches
- `nano`: ≤ 0.05 nm/pixel, 15 nm patches
- `meso`: ≤ 0.5 nm/pixel, 100 nm patches
- `micro`: > 0.5 nm/pixel, 1000 nm patches

The important point is not the names themselves; it is that each SSL model sees
patches representing comparable physical scales.

**`scale_group` is an observation-scale bucket, not a crystallinity or material
class.** `lattice` means "imaged at atomic-scale physical resolution," not
"crystalline" -- an amorphous region imaged at 0.007 nm/pixel is still
`lattice`. The pipeline does not split training or models by crystallinity;
one SSL encoder per scale group sees both crystalline and amorphous patches
together, by design, so the representation is not told in advance which is
which. Bragg-score bands (below) are a separate, viewer-side reading of a
continuous score and are never used to filter what a model trains on.

## 1. Installation on Ubuntu

```bash
cd ~/tem_ssl_pilot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

For an NVIDIA GPU, install a CUDA-enabled PyTorch build appropriate for your system
if the default pip install gives a CPU-only build.

Check:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import hyperspy.api as hs; print('HyperSpy OK')"
```

## 2. Put DM4 files in one input folder

Example:

```text
~/TEM_data/
├── sample-0001.dm4
├── sample-0002.dm4
└── ...
```

Subfolders are searched recursively.

## 3. Prepare the pilot dataset

```bash
python tem_ssl_pilot.py prepare \
  ~/TEM_data \
  ~/TEM_SSL_pilot
```

Outputs:

```text
~/TEM_SSL_pilot/
├── dataset/
│   ├── manifest.csv
│   ├── sources.csv
│   ├── prepare_status.csv
│   ├── metadata/
│   └── patches/
│       ├── lattice/
│       ├── nano/
│       ├── meso/
│       └── micro/
└── pilot_config_used.json
```

Inspect:

```bash
column -s, -t < ~/TEM_SSL_pilot/dataset/sources.csv | less -S
```

## 4. Train self-supervised models

Pilot run:

```bash
python tem_ssl_pilot.py train \
  ~/TEM_SSL_pilot \
  --all-eligible \
  --epochs 10 \
  --batch-size 64
```

A more serious pilot can use 50–100 epochs after confirming that the pipeline works.

Groups with fewer than three independent source files are automatically skipped,
because a source-level train/validation/test split cannot be created.

To train only the lattice group:

```bash
python tem_ssl_pilot.py train \
  ~/TEM_SSL_pilot \
  --groups lattice \
  --epochs 20
```

## 5. Extract learned and handcrafted features

```bash
python tem_ssl_pilot.py extract \
  ~/TEM_SSL_pilot \
  --all-trained
```

Outputs under `features/` include:

- `ssl_embeddings_lattice.csv`
- `handcrafted_lattice.csv`
- corresponding files for other trained scale groups

The SSL vector is the 512-D ResNet-18 encoder output, not the projection-head vector.

## 6. Feature selection and exploratory analysis

```bash
python tem_ssl_pilot.py analyze \
  ~/TEM_SSL_pilot \
  --all-extracted
```

Per group:

```text
analysis/<group>/
├── handcrafted_feature_selection.csv
├── handcrafted_feature_correlation.csv
├── handcrafted_selected_raw.csv
├── pca_handcrafted.csv
├── pca_handcrafted_variance.csv
├── pca_ssl.csv
├── pca_ssl_variance.csv
└── cross_source_nearest_neighbors.csv
```

`cross_source_nearest_neighbors.csv` deliberately excludes patches from the same
source DM4 when finding neighbours. This is useful for checking whether the
representation generalizes across independent images rather than merely recognizing
a particular field of view.

## 7. Browser viewer

```bash
streamlit run viewer.py -- ~/TEM_SSL_pilot
```

Then open the address shown by Streamlit, normally:

```text
http://localhost:8501
```

The viewer can display:

- source metadata, labelled by observation scale (e.g. "High-resolution /
  atomic scale (lattice)") rather than the raw `scale_group` key
- each patch, with its Bragg score (`bragg_ratio`, `bragg_d_nm`,
  `bragg_pixels`) when the manifest has one
- a Bragg-score distribution histogram and a High / Ambiguous / Low band
  filter that applies to both the Patches tab and the PCA scatters
- cross-source nearest-neighbour patches
- SSL PCA and handcrafted PCA, coloured by Bragg-score band when a
  `crystallinity_probe.csv` with valid thresholds is available

**The Bragg-score bands are not phase labels.** They are a per-dataset
relative reading of a continuous FFT score (see `analyze`'s crystallinity
probe, above): "High" means "high relative to this dataset's own score
distribution," not "confirmed crystalline." Treat them as a viewer-side
exploration aid, not as ground truth.

Datasets prepared before this scoring was added have no `bragg_ratio` column;
the viewer runs normally against them, just without the Bragg-score tab's
histogram and bands. If `crystallinity_probe.csv` is missing, empty, or its
thresholds are non-finite, the viewer falls back the same way and says so.

## Why the augmentation is conservative

For HRTEM/TEM, arbitrary geometric augmentation can alter scientifically meaningful
spatial frequencies. The pilot therefore uses:

- horizontal/vertical flips
- exact 90° rotations
- mild gamma variation
- mild noise
- occasional mild Gaussian blur

It does **not** use random resize/crop during SimCLR training. Physical field of view
was already standardized during dataset preparation.

## Critical interpretation rule

Many patches from one DM4 are *not* many independent experiments.

Never report a random patch-level train/test split as evidence of generalization.
The script intentionally splits by `source_id`, which corresponds to the original
DM4/signal.

For a publishable study, add more independently acquired DM4 source images,
preferably across specimens, sessions and/or acquisition conditions that match the
scientific generalization claim.
