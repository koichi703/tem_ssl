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

## How the self-supervised model learns

Each eligible scale group is trained independently with a **SimCLR-style
contrastive objective**. Two independently augmented views of the same patch
are pulled together in embedding space, while every other patch in the same
batch is pushed apart. No label is used anywhere in this step -- the model
never sees "crystalline" or "amorphous," only "these two crops came from the
same patch" (via augmentation) and "these are different patches" (via the
rest of the batch).

- **Encoder**: ResNet-18 with `conv1` changed to accept single-channel
  (grayscale) input. Its 512-D pooled output -- not the 128-D projection-head
  output -- is what `extract` saves as the SSL feature vector; the
  projection head exists only to make the contrastive loss easier to
  optimize and is discarded afterward, as in the original SimCLR paper.
- **Loss**: NT-Xent (normalized temperature-scaled cross-entropy) over
  cosine similarities between every pair in a batch, temperature 0.2 by
  default (`--temperature`).
- **Splits are by source DM4, not by patch.** Two patches from the same
  micrograph are correlated -- same specimen region, same acquisition
  conditions -- so a random patch-level split would leak information
  between train and validation. `grouped_split` assigns whole source files
  to train/val/test, and a group needs at least 3 independent sources
  before it is trained at all (see "Critical interpretation rule" below).
- **Augmentation is physically conservative** -- see "Why the augmentation
  is conservative" below. The model is never shown a randomly resized or
  cropped patch, because physical field of view was already standardized in
  `prepare`, and arbitrary geometric distortion can alter spatial
  frequencies that carry real scientific meaning in HRTEM.

## Algorithm: Bragg-score crystallinity scoring

`prepare` also computes a per-patch **Bragg score** directly from the
patch's own power spectrum: a lightweight, unsupervised signal for how
"crystalline" (discrete lattice reflections) versus "amorphous" (diffuse
halo) it looks. It is used only by the viewer and the crystallinity probe
below -- **never to filter, label, or split training data.** One SSL encoder
per scale group still trains on crystalline and amorphous patches together,
by design.

1. Take the patch's 2D FFT power spectrum (Hann-windowed to suppress edge
   artifacts).
2. Restrict to a physically calibrated d-spacing band (0.12–1.0 nm by
   default), clamped to 90% of the *acquisition* Nyquist frequency rather
   than the resampled 224 px grid's -- so an upsampled patch can't have
   interpolation ringing mistaken for a reflection, and a scale group too
   coarse to resolve that d-spacing (`meso`, `micro`, or any individual
   source whose pixel size makes the band unusable) reports **no score at
   all**: `bragg_ratio`, `bragg_d_nm`, and `bragg_pixels` all come back as
   `NaN`, never as a measured-looking `0.0`, so an unresolvable patch is
   never mistaken for a genuinely low-crystallinity one downstream.
3. For each radial ring in that band, compute `max power / median power`
   around the ring. A discrete Bragg reflection concentrates power at a few
   azimuths (high ratio); an amorphous halo is flat in angle (ratio near
   1). The ratio is further divided by its expectation under pure noise --
   `(ln(ring pixel count) + γ) / ln 2` -- so rings of different radius (and
   therefore different pixel counts) stay directly comparable.
4. `bragg_ratio` is the strongest ring's corrected ratio. `bragg_d_nm` is
   the d-spacing of the reflection that produced *that* ratio, not simply
   the brightest pixel in the band, which can belong to low-frequency
   morphology instead of the actual reflection. `bragg_pixels` counts
   pixels well above their own ring's local background.

Frequencies are calibrated per axis from the source's true pixel size, so a
source with non-square physical pixels doesn't have its genuinely circular
amorphous halo read as an elliptical -- and falsely high-ratio -- reflection.

## Algorithm: does the SSL representation encode crystallinity?

`analyze` runs a **linear probe**: patches at the extreme quantiles of
`bragg_ratio` (confidently high vs. confidently low; the ambiguous middle is
discarded) become a frozen-feature classification target for logistic
regression, written to `analysis/<group>/crystallinity_probe.csv`.

- **Evaluated only on sources the encoder never trained on -- and the
  crystalline/amorphous quantile cut is fit without them too.** When
  `models/<group>/split_manifest.csv` has a `test` split, the `hi`/`lo`
  thresholds are fit on the non-test sources only, so a held-out source's
  own scores never influence the very cut used to label it. If those test
  sources then carry both classes under that cut, they alone are evaluated
  (`protocol` = `encoder-held-out`).

  Two different situations fall back to leave-one-source-out instead, both
  recorded as `all-sources-encoder-exposed`: no usable split exists at all,
  so there is no held-out set to protect and every source both fits the
  thresholds and is evaluated; or a split exists but its test source
  doesn't carry both classes under the non-held-out thresholds. The second
  case reuses those *same* thresholds rather than refitting from every
  source -- so a held-out source's scores still never influence the cut --
  and drops that source from the fallback entirely, neither fitting on it
  nor evaluating it, so it is never counted as "encoder-exposed" under a
  protocol name that claims exactly that.
- **Per-fold standardization.** Any column-wise scaler is fit on the
  training fold only -- fitting it on the whole set would leak the held-out
  source's own statistics into an L2-regularized classifier and inflate the
  reported AUC.
- **Circular features are flagged, not silently ranked.** `fft_bragg_*`
  reproduces the label almost by definition, and the rest of the `fft_*`
  descriptors share its power spectrum; both are reported (`circular=True`)
  for reference but excluded from the ranking, so a spectrum-derived
  feature can never be reported as "best."

This is the closest thing in the pipeline to a downstream evaluation task: a
genuine test of whether the *unsupervised* 512-D embedding separates
crystalline from amorphous structure, without ever having been told which
patch is which.

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

The sidebar picks the **observation scale** (`lattice`, `nano`, `meso`,
`micro`, shown as e.g. "High-resolution / atomic scale (lattice)" rather
than the raw key) and, when a crystallinity probe is available, a
**Bragg-score band filter** (High / Ambiguous / Low, plus an "unscored"
option) that applies to the **Patches tab and both PCA scatters**. The
unscored option matters for a scale group that mixes scorable and
acquisition-Nyquist-unscorable sources (see "Algorithm: Bragg-score
crystallinity scoring" above): without it, those patches would silently
disappear from the source picker and both PCA tabs even with High,
Ambiguous, and Low all selected. The Bragg-score tab's histogram and
per-band counts intentionally show the whole current scale group regardless
of this filter -- it is the reference distribution the bands were cut from,
so filtering it would be circular. The Sources tab is unaffected too, since
it reads `sources.csv` rather than per-patch scores.

### Patches

Pick a source, then a patch, to see its image, patch-level metadata, its
Bragg score (`bragg_ratio`, `bragg_d_nm`, `bragg_pixels`) as labelled
metrics, and its top-5 cross-source nearest neighbours -- patches from
*other* source DM4 files that the SSL embedding considers most similar,
useful for a quick visual check of whether the representation generalizes
across independent images rather than recognizing one field of view.

### Bragg score

The distribution of `bragg_ratio` across the current scale group as a
histogram (log-scaled x-axis; the score is heavily right-skewed), plus the
patch/source count in each band and a per-source breakdown of band
composition -- read this before deciding whether the training sampler needs
to change for a source-imbalance reason, not as a cue to change it
automatically.

**Click a bar** to see every patch whose score falls in that bin, as an
image grid below the chart (capped at the first 32 patches, in manifest row
order, if the bin holds more than that; the caption below the grid reports
the true total).

### Sources

The `sources.csv` rows for the current scale group: physical pixel size,
field of view, magnification, patch count.

### SSL PCA / Handcrafted PCA

A PCA projection (PC1 vs. PC2) of the 512-D SSL embedding, and separately
of the selected handcrafted descriptors, each coloured by Bragg-score band
when thresholds are available. **Drag a rectangle** over the scatter to see
every patch inside it as an image grid below (same 32-patch cap as the
Bragg-score histogram).

Comparing the two PCA tabs side by side is informative on its own: if the
handcrafted-feature PCA separates bands cleanly while the SSL PCA does not
(or vice versa), that is a direct, visual answer to "does the learned
representation capture what the hand-designed descriptors already capture,
more, or less?"

### On the Bragg-score bands

**They are not phase labels.** They are a per-dataset relative reading of a
continuous FFT score (see "Algorithm: Bragg-score crystallinity scoring"
above): "High" means "high relative to this dataset's own score
distribution," not "confirmed crystalline." Treat them as a viewer-side
exploration aid, not as ground truth.

The viewer degrades in two independent steps, each announced in the sidebar
rather than silently dropped:

- No `bragg_ratio` column (a dataset prepared before this scoring was
  added): the Bragg-score tab has nothing to show at all -- no histogram,
  no bands.
- `bragg_ratio` is present but `crystallinity_probe.csv` is missing, empty,
  unparseable, or its thresholds are non-finite or inverted: the histogram
  and per-patch scores still render, since those come from the manifest
  alone; only the High/Ambiguous/Low bands, the band filter, and the PCA
  colouring are unavailable, because those need `analyze`'s thresholds.

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
