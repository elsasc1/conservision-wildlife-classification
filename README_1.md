# Conser-vision — Wildlife species classification from camera traps

Classifying camera-trap images from Taï National Park (Côte d'Ivoire) into 7 animal groups plus **blank** (no animal).

**Public leaderboard: log loss 0.9224 — ranked 35th** · October 2026
Competition page: [DrivenData — Conser-vision Practice Area](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/)

## The problem

- 16,488 training images and 4,464 test images, with 8 mutually exclusive classes: `antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, `rodent`.
- Metric: multi-class **log loss**. It heavily penalises confident mistakes, so calibration matters as much as accuracy.
- **The key difficulty is domain shift.** Every image comes from a fixed camera *site*, and **no test site appears in the training set**. A model can learn the background instead of the animal: random cross-validation would reward this, and the test set would punish it.
- Rules: no external data. Public pre-trained models are allowed.

## Approach

All code runs on a **free Google Colab T4 GPU**. Every long step is cached on Google Drive (data, resized images, embeddings, per-fold predictions), so a disconnected session resumes where it stopped.

1. **Site-grouped validation.** `StratifiedGroupKFold` over camera sites, with 5 folds. Every validation fold contains only unseen cameras, as in the test set. The same folds are used for every model.
2. **Fast preprocessing.** Images are resized once to a 640 px long side, using JPEG draft-mode decoding. This removes the CPU bottleneck when feeding the GPU.
3. **Model A: frozen foundation-model embeddings + logistic regression.**
   - Embeddings from **DINOv2 ViT-B/14** and **CLIP ViT-L/14**, with horizontal-flip test-time augmentation.
   - Standardised features, multinomial logistic regression, C chosen on out-of-fold (OOF) log loss.
   - It is robust to new sites, because the backbones have never seen these cameras.
4. **Model B: fine-tuned ConvNeXt-Tiny** (ImageNet-22k pre-trained).
   - 288×512 input, to keep small animals visible and preserve the landscape aspect ratio.
   - Augmentations: random resized crops (60–100%, so the animal is not cut out), horizontal flip, colour jitter, and **random grayscale** to mimic night-time infrared images.
   - AdamW, one-epoch warmup then cosine decay, mixed precision. The best epoch is selected on validation log loss, and test predictions use flip TTA.
5. **Blending + temperature scaling.**
   - A and B are averaged with a weight w chosen on OOF predictions. If only some folds of B are trained, w is chosen on the images those folds cover.
   - A single temperature T is then fitted on OOF log-probabilities to correct over- or under-confidence.

## Results

| Submission | Validation (site-grouped OOF) | Public leaderboard |
|---|---|---|
| Model A only (DINOv2 + CLIP embeddings, calibrated) | log loss ≈ 1.0 (0.93–1.15 depending on the fold) | – |
| **Model A + Model B (2 of 5 folds trained), blended and calibrated** | – | **0.9224 (35th)** |

Error analysis on the out-of-fold predictions of model A:
- **Rodents are the hardest class** (32% recall). They are confused with `blank` and `civet_genet`: they are small and mostly seen at night.
- Birds reach 58% recall, and many missed animals are predicted as `blank`.
- These are exactly the cases where higher resolution and fine-tuning help, which is why model B improves the score.

## Next steps

- Train the remaining 3 folds of model B. The folds resume automatically, one per Colab session.
- Raise the resolution to 384×672 for small animals, and add a second fine-tuned backbone to the blend.

## How to run

1. Open `conservision_colab.ipynb` in Google Colab, then select **Runtime → Change runtime type → T4 GPU**.
2. **Run all.** On the first run, the notebook asks for the download link of `competition.zip`, copied from the competition's data page. The data is then cached on your Google Drive.
3. The notebook writes `submission_A.csv` (model A) and `submission_final.csv` (blend).

Data is not included in this repository: the competition rules forbid redistributing it. The notebook's comments and explanations are in French.

## Data citation

The Pan African Programme: The Cultured Chimpanzee, Wild Chimpanzee Foundation, DrivenData. (2022). *Conser-vision Practice Area: Image Classification.* Retrieved October 2026 from https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/.
