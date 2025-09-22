# Few-shot Robust Speaker Recognition with TTS  
*SpeakerRPL V2: Robust Open-set Speaker Identification through Enhanced Few-shot Foundation Tuning and Model Fusion*

## 📖 Overview

Speaker recognition is a cornerstone for secure authentication and personalized voice assistants in smart home environments. To further improve robustness, we propose a model fusion strategy and a novel model selection policy that identifies the most suitable candidate models for fusion, ultimately enhancing robustness and improving open-set speaker identification performance.

## 🧠 Key Contributions

- **Improved Open-set Learning Objective** integrating reciprocal points learning, logit normalization, and adaptive anchor learning to yield robust speaker embeddings.
- **Effective Model Fusion Strategy** to reduce inherent randomness and enhance generalization.
- **Post-tuning Model Selection Policy** to preserve the most effective tuned models, thus ensuring optimal fusion performance.

Comprehensive experiments of SpeakerRPL V2 on diverse speaker recognition benchmarks validate the effectiveness and reliability of the proposed method across different scenarios.

## 🧩 SpeakerRPL Versions

This repository hosts the ongoing development of the **SpeakerRPL (Speaker Reciprocal Points Learning)** framework for robust speaker recognition.

| Version                      | Description                                                        |
|-------------------------------|--------------------------------------------------------------------|
| **SpeakerRPL V1**             | Initial version. See: [SpeakerRPL](https://github.com/zhiyongchenGREAT/speaker-reciprocal-points-learning)                                         |
| **Enhanced SpeakerRPL**       | Includes optimized synthetic data selection for time/emotion-robust open-set identification (OSI). |
| **🎉 SpeakerRPL V2**          | The latest version featuring **optimal open-set learning**, **model fusion** and **enhanced selection policies**. Implemented in `loss/SpeakerRPLv2.py`.|

## 🚀 Quick Start

### Run few-shot training:

- **SpeakerRPL V1:**
```bash
python osr_spk_eres.py --loss SpeakerRPL --finetune-data-split {} --evaluation-data-split {}
```

- **SpeakerRPL V2:**
```bash
python osr_spk_eres_fusion.py --loss SpeakerRPLv2 ----split-id {}
```

Other loss functions can be tried by varying the *loss* parameter.

### Evaluate with Direct Enrollment & Cosine Scoring:

```bash
inference_for_direct_baseline.ipynb
```

### Synthetic Data Selection (Demo):

```bash
syn_select_controlling_policy.ipynb
```

*Note*: A sufficient number of new speaker samples and utterances per speaker (for both unknown and target speakers) should be synthesized and sampled using the above controlling policy.

## 📂 Datasets

### Core Training & Testing Data

Our training and testing datasets are available on Hugging Face:

<a href="https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="30" />
</a>

[Robust Speaker Recognition OSI with TTS](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

We provide datasets (in ERes2Net embedding format) generated using multiple data augmentation and sampling strategies, as described in the paper. These datasets are directly usable with the training script and include evaluation splits for all four benchmark datasets. We also provide the original waveforms of VoxCeleb2(test) dataset for each split to facilitate further research.

| Dataset                       | Description                                                        |
|-------------------------------|--------------------------------------------------------------------|
| **VoxCeleb2(test)**                     | 110+ speakers in the wild                                           |
| **3D-Speaker**                 | dataset aross dialect/distance/device                   |
| **ESD**  | strong emotion variation                                    |
| **Vox1-O(revised)**            | revised from vox1-o trial list of 40 speakers            |

⚠️ *Note:* Vox1-O(revised) trial list can be found in `vox1-o_sid.txt`, revised by `vox1-o_revise.ipynb`.



## 🗂 Dataset Directory Structure (Embeddings for Experiments)

The embedding datasets <a href="https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="30" />
</a> for each split are organized as follows:

- **Enrollment Finetuning Embeddings for Each Split (ERes2NetV2)**

| Speaker IDs | Description                |
|-------------|----------------------------|
| 0 - 4       | Target speakers            |
| > 4        | Synthetic unknown speakers |

- **Test Embeddings for Each Split (ERes2NetV2)**

| Speaker IDs | Description      |
|-------------|------------------|
| 0 - 4       | Target speakers  |
| > 4        | Outlier speakers |

⚠️ *Note:* The mapping between speaker IDs in each experimental split and their corresponding true IDs is different.

## Citation

If you use our work, please cite:

```
@article{ThisPaperReference,
  title={SpeakerRPL V2: Robust Open-set Speaker Identification through Enhanced Few-shot Foundation Tuning and Model Fusion},
  journal={ICASSP 2026},
  year={2026}
}
```

---

Contributions and feedback are welcome!
