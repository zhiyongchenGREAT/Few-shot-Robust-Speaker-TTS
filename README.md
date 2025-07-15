# Few-shot Robust Speaker Recognition with TTS  
*Towards Robust Speaker Recognition against Intrinsic Variation with Foundation Model Few-shot Tuning and Effective Speech Synthesis*

## Overview

Speaker recognition is a cornerstone for secure authentication and personalized voice assistants in smart home environments. However, intrinsic speaker variability—such as aging and emotional fluctuations—poses significant challenges. Traditional approaches, which rely heavily on pretraining and large datasets, often struggle to adapt to dynamic conditions.

To overcome these limitations, we introduce a novel framework for time-varying and emotion-robust open-set identification (OSI). Our approach leverages:

- **Few-shot foundation model tuning** at enrollment, enabling rapid adaptation with limited data.
- **Style-rich zero-shot text-to-speech (TTS) synthesis** to augment training data with diverse speech characteristics.
- **Optimized synthetic data selection strategies** and **open-set SpeakerRPL loss training** to enhance generalization and robustness against both intrinsic variability and unknown outliers.

Our method demonstrates strong performance across multiple emotionally diverse and time-varying benchmarks, pushing the boundaries of robust speaker recognition in real-world scenarios.

## SpeakerRPL

This repository hosts the ongoing development of the SpeakerRPL (Speaker Reciprocal Points Learning) for robust speaker recognition.

- **SpeakerRPL V1**: The first version of SpeakerRPL. For details, please see our other repository: [SpeakerRPL](https://github.com/zhiyongchenGREAT/speaker-reciprocal-points-learning)
- **Enhanced SpeakerRPL (This Repository)**: This version includes improvements of optimized synthetic data selection strategies for time-varying and emotion-robust open-set identification (OSI).
- **🎉 SpeakerRPL V2 (Latest Version)**: The latest version featuring model fusion and further enhancements. The implementation is in `loss/SpeakerRPLv2.py` and can be run using `osr_spk_eres_fusion.py`.

## Code

Run the few-shot training script for each evaluation split:

```bash
python osr_spk_eres.py --loss SpeakerRPL --finetune-data-split {} --evaluation-data-split {}
```

Other loss functions can be tried by varying the *loss* parameter.

Compare and evaluate with direct enrollment and cosine similarity scoring. Refer to:

```bash
inference_for_direct_baseline.ipynb
```

Example code for implementing a controlling policy for synthesis data selection (for demonstration purposes):

```bash
syn_select_controlling_policy.ipynb
```

*Note*: A sufficient number of new speaker samples and utterances per speaker (for both unknown and target speakers) should be synthesized and sampled using the above controlling policy.

## Datasets

### Core Training Split Data

Our training and testing datasets are available on Hugging Face:

<a href="https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="30" />
</a>

[Robust Speaker Recognition OSI with TTS](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

We provide augmented datasets (in ERes2Net embedding format) generated using multiple data augmentation and sampling strategies, as described in the paper. These datasets are directly usable with the training script and include evaluation splits for all four benchmark datasets. We also provide the original, unaugmented, and unarranged waveforms for each speaker to facilitate further research.

| Dataset                       | Description                                                        |
|-------------------------------|--------------------------------------------------------------------|
| **ESD**                     | Strong emotion variation                                           |
| **IEMOCAP**                 | Strong emotion variation and wild-collected data                   |
| **Voxceleb2(test)-Voxwatch**  | 110+ speakers in the wild                                          |
| **SpeakerAging**            | 15+ speakers with short-term variation (1-year timespan)            |

### Pretrained Weights

Access the pretrained weights of one split for ESD and Voxwatch for demonstration:

<a href="https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="30" />
</a>

[Robust Speaker Recognition OSI with TTS](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

```bash
osr_spk_eres.py --loss SpeakerRPL --model-path {} --evaluation-data-split {}
```

## Dataset Directory Structure (Embeddings for Experiments)

The embedding datasets <a href="https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="30" />
</a> for each split are organized as follows:

### Enrollment Finetuning Embeddings for Each Split (ERes2NetV2)

| Speaker IDs | Description                |
|-------------|----------------------------|
| 0 - 4       | Target speakers            |
| > 4        | Synthetic unknown speakers |

### Test Embeddings for Each Split (ERes2NetV2)

| Speaker IDs | Description      |
|-------------|------------------|
| 0 - 4       | Target speakers  |
| > 4        | Outlier speakers |

*Note:* The mapping between speaker IDs in each experimental split and their corresponding true IDs is different.

## Citation

If you use our work, please cite:

```
@article{ThisPaperReference,
  title={Towards Robust Speaker Recognition against Intrinsic Variation with Foundation Model Few-shot Tuning and Effective Speech Synthesis},
  journal={Interspeech 2025},
  year={2025}
}
```

---

Contributions and feedback are welcome!
