# Few-shot Robust Speaker Recognition with TTS
Towards Robust Speaker Recognition against Intrinsic Variation with Foundation Model Few-shot Tuning and Effective Speech Synthesis
## Overview

Speaker recognition is a cornerstone for secure authentication and personalized voice assistants in smart home environments. However, intrinsic speaker variability—such as aging and emotional fluctuations—poses significant challenges. Traditional approaches, which rely heavily on pretraining and large datasets, often struggle to adapt to dynamic conditions.

To overcome these limitations, we introduce a novel framework for time-varying and emotion-robust open-set identification (OSI). Our approach leverages:

- **Few-shot foundation model tuning** at enrollment, enabling rapid adaptation with limited data.
- **Style-rich zero-shot text-to-speech (TTS) synthesis** to augment training data with diverse speech characteristics.
- **Optimized synthetic data selection strategies** and **open-set SpeakerRPL loss training** to enhance generalization and robustness against both intrinsic variability and unknown outliers.

Our method demonstrates strong performance across multiple emotionally diverse and time-varying benchmarks, pushing the boundaries of robust speaker recognition in real-world scenarios.

## Code

Run the few-shot training script for each evaluation split:

```bash
python osr_spk_eres.py --loss SpeakerRPL --finetune_data_split {} --evaluation_data_split {}
```

Compare and evaluate with *direct* enrollment and cosine simlarity scoring. Refer to:

```
inference_for_direct_baseline.ipynb
```

## Datasets

### Training Data

Our training and testing datasets are available on Hugging Face:

<a href="https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="30" />
</a>

[Robust Speaker Recognition OSI with TTS](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

### Pretrained Weights

Access the pretrained weights:

<a href="https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="30" />
</a>

[Robust Speaker Recognition OSI with TTS](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

## Dataset Directory Structure

The datasets are organized as follows:

### Enrollment Finetune Embeddings for all splits (ERes2NetV2)

| Speaker IDs | Description                |
|-------------|----------------------------|
| 0 - 4       | Target speakers            |
| > 4        | Synthetic unknown speakers |

### Test Embeddings for all splits (ERes2NetV2)

| Speaker IDs | Description      |
|-------------|------------------|
| 0 - 4       | Target speakers  |
| > 4        | Outlier speakers |

*Note:* The mapping between speaker IDs and their corresponding categories differs between the enrollment and test splits.

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
