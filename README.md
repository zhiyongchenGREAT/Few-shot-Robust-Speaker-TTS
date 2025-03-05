# Few-shot Robust Speaker Recognition with TTS

## Overview

Speaker recognition is crucial for secure authentication and personalized voice assistants in smart home environments. However, intrinsic speaker variability—such as aging and emotional fluctuations—poses significant challenges. Traditional approaches rely heavily on pretraining and large datasets, limiting adaptability to dynamic conditions.

To address these issues, we introduce a novel framework for time-varying and emotion-robust open-set identification (OSI). Our approach leverages:

- **Few-shot foundation model tuning** at enrollment time.
- **Style-rich zero-shot text-to-speech (TTS) synthesis** to augment training data.
- **Optimized synthetic data selection strategies** and **open-set loss functions** to improve generalization and robustness.

Our method enhances resilience against both intrinsic speaker variability and unknown outliers, demonstrating strong performance across multiple emotionally diverse and time-varying benchmarks.

## Code

Run the training script:

```bash
python osr_spk_eres.py
```

## Datasets

Our training and testing datasets are available on Hugging Face:

[![Hugging Face](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

[Robust Speaker Recognition OSI with TTS](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

## Pretrained Weights

Access pretrained weights from our Hugging Face repository:

[![Hugging Face](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

[Robust Speaker Recognition OSI with TTS](https://huggingface.co/datasets/zhiyongchen/robust_speaker_recognition_OSI_with_TTS)

## Note

### Dataset Directory Structure

The training and testing datasets are organized as follows:

- #### enrollment finetune embeddings (ERes2NetV2)

| Speaker IDs | Description                |
|-------------|----------------------------|
| 0 - 4       | Target speakers            |
| 5 - 16      | Synthetic unknown speakers |

- #### test embeddings (ERes2NetV2)

| Speaker IDs | Description       |
|-------------|-------------------|
| 0 - 4       | Target speakers   |
| > 16        | Outliers          |



## Citation

If you use our work, please cite:

```
@article{YourPaperReference,
  title={Towards Robust Speaker Recognition against Intrinsic Variation with Foundation Model Few-shot Tuning and Effective Speech Synthesis},
  journal={Interspeech 2025},
  year={2025}
}
```

---

This project aims to push the boundaries of speaker recognition in real-world dynamic environments. Contributions and feedback are welcome!
