# LUSD: Localized Update Score Distillation for Text-Guided Image Editing

### [Paper](https://arxiv.org/abs/2503.11054)

While diffusion models show promising results in image editing given a target prompt, achieving both prompt fidelity and background preservation remains difficult. Recent works have introduced score distillation techniques that leverage the rich generative prior of text-to-image diffusion models to solve this task without additional fine-tuning. However, these methods often struggle with tasks such as object insertion. Our investigation of these failures reveals significant variations in gradient magnitude and spatial distribution, making hyperparameter tuning highly input-specific or unsuccessful. To address this, we propose two simple yet effective modifications: attention-based spatial regularization and gradient filtering-normalization, both aimed at reducing these variations during gradient updates. Experimental results show our method outperforms state-of-the-art score distillation techniques in prompt fidelity, improving successful edits while preserving the background. Users also preferred our method over state-of-the-art techniques across three metrics, and by 58-64% overall.

## Installation

```shell
python -m venv ./envs/lusd
source envs/lusd/bin/activate
pip install -r requirements.txt
```

## Inference

### 1. Preparing inputs

Create a directory `<input_dir>` with the following structure:
```shell
<input_dir>
    - images
        - image1.jpg
        - image2.jpg
        ...
    index.csv
```
Prompts are specified in `index.csv`. Refer to `samples/` as an example.

### 2. Editing image

```shell
python run.py --input_dir <input_dir> --output_dir <output_dir>
```

## Evaluation
We adopt the evaluation code from [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush/tree/main/evaluation).
```shell
CUDA_VISIBLE_DEVICES=0 python eval_magicbrush.py \
    --generated_path <path_to_results> \
    --gt_path magicbrush/images \
    --caption_path magicbrush/local_descriptions.json \
    --global_caption_path magicbrush/global_descriptions.json \
    --save_path <path_to_outputs>
```
We provide code for computing a score summary in `eval.ipynb`.

## Citation

```
@misc{chinchuthakun2025lusd,
      title={LUSD: Localized Update Score Distillation for Text-Guided Image Editing}, 
      author={Worameth Chinchuthakun and Tossaporn Saengja and Nontawat Tritrong and Pitchaporn Rewatbowornwong and Pramook Khungurn and Supasorn Suwajanakorn},
      year={2025},
      eprint={2503.11054},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2503.11054}, 
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)