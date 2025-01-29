import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from colocation.attn_processor import (
    CustomAttentionStoreClassPrompts,
    StoredAttnClassPromptsProcessor,
    register_attention_control,
)
from colocation.editing_pipeline import image_optimization
from colocation.pipeline_utils import load_512

import pandas as pd
from colocation.prompt_utils import find_prompt_difference, compute_target_index

MODEL_ID = "runwayml/stable-diffusion-v1-5"

def create_args():
    parser = argparse.ArgumentParser()

    # common config
    parser.add_argument("--lr", type=float, default=1)

    # stage 1 (generate)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--reg_scale", type=float, default=2e-2)
    parser.add_argument("--t_min", type=int, default=50)
    parser.add_argument("--t_max", type=int, default=950)

    # iterative
    parser.add_argument("--num_seed", type=int, default=1)

    # data io
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--save_step", type=int, default=30)
    parser.add_argument("--output_dir", type=str, required=True)

    # hypertune stuff
    parser.add_argument("--ds_high", type=float, default=0.15)
    parser.add_argument("--ds_low", type=float, default=0.01)
    parser.add_argument("--speed", type=float, default=5)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--threshold_decay_rate", type=float, default=0.99)

    parser.add_argument("--cross_attn_momentum", type=float, default=0.1)

    parser.add_argument("--init_seed", type=int, default=0)

    parser.add_argument('--no_direction_scale', dest='use_direction_scale', action='store_false')
    parser.set_defaults(use_direction_scale=True)

    return parser

def prepare_data(input_dir):
    def get_target_index(tokenizer, source_prompt, target_prompt):
        prior_text, diff_tokens, is_noun = find_prompt_difference(source_prompt, target_prompt)
        target_index = compute_target_index(tokenizer, prior_text, diff_tokens, is_noun)
        return target_index

    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    tokenizer = pipeline.tokenizer

    df = pd.read_csv(f"{input_dir}/index.csv")
    print(len(df))
    df = df.fillna("")
    df["img_path"] = df["img_path"].apply(lambda x: os.path.join(f"{input_dir}/images", x))
    df["target"] = None

    df["target_index"] = df.apply(
        lambda row: get_target_index(tokenizer, row["source_prompt"], row["target_prompt"]), axis=1
    )
    dataset = df.to_dict(orient="records")
    
    return dataset


if __name__ == "__main__":
    args = create_args().parse_args()
    dataset = prepare_data(args.input_dir)

    device = torch.device("cuda:0")
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(device)
    controller = CustomAttentionStoreClassPrompts(start=0, end=args.num_steps)
    register_attention_control(
        pipeline, controller, StoredAttnClassPromptsProcessor, use_self_attn=True,
    )

    output_dir = args.output_dir
    for n in range(args.num_seed):
        for k, data in enumerate(dataset):

            image = load_512(data["img_path"])
            source_prompt = data["source_prompt"]
            target_prompt = data["target_prompt"]
            target_object = data["target"]
            target_index = data["target_index"]

            print(
                f"Generating image {k} with prompt: {target_prompt} / neg prompt: {source_prompt} / target_object: {target_object}"
            )

            seed = args.init_seed + n
            torch.manual_seed(seed)

            sub_path = f"{str(k).zfill(3)}_seed{seed}"
            cache_dir = os.path.join(output_dir, ".cache", sub_path)
            save_path = os.path.join(output_dir, f"{sub_path}.png")
            
            controller.reset()
            output = image_optimization(
                pipeline, controller, image, source_prompt, target_prompt,
                num_iters=args.num_steps,
                reg_scale=args.reg_scale,
                learning_rate=args.lr,
                t_min=args.t_min,
                t_max=args.t_max,
                device=device,

                ######
                ds_high=args.ds_high,
                ds_low=args.ds_low,
                speed=args.speed,
                threshold=args.threshold,
                threshold_decay_rate=args.threshold_decay_rate,
                cross_attn_momentum=args.cross_attn_momentum,
                target_index=target_index,

                outdir=cache_dir,
                save_step=args.save_step,
                use_direction_scale=args.use_direction_scale,
            )

            os.makedirs(output_dir, exist_ok=True)
            output.save(save_path)

