from pathlib import Path

import torch

from diffusers import DiffusionPipeline
from PIL import Image


def save_images(image, filename):
    image.save(f"{filename}.png")
    print(f"Saved to: {filename}.png")


def log(message: str) -> None:
    print("="*40)
    print(message)
    print("="*40)


def main(prompt: str, filename: Path):
    log("Load base and refiner")

    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    # allow CPU offloading if GPU vRAM is too small or not available
    base.enable_model_cpu_offload()

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    # allow CPU offloading if GPU vRAM is too small or not available
    refiner.enable_model_cpu_offload()

    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    log(f"Prompt used: {prompt}")

    # run both experts
    base_result = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images[0]

    refiner_result = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=base_result,
    ).images[0]

    log("Save refiner result ...")
    save_images(refiner_result, filename)
    log("Refiner result saved.")


if __name__ == "__main__":
    prompt = input("Prompt: ")

    if not prompt.strip():
        prompt = "A majestic lion jumping from a big stone at night"

    filename = input("Filename (without extension): ")

    if not filename.strip():
        filename = Path("majestic_lion")
    else:
        filename = Path(filename)

    main(prompt, filename)
