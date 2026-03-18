"""
Download all required datasets and models for Coralation.
Run this once before running the analysis to ensure everything is cached.

Models are loaded and then offloaded from GPU to free VRAM for the next model,
matching the behavior of the analysis pipeline.
"""
import gc
import sys

import torch


def offload():
    """Free GPU memory after loading a model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("Coralation Model Downloader")
    print("=" * 60)

    success = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. ImageNet dataset
    print("\n[1/6] Downloading ImageNet-1k dataset (this may take a while)...")
    print("      Note: You must accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    try:
        from datasets import load_dataset
        from src.config import load_hf_token
        token = load_hf_token()
        if not token:
            print("      ERROR: No HuggingFace token found!")
            print("      Set HF_TOKEN env var or add token to .token file")
            success = False
        else:
            # Download full dataset (not streaming) so it's cached
            ds = load_dataset("ILSVRC/imagenet-1k", split="validation", token=token)
            print(f"      OK: Dataset downloaded ({len(ds)} samples)")
            del ds
    except Exception as e:
        print(f"      ERROR: {e}")
        if "gated" in str(e).lower() or "403" in str(e):
            print("      FIX: Accept dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        success = False

    # 2. ResNet-50
    print("\n[2/6] Downloading ResNet-50 classifier...")
    try:
        from torchvision import models
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.to(device)
        print("      OK: ResNet-50 loaded to", device)
        del model
        offload()
        print("      OK: Offloaded from GPU")
    except Exception as e:
        print(f"      ERROR: {e}")
        success = False

    # 3. Qwen VLM
    print("\n[3/6] Downloading Qwen2.5-VL-7B vision-language model (~15GB)...")
    print("      This may take a while...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print("      OK: Processor loaded")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
        ).to(device)
        print("      OK: Qwen VLM loaded to", device)
        del model, processor
        offload()
        print("      OK: Offloaded from GPU")
    except Exception as e:
        print(f"      ERROR: {e}")
        if "disk" in str(e).lower() or "space" in str(e).lower():
            print("      FIX: Free up disk space (need ~15GB)")
        success = False

    # 4. Flux2 Klein KV (primary editor)
    print("\n[4/6] Downloading Flux2-Klein-9B-KV image editor (~18GB)...")
    print("      Note: You must accept terms at https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv")
    try:
        from diffusers import Flux2KleinKVPipeline

        pipe = Flux2KleinKVPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9b-kv",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        print("      OK: Flux2-Klein-KV loaded with CPU offload")
        del pipe
        offload()
        print("      OK: Offloaded from GPU")
    except ImportError as e:
        if "Flux2KleinKVPipeline" in str(e):
            print("      ERROR: Flux2KleinKVPipeline not available!")
            print("      FIX: Run 'uv pip install git+https://github.com/huggingface/diffusers.git'")
        else:
            print(f"      ERROR: {e}")
        success = False
    except Exception as e:
        print(f"      ERROR: {e}")
        if "gated" in str(e).lower() or "403" in str(e):
            print("      FIX: Accept model terms at https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv")
        success = False

    # 5. Qwen-Image-Edit (backup)
    print("\n[5/6] Downloading Qwen-Image-Edit image editor (~40GB)...")
    print("      This is a large model and may take a while...")
    print("      Using FP8 + group offloading to reduce VRAM usage...")
    try:
        from diffusers import QwenImageEditPipeline
        from diffusers.hooks import apply_group_offloading

        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
        )

        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")

        # FP8 storage with bfloat16 compute
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )

        # Group offloading
        pipe.transformer.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )

        apply_group_offloading(
            pipe.text_encoder,
            onload_device=onload_device,
            offload_type="block_level",
            num_blocks_per_group=2,
        )

        apply_group_offloading(
            pipe.vae,
            onload_device=onload_device,
            offload_type="leaf_level",
        )

        print("      OK: Qwen-Image-Edit loaded with FP8 + group offload")
        del pipe
        offload()
        print("      OK: Offloaded from GPU")
    except ImportError as e:
        if "QwenImageEditPipeline" in str(e):
            print("      WARNING: QwenImageEditPipeline not available (optional)")
        else:
            print(f"      WARNING: {e}")
    except Exception as e:
        print(f"      WARNING: {e}")
        print("      (This is optional if using Flux2)")

    # 6. InstructPix2Pix (fallback)
    print("\n[6/6] Downloading InstructPix2Pix (fallback editor, ~5GB)...")
    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
        ).to(device)
        print("      OK: InstructPix2Pix loaded to", device)
        del pipe
        offload()
        print("      OK: Offloaded from GPU")
    except Exception as e:
        print(f"      WARNING: {e}")
        print("      (This is optional - Qwen-Image-Edit is the primary editor)")

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("All models downloaded successfully!")
        print("You can now run: uv run python main.py")
    else:
        print("Some downloads FAILED. Fix the errors above and retry.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
