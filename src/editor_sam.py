"""
Precise image editing using SAM (Segment Anything) + Stable Diffusion Inpainting.

This approach:
1. Uses a VLM to identify what region to edit (get bounding box or description)
2. Uses SAM to segment that region precisely
3. Uses SD Inpainting to fill/modify the region

Much more precise than InstructPix2Pix for feature removal/modification.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)


@dataclass
class EditRegion:
    """Represents a region to edit in an image."""
    mask: Image.Image  # Binary mask (white = edit region)
    description: str   # What's in this region
    bbox: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2


class SAMInpaintEditor:
    """
    Uses SAM for segmentation and SD Inpainting for editing.

    More precise than InstructPix2Pix for:
    - Removing specific features (ears, tail, etc.)
    - Modifying specific regions
    - Adding elements to specific locations
    """

    def __init__(
        self,
        sam_model: str = "facebook/sam-vit-base",
        inpaint_model: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.sam_model_name = sam_model
        self.inpaint_model_name = inpaint_model

        self._sam_pipeline = None
        self._inpaint_pipeline = None
        self._sam_loaded = False
        self._inpaint_loaded = False

    def _ensure_sam(self):
        """Lazy load SAM model."""
        if self._sam_loaded:
            return

        from transformers import SamModel, SamProcessor

        logger.info("Loading SAM: %s", self.sam_model_name)
        self._sam_model = SamModel.from_pretrained(self.sam_model_name).to(self.device)
        self._sam_processor = SamProcessor.from_pretrained(self.sam_model_name)
        self._sam_loaded = True
        logger.info("SAM loaded")

    def _ensure_inpaint(self):
        """Lazy load inpainting model."""
        if self._inpaint_loaded:
            return

        from diffusers import StableDiffusionInpaintPipeline

        logger.info("Loading Inpainting: %s", self.inpaint_model_name)
        self._inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.inpaint_model_name,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        ).to(self.device)
        self._inpaint_pipeline.set_progress_bar_config(disable=True)
        self._inpaint_loaded = True
        logger.info("Inpainting model loaded")

    def offload(self):
        """Move models to CPU to free VRAM."""
        if self._sam_loaded:
            self._sam_model.to("cpu")
        if self._inpaint_loaded:
            self._inpaint_pipeline.to("cpu")
        torch.cuda.empty_cache()

    def load_to_gpu(self):
        """Move models back to GPU."""
        if self._sam_loaded:
            self._sam_model.to(self.device)
        if self._inpaint_loaded:
            self._inpaint_pipeline.to(self.device)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_with_points(
        self,
        image: Image.Image,
        points: list[tuple[int, int]],
        point_labels: list[int] | None = None,
    ) -> Image.Image:
        """
        Segment a region using point prompts.

        Args:
            image: Input image
            points: List of (x, y) coordinates
            point_labels: 1 for foreground, 0 for background (default: all foreground)

        Returns:
            Binary mask as PIL Image
        """
        self._ensure_sam()

        if point_labels is None:
            point_labels = [1] * len(points)

        inputs = self._sam_processor(
            image,
            input_points=[points],
            input_labels=[point_labels],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        masks = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        # Take the best mask
        mask_np = masks[0][0, 0].numpy().astype(np.uint8) * 255
        return Image.fromarray(mask_np)

    def segment_with_box(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> Image.Image:
        """
        Segment a region using a bounding box.

        Args:
            image: Input image
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            Binary mask as PIL Image
        """
        self._ensure_sam()

        inputs = self._sam_processor(
            image,
            input_boxes=[[list(bbox)]],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        masks = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        mask_np = masks[0][0, 0].numpy().astype(np.uint8) * 255
        return Image.fromarray(mask_np)

    def auto_segment_region(
        self,
        image: Image.Image,
        region_hint: str,
        vlm_analyzer=None,
    ) -> Image.Image | None:
        """
        Automatically segment a region based on a text description.
        Uses the VLM to identify the region, then SAM to segment it.

        Args:
            image: Input image
            region_hint: Text description of what to segment (e.g., "the cat's ears")
            vlm_analyzer: Optional VLM analyzer for getting region coordinates

        Returns:
            Binary mask or None if region not found
        """
        # For now, use a simple heuristic approach
        # In a full implementation, we'd use the VLM to get bounding boxes

        # Use center point as a fallback
        w, h = image.size
        center_point = (w // 2, h // 2)

        return self.segment_with_points(image, [center_point])

    # ------------------------------------------------------------------
    # Inpainting
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> Image.Image:
        """
        Inpaint a masked region with the given prompt.

        Args:
            image: Original image
            mask: Binary mask (white = inpaint region)
            prompt: What to generate in the masked region
            negative_prompt: What to avoid
            num_steps: Diffusion steps
            guidance_scale: How strongly to follow the prompt
            seed: Random seed

        Returns:
            Inpainted image
        """
        self._ensure_inpaint()

        # Resize to 512x512 for SD inpainting
        original_size = image.size
        image_resized = image.convert("RGB").resize((512, 512), Image.LANCZOS)
        mask_resized = mask.convert("L").resize((512, 512), Image.NEAREST)

        # Dilate mask slightly for better blending
        mask_resized = mask_resized.filter(ImageFilter.MaxFilter(5))

        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self._inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        edited = result.images[0]
        return edited.resize(original_size, Image.LANCZOS)

    def remove_region(
        self,
        image: Image.Image,
        mask: Image.Image,
        fill_prompt: str = "background, seamless continuation",
        seed: int = 42,
    ) -> Image.Image:
        """
        Remove a masked region by inpainting with background.

        Args:
            image: Original image
            mask: Binary mask of region to remove
            fill_prompt: What to fill the region with
            seed: Random seed

        Returns:
            Image with region removed
        """
        return self.inpaint(
            image, mask, fill_prompt,
            negative_prompt="object, distinct feature, artifact",
            seed=seed,
        )

    # ------------------------------------------------------------------
    # High-level edit operations
    # ------------------------------------------------------------------

    def edit_with_instruction(
        self,
        image: Image.Image,
        instruction: str,
        seed: int = 42,
        vlm_analyzer=None,
    ) -> Image.Image:
        """
        Apply an edit instruction using SAM + inpainting.

        Parses the instruction to determine:
        - What region to target
        - Whether to remove, modify, or add
        - What to fill with

        Args:
            image: Input image
            instruction: Natural language edit instruction
            seed: Random seed
            vlm_analyzer: Optional VLM for better region detection

        Returns:
            Edited image
        """
        instruction_lower = instruction.lower()

        # Determine edit type from instruction
        is_removal = any(word in instruction_lower for word in
                        ["remove", "delete", "erase", "hide", "eliminate"])
        is_addition = any(word in instruction_lower for word in
                         ["add", "insert", "place", "put", "include"])
        is_change = any(word in instruction_lower for word in
                       ["change", "make", "turn", "convert", "replace"])

        # For removal: segment the target and inpaint with background
        if is_removal:
            # Try to extract what to remove
            # e.g., "remove the cat's ears" -> segment ears, inpaint background
            mask = self.auto_segment_region(image, instruction, vlm_analyzer)
            if mask:
                return self.remove_region(image, mask, seed=seed)

        # For changes: segment region and inpaint with new content
        if is_change:
            mask = self.auto_segment_region(image, instruction, vlm_analyzer)
            if mask:
                # Extract what to change to from instruction
                fill_prompt = instruction  # Use full instruction as prompt
                return self.inpaint(image, mask, fill_prompt, seed=seed)

        # Fallback: create a center region mask and apply the edit
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        # Draw an ellipse in the center third of the image
        draw.ellipse(
            [w // 3, h // 3, 2 * w // 3, 2 * h // 3],
            fill=255,
        )

        return self.inpaint(image, mask, instruction, seed=seed)


class HybridEditor:
    """
    Combines InstructPix2Pix (for global edits) and SAM+Inpaint (for precise edits).

    Chooses the best approach based on the edit instruction.
    """

    def __init__(self, device: str = "cuda", dtype: str = "float16"):
        self.device = device
        self.dtype = dtype
        self._pix2pix = None
        self._sam_inpaint = None

    def _ensure_pix2pix(self):
        if self._pix2pix is None:
            from .editor import ImageEditor
            self._pix2pix = ImageEditor(device=self.device, dtype=self.dtype)

    def _ensure_sam_inpaint(self):
        if self._sam_inpaint is None:
            self._sam_inpaint = SAMInpaintEditor(device=self.device, dtype=self.dtype)

    def offload(self):
        if self._pix2pix:
            self._pix2pix.offload()
        if self._sam_inpaint:
            self._sam_inpaint.offload()

    def load_to_gpu(self):
        if self._pix2pix:
            self._pix2pix.load_to_gpu()
        if self._sam_inpaint:
            self._sam_inpaint.load_to_gpu()

    def edit(
        self,
        image: Image.Image,
        instruction: str,
        seed: int = 42,
        use_sam: bool | None = None,
    ) -> Image.Image:
        """
        Apply edit instruction, automatically choosing the best method.

        Args:
            image: Input image
            instruction: Edit instruction
            seed: Random seed
            use_sam: Force SAM+inpaint (None = auto-detect)

        Returns:
            Edited image
        """
        instruction_lower = instruction.lower()

        # Auto-detect whether to use SAM+inpaint
        if use_sam is None:
            # Use SAM for precise removal/modification
            use_sam = any(word in instruction_lower for word in [
                "remove", "delete", "erase",  # Removal
                "cut", "crop",                # Cutting
                "segment", "isolate",         # Segmentation
            ])

        if use_sam:
            self._ensure_sam_inpaint()
            return self._sam_inpaint.edit_with_instruction(image, instruction, seed=seed)
        else:
            self._ensure_pix2pix()
            return self._pix2pix.edit(image, instruction, seed=seed)
