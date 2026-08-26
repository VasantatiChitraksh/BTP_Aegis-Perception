from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class DiffusionSettings:
    model_id: str
    revision: str | None = None
    device: str = "cuda"
    strength: float = 0.25
    guidance_scale: float = 5.0
    num_inference_steps: int = 20


class DiffusionWeatherGenerator:
    """Optional image-to-image weather synthesis adapter.

    Diffusion may alter object geometry or identity. Outputs must pass annotation
    consistency checks and must never be used as the paper's held-out test set.
    """

    def __init__(self, settings: DiffusionSettings) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            raise RuntimeError(
                "Install the diffusion extra: pip install -e '.[diffusion]'"
            ) from exc

        if not 0.0 < settings.strength <= 0.5:
            raise ValueError("Use strength in (0, 0.5] to limit structural drift")
        self.settings = settings
        dtype = torch.float16 if settings.device.startswith("cuda") else torch.float32
        load_options = {"torch_dtype": dtype, "use_safetensors": True}
        if settings.revision:
            load_options["revision"] = settings.revision
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            settings.model_id, **load_options
        )
        self.pipeline.to(settings.device)

    def generate(self, image: Image.Image, *, condition: str, seed: int) -> Image.Image:
        import torch

        prompts = {
            "fog": "the exact same driving scene in dense natural fog, preserve every object",
            "rain": "the exact same driving scene in heavy rain, preserve every object",
            "snow": "the exact same driving scene during snowfall, preserve every object",
            "smoke": "the exact same driving scene with drifting smoke, preserve every object",
        }
        if condition not in prompts:
            raise ValueError(f"Unsupported diffusion condition: {condition}")
        generator_device = (
            self.settings.device if self.settings.device.startswith("cuda") else "cpu"
        )
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        output = self.pipeline(
            prompt=prompts[condition],
            negative_prompt=(
                "new objects, missing objects, moved vehicles, changed road geometry, text"
            ),
            image=image.convert("RGB"),
            strength=self.settings.strength,
            guidance_scale=self.settings.guidance_scale,
            num_inference_steps=self.settings.num_inference_steps,
            generator=generator,
        )
        return output.images[0]
