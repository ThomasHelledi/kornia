"""Diffusion img2img — AI-enhanced frame generation via SDXL-Turbo / FLUX.

Takes procedural frames from the Go engine and enhances them with neural detail
using latent diffusion img2img. Low strength (0.3-0.5) preserves the procedural
structure while adding photorealistic texture, lighting, and depth.

Architecture:
  Go engine → procedural RGBA frame → binary protocol → this effect
  → resize to model resolution → img2img (4 steps, LCM) → resize back → RGBA

Models (auto-selected by device):
  - SDXL-Turbo (2.3B): 4 steps native, no scheduler tricks needed
  - SD-Turbo (1.7B): Fastest, single step possible, lower quality
  - FLUX Schnell (12B): Best quality, needs more VRAM

Audio-reactive parameters:
  - strength varies with energy (calm=0.5 more AI, climax=0.25 preserve structure)
  - prompt switches per section type (intro→ambient, chorus→intense)
"""

import sys
import torch
import numpy as np

from .audio_reactive import AudioReactiveEngine


# Per-environment style prompts — maps atlas-synth environments to diffusion prompts
ENV_PROMPTS = {
    'spectral-mesh': 'iridescent wireframe grid floating in cosmic void, neon blue energy streams, holographic, sci-fi, 8k',
    'black-waters': 'dark underwater portal with teal and amber energy rings, bioluminescent, cinematic',
    'cymatics': 'vibrating sand patterns on a metal plate, macro photography, physics simulation, ultra detailed',
    'reaction-diffusion': 'organic morphogenesis patterns, biological cells dividing, microscope photography, nature',
    'cosmic-nebula': 'volumetric nebula gas clouds with stellar core, deep space photography, Hubble telescope',
    'cosmic-energy': 'plasma ball with electric arcs, lightning, tesla coil, high voltage, dark background',
    'aurora': 'northern lights over dark landscape, flowing curtains of green and purple light, long exposure',
    'neon-cathedral': 'cyberpunk corridor with neon signs, rain reflections, Blade Runner atmosphere, cinematic',
    'lava': 'volcanic lava flow, incandescent magma, dark rocks, extreme heat, cinematic lighting',
    'ocean': 'deep underwater scene with caustic light patterns, coral reef, bioluminescent creatures',
    'forest': 'ancient forest with volumetric light rays through canopy, moss, mystical atmosphere',
    'space': 'deep space with galaxies and nebulae, stars, cosmic dust, cinematic astrophotography',
    'tree': 'majestic ancient tree with golden leaves falling, volumetric light, magical atmosphere',
    'starry': 'Van Gogh starry night with swirling spirals, oil painting texture, impressionist',
    'prism': 'light refracting through crystal prism, rainbow spectrum, dark background, Pink Floyd',
}

DEFAULT_PROMPT = 'cinematic high quality visualization, ultra detailed, 8k, volumetric lighting'
NEGATIVE_PROMPT = 'blurry, low quality, text, watermark, logo, pixelated, jpeg artifacts, deformed'

# Section-type prompt modifiers
SECTION_MODIFIERS = {
    'intro': ', calm ambient atmosphere, gentle',
    'verse': ', moderate energy, flowing',
    'chorus': ', intense energy, dramatic, vibrant colors',
    'climax': ', explosive energy, maximum intensity, epic',
    'bridge': ', transitional, ethereal, dreamlike',
    'outro': ', fading, peaceful, serene',
    'breakdown': ', minimal, sparse, dark',
    'build': ', building tension, rising energy',
}


class DiffusionEffect:
    """AI-enhanced frame generation via latent diffusion img2img.

    Params:
        model: str — 'sdxl-turbo' | 'sd-turbo' | 'flux-schnell' (default: auto)
        strength: float — img2img denoising strength 0.0-1.0 (default: 0.35)
        steps: int — inference steps (default: 4)
        guidance_scale: float — CFG scale (default: 0.0 for turbo, 3.5 for flux)
        prompt: str — override prompt (default: auto from environment)
        env: str — environment name for auto-prompt lookup
        section: str — section type for prompt modifier
        resolution: int — model input resolution (default: 512 for turbo, 1024 for flux)

    Audio-reactive:
        bass: float — bass energy (0-1), scales strength inversely
        energy: float — overall energy, adjusts prompt intensity
        beat_pulse: float — beat indicator (0-1), triggers momentary strength boost
    """

    def __init__(self, device):
        self.device = device
        self.pipe = None
        self.model_name = None
        self.audio = AudioReactiveEngine()
        self._frame_count = 0

    def _load_model(self, model_name='auto'):
        """Lazy-load diffusion pipeline on first frame."""
        if self.pipe is not None and self.model_name == model_name:
            return

        if model_name == 'auto':
            # Auto-select based on device and VRAM
            if self.device.type == 'cuda':
                vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
                if vram_gb >= 12:
                    model_name = 'sdxl-turbo'
                else:
                    model_name = 'sd-turbo'
            elif self.device.type == 'mps':
                model_name = 'sdxl-turbo'  # Works on Apple Silicon unified memory
            else:
                model_name = 'sd-turbo'  # CPU fallback — smallest model

        sys.stderr.write(f'kornia-server: loading diffusion model {model_name}...\n')
        sys.stderr.flush()

        try:
            from diffusers import AutoPipelineForImage2Image
            import torch as th

            if model_name == 'sdxl-turbo':
                self.pipe = AutoPipelineForImage2Image.from_pretrained(
                    'stabilityai/sdxl-turbo',
                    torch_dtype=th.float16 if self.device.type == 'cuda' else th.float32,
                    variant='fp16' if self.device.type == 'cuda' else None,
                ).to(self.device)
                self._default_steps = 4
                self._default_guidance = 0.0  # Turbo models don't use CFG
                self._default_resolution = 512
            elif model_name == 'sd-turbo':
                self.pipe = AutoPipelineForImage2Image.from_pretrained(
                    'stabilityai/sd-turbo',
                    torch_dtype=th.float16 if self.device.type == 'cuda' else th.float32,
                    variant='fp16' if self.device.type == 'cuda' else None,
                ).to(self.device)
                self._default_steps = 4
                self._default_guidance = 0.0
                self._default_resolution = 512
            else:
                raise ValueError(f'Unknown model: {model_name}')

            # Optimize for speed
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()

            self.model_name = model_name
            sys.stderr.write(f'kornia-server: {model_name} loaded on {self.device}\n')
            sys.stderr.flush()

        except ImportError:
            sys.stderr.write('kornia-server: ERROR — diffusers not installed. Run: pip install diffusers transformers accelerate\n')
            sys.stderr.flush()
            self.pipe = None
        except Exception as e:
            sys.stderr.write(f'kornia-server: ERROR loading diffusion model: {e}\n')
            sys.stderr.flush()
            self.pipe = None

    def process(self, tensor, params):
        """Process a single frame through diffusion img2img.

        Args:
            tensor: (1, 3, H, W) float tensor [0, 1]
            params: dict with effect parameters

        Returns:
            (1, 3, H, W) float tensor [0, 1]
        """
        model_name = params.get('model', 'auto')
        self._load_model(model_name)

        if self.pipe is None:
            return tensor  # Fallback: passthrough if model failed to load

        self._frame_count += 1

        # Extract params with defaults
        base_strength = float(params.get('strength', 0.35))
        steps = int(params.get('steps', self._default_steps))
        guidance = float(params.get('guidance_scale', self._default_guidance))
        resolution = int(params.get('resolution', self._default_resolution))

        # Build prompt from environment + section
        env = params.get('env', '')
        section = params.get('section', '')
        prompt = params.get('prompt', '')

        if not prompt:
            prompt = ENV_PROMPTS.get(env, DEFAULT_PROMPT)
            modifier = SECTION_MODIFIERS.get(section, '')
            prompt += modifier

        negative_prompt = params.get('negative_prompt', NEGATIVE_PROMPT)

        # Audio-reactive strength adjustment
        bass = float(params.get('bass', 0))
        energy = float(params.get('energy', 0.5))
        beat_pulse = float(params.get('beat_pulse', 0))

        # Smooth energy for stable strength
        smooth_energy = self.audio.smooth('energy', energy, attack=0.2, release=0.05)

        # High energy → less AI (preserve the intense procedural rendering)
        # Low energy → more AI (calm sections benefit from neural detail)
        energy_factor = 1.0 - 0.3 * smooth_energy
        strength = base_strength * energy_factor

        # Beat pulse: momentary strength reduction (preserve beat impact)
        if beat_pulse > 0.5:
            strength *= 0.8

        strength = max(0.1, min(0.7, strength))

        # Convert tensor to PIL Image for diffusers
        from PIL import Image
        import torchvision.transforms.functional as TF

        # Resize to model resolution (preserving aspect ratio)
        _, _, h, w = tensor.shape
        if w > h:
            new_w = resolution
            new_h = max(64, int(resolution * h / w))
        else:
            new_h = resolution
            new_w = max(64, int(resolution * w / h))
        # Round to 8 (VAE requirement)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        resized = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
        )

        # Tensor → PIL Image
        img = TF.to_pil_image(resized.squeeze(0).clamp(0, 1))

        # Run img2img
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]

        # PIL → tensor
        result_tensor = TF.to_tensor(result).unsqueeze(0).to(self.device)

        # Resize back to original resolution
        if result_tensor.shape[2] != h or result_tensor.shape[3] != w:
            result_tensor = torch.nn.functional.interpolate(
                result_tensor, size=(h, w), mode='bilinear', align_corners=False
            )

        # Log first few frames
        if self._frame_count <= 3:
            sys.stderr.write(
                f'kornia-server: diffusion frame {self._frame_count} '
                f'(strength={strength:.2f}, steps={steps}, {new_w}x{new_h})\n'
            )
            sys.stderr.flush()

        return result_tensor.clamp(0, 1)
