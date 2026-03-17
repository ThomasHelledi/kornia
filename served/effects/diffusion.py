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

Audio-reactive parameters (from Go audioReactiveParams):
  - audio_intensity: composite energy (bass*0.5 + mid*0.3 + high*0.2)
  - bass_energy/mid_energy/high_energy: 3-band energy
  - beat_pulse: beat indicator (0-1)
  - section: intro/verse/chorus/climax/bridge/outro
  - spectral_flux: onset detection proxy

CUDA optimizations (helledi-pc RTX 2070S):
  - fp16 inference (half memory, 2x speed)
  - torch.compile() with reduce-overhead mode
  - xformers memory efficient attention
  - Temporal coherence via frame blending

Model cache: Set HF_HOME env var to control storage location.
  helledi-pc: HF_HOME=Z:/huggingface (large NVMe drive)
"""

import os
import sys
import torch
import numpy as np

from .audio_reactive import AudioReactiveEngine


# Per-environment style prompts — maps atlas-synth environments to diffusion prompts
ENV_PROMPTS = {
    'spectral-mesh': 'iridescent wireframe grid floating in cosmic void, neon blue energy streams, holographic, sci-fi, 8k',
    'spectral-xyz': 'holographic 3D axis visualization, neon grid, scientific visualization, data art, 8k',
    'spectral-complex': 'complex holographic waveform, mathematical beauty, neon topology, 8k',
    'black-waters': 'dark underwater portal with teal and amber energy rings, bioluminescent, cinematic',
    'black-waters-storm': 'violent underwater storm, dark turbulent waters, lightning bolts, dramatic',
    'black-waters-deep': 'deep abyss portal, pressure, darkness, faint bioluminescence, ominous',
    'cymatics': 'vibrating sand patterns on a metal plate, macro photography, physics simulation, ultra detailed',
    'cym-ferrofluid': 'ferrofluid spikes responding to magnetic field, metallic black liquid, macro, sharp',
    'cym-tesla': 'tesla coil plasma arcs, purple lightning, high voltage experiment, cinematic',
    'reaction-diffusion': 'organic morphogenesis patterns, biological cells dividing, microscope photography, nature',
    'rd-mitosis': 'cell mitosis under electron microscope, biological division, scientific photography',
    'cosmic-nebula': 'volumetric nebula gas clouds with stellar core, deep space photography, Hubble telescope',
    'cosmic-dna': 'DNA double helix made of starlight, cosmic biology, deep space, cinematic',
    'cosmic-bigbang': 'big bang explosion, primordial energy, cosmic radiation, intense light',
    'cosmic-energy': 'plasma ball with electric arcs, lightning, tesla coil, high voltage, dark background',
    'cosmic-clouds': 'volumetric cloud formation, storm, dramatic sky, cinematic weather photography',
    'cosmic-water': 'underwater cosmic scene, jellyfish nebula, bioluminescent deep sea, ethereal',
    'cosmic-synapses': 'neural network firing, brain synapses, electric impulses, neuroscience visualization',
    'neon-cathedral': 'cyberpunk corridor with neon signs, rain reflections, Blade Runner atmosphere, cinematic',
    'cybercity': 'futuristic cityscape at night, neon lights, flying vehicles, rain, Blade Runner, 8k',
    'prism': 'light refracting through crystal prism, rainbow spectrum, dark background, Pink Floyd',
    'solar-ascension': 'sun rising behind mountains, golden hour, volumetric god rays, epic landscape',
    'eye': 'extreme close-up of human eye, iris detail, macro photography, reflections, 8k',
    'eye-pulse': 'pulsating eye with dilating pupil, hypnotic, psychedelic colors, macro',
    'eye-bloom': 'eye with light bloom and lens flare, dreamy, soft focus, cinematic',
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
        model: str — 'sdxl-turbo' | 'sd-turbo' (default: auto)
        strength: float — img2img denoising strength 0.0-1.0 (default: 0.35)
        steps: int — inference steps (default: 4)
        guidance_scale: float — CFG scale (default: 0.0 for turbo)
        prompt: str — override prompt (default: auto from environment)
        env: str — environment name for auto-prompt lookup
        section: str — section type for prompt modifier
        resolution: int — model input resolution (default: 512)
        temporal_blend: float — blend with previous frame 0.0-1.0 (default: 0.15)
    """

    def __init__(self, device):
        self.device = device
        self.pipe = None
        self.model_name = None
        self.audio = AudioReactiveEngine()
        self._frame_count = 0
        self._prev_result = None  # For temporal coherence
        self._compiled = False

    def _load_model(self, model_name='auto'):
        """Lazy-load diffusion pipeline on first frame."""
        if self.pipe is not None and self.model_name == model_name:
            return

        if model_name == 'auto':
            if self.device.type == 'cuda':
                vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
                if vram_gb >= 10:
                    model_name = 'sdxl-turbo'
                else:
                    model_name = 'sd-turbo'
            elif self.device.type == 'mps':
                model_name = 'sdxl-turbo'
            else:
                model_name = 'sd-turbo'

        sys.stderr.write(f'kornia-server: loading diffusion model {model_name}...\n')
        sys.stderr.flush()

        try:
            from diffusers import AutoPipelineForImage2Image

            is_cuda = self.device.type == 'cuda'

            if model_name in ('sdxl-turbo', 'sd-turbo'):
                repo = 'stabilityai/sdxl-turbo' if model_name == 'sdxl-turbo' else 'stabilityai/sd-turbo'
                self.pipe = AutoPipelineForImage2Image.from_pretrained(
                    repo,
                    torch_dtype=torch.float16 if is_cuda else torch.float32,
                    variant='fp16' if is_cuda else None,
                ).to(self.device)
                self._default_steps = 4
                self._default_guidance = 0.0
                self._default_resolution = 512
            else:
                raise ValueError(f'Unknown model: {model_name}')

            # CUDA optimizations
            if is_cuda:
                # Memory efficient attention (xformers or PyTorch 2.0 SDPA)
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    sys.stderr.write('kornia-server: xformers enabled\n')
                except Exception:
                    pass  # PyTorch 2.0+ has SDPA by default

                # VAE slicing for lower VRAM
                if hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()

                # torch.compile for 20-30% speedup (warmup on first frame)
                try:
                    self.pipe.unet = torch.compile(
                        self.pipe.unet, mode='reduce-overhead', fullgraph=True
                    )
                    self._compiled = True
                    sys.stderr.write('kornia-server: torch.compile enabled (reduce-overhead)\n')
                except Exception as e:
                    sys.stderr.write(f'kornia-server: torch.compile skipped: {e}\n')

            # MPS optimizations
            elif self.device.type == 'mps':
                if hasattr(self.pipe, 'enable_attention_slicing'):
                    self.pipe.enable_attention_slicing()

            self.model_name = model_name
            sys.stderr.write(
                f'kornia-server: {model_name} loaded on {self.device}'
                f' (fp16={is_cuda}, compiled={self._compiled})\n'
            )
            sys.stderr.flush()

        except ImportError:
            sys.stderr.write(
                'kornia-server: ERROR — diffusers not installed. '
                'Run: pip install diffusers transformers accelerate\n'
            )
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
            params: dict with effect parameters + audio-reactive data from Go

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
        temporal_blend = float(params.get('temporal_blend', 0.15))

        # Build prompt from environment + section
        env = params.get('env', '')
        section = params.get('section', '')
        prompt = params.get('prompt', '')

        if not prompt:
            prompt = ENV_PROMPTS.get(env, DEFAULT_PROMPT)
            modifier = SECTION_MODIFIERS.get(section, '')
            prompt += modifier

        negative_prompt = params.get('negative_prompt', NEGATIVE_PROMPT)

        # Audio-reactive strength — use Go's param names
        # Go sends: audio_intensity, bass_energy, mid_energy, high_energy, beat_pulse
        audio_intensity = float(params.get('audio_intensity', 0.5))
        beat_pulse = float(params.get('beat_pulse', 0))
        spectral_flux = float(params.get('spectral_flux', 0))

        # Smooth energy for stable strength
        smooth_energy = self.audio.smooth('energy', audio_intensity, attack=0.2, release=0.05)

        # High energy → less AI (preserve the intense procedural rendering)
        # Low energy → more AI (calm sections benefit from neural detail)
        energy_factor = 1.0 - 0.3 * smooth_energy
        strength = base_strength * energy_factor

        # Beat pulse: momentary strength reduction (preserve beat impact)
        if beat_pulse > 0.5:
            strength *= 0.8

        # Spectral flux onset: brief boost to create visual "hits"
        onset = self.audio.onset_strength(spectral_flux)
        if onset > 0.3:
            strength = min(strength * 1.15, 0.7)

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
        img = TF.to_pil_image(resized.squeeze(0).clamp(0, 1).cpu())

        # Run img2img — use autocast for CUDA fp16 inference
        if self.device.type == 'cuda':
            with torch.no_grad(), torch.cuda.amp.autocast():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=img,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                ).images[0]
        else:
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

        # Temporal coherence: blend with previous frame to reduce flicker
        if temporal_blend > 0 and self._prev_result is not None:
            if self._prev_result.shape == result_tensor.shape:
                result_tensor = (
                    (1.0 - temporal_blend) * result_tensor
                    + temporal_blend * self._prev_result
                )

        # Cache for temporal blending
        self._prev_result = result_tensor.detach().clone()

        # Log periodically
        if self._frame_count <= 3 or self._frame_count % 100 == 0:
            fps_note = ''
            if self._frame_count > 1 and hasattr(self, '_last_log_frame'):
                # Rough throughput estimate
                fps_note = f', ~{self._frame_count} frames total'
            sys.stderr.write(
                f'kornia-server: diffusion frame {self._frame_count} '
                f'(strength={strength:.2f}, steps={steps}, {new_w}x{new_h}{fps_note})\n'
            )
            sys.stderr.flush()
            self._last_log_frame = self._frame_count

        return result_tensor.clamp(0, 1)
