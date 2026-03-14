"""Video Analyzer — Extract visual DNA from reference video keyframes.

Batch analysis of extracted keyframes for analyze-video command.
NOT a per-frame VisionServer effect; called once during analysis.

Metrics per frame:
  - dominant_colors: Top 3 RGB via k-means (pure torch, no sklearn)
  - brightness: Mean luminance 0-1
  - color_temperature: Warm/cool from HSV hue distribution
  - edge_density: Canny edge pixel percentage
  - saturation: Mean saturation from HSV
"""

import torch
import kornia.color as KC
import kornia.filters as KF
from PIL import Image
import torchvision.transforms.functional as TF


def load_image_tensor(path, device='cpu', max_dim=512):
    """Load image file as (1, 3, H, W) float tensor."""
    img = Image.open(path).convert('RGB')
    # Resize keeping aspect ratio
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    return tensor


def kmeans_colors(tensor, k=3, iterations=10):
    """Extract top-k dominant colors via iterative k-means on GPU/CPU.

    Args:
        tensor: (1, 3, H, W) float tensor 0-1
        k: number of clusters
        iterations: convergence iterations

    Returns:
        list of (R, G, B) tuples (0-255)
    """
    # Reshape to (N, 3)
    pixels = tensor.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    n = pixels.shape[0]

    # Subsample for speed (max 10k pixels)
    if n > 10000:
        idx = torch.randperm(n, device=pixels.device)[:10000]
        pixels = pixels[idx]
        n = 10000

    # Initialize centroids randomly
    idx = torch.randperm(n, device=pixels.device)[:k]
    centroids = pixels[idx].clone()

    for _ in range(iterations):
        # Assign pixels to nearest centroid
        dists = torch.cdist(pixels.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
        labels = dists.argmin(dim=1)

        # Update centroids
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = pixels[mask].mean(dim=0)

    # Sort by cluster size (largest first)
    _, labels = torch.cdist(pixels.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0).min(dim=1)
    counts = torch.bincount(labels, minlength=k).float()
    order = counts.argsort(descending=True)
    centroids = centroids[order]

    return [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in centroids.cpu()]


def analyze_frame(tensor):
    """Analyze a single frame tensor (1, 3, H, W).

    Returns dict with: dominant_colors, brightness, color_temperature,
                        edge_density, saturation
    """
    # Brightness: mean luminance
    gray = KC.rgb_to_grayscale(tensor)
    brightness = gray.mean().item()

    # HSV analysis
    hsv = KC.rgb_to_hsv(tensor.clamp(1e-6, 1))
    hue = hsv[:, 0:1]       # 0 to 2*pi
    sat = hsv[:, 1:2]       # 0 to 1
    saturation = sat.mean().item()

    # Color temperature: warm (red/orange/yellow, hue 0-1.2 or 5.5-6.28)
    # vs cool (blue/cyan/green, hue 1.8-4.5)
    hue_flat = hue.flatten()
    warm_mask = (hue_flat < 1.2) | (hue_flat > 5.5)
    cool_mask = (hue_flat > 1.8) & (hue_flat < 4.5)
    warm_pct = warm_mask.float().mean().item()
    cool_pct = cool_mask.float().mean().item()
    total = warm_pct + cool_pct + 1e-6
    color_temperature = warm_pct / total  # 1.0 = fully warm, 0.0 = fully cool

    # Edge density via Canny
    _, edges = KF.canny(gray, low_threshold=0.08, high_threshold=0.25)
    edge_density = edges.float().mean().item()

    # Dominant colors
    colors = kmeans_colors(tensor, k=3)

    return {
        'dominant_colors': colors,
        'brightness': round(brightness, 3),
        'color_temperature': round(color_temperature, 3),
        'edge_density': round(edge_density, 4),
        'saturation': round(saturation, 3),
    }


def analyze_frames(paths, device='cpu', max_dim=512):
    """Batch analyze multiple keyframe images.

    Args:
        paths: list of image file paths
        device: torch device
        max_dim: max image dimension for analysis

    Returns:
        list of analysis dicts (one per frame)
    """
    results = []
    for path in paths:
        tensor = load_image_tensor(path, device=device, max_dim=max_dim)
        result = analyze_frame(tensor)
        result['path'] = path
        results.append(result)
    return results


def extract_edge_guide(path, output_path, device='cpu', max_dim=512):
    """Extract Canny edge map from keyframe, save as grayscale PNG.

    Args:
        path: input image path
        output_path: where to save edge map PNG
        device: torch device
        max_dim: max dimension
    """
    tensor = load_image_tensor(path, device=device, max_dim=max_dim)
    gray = KC.rgb_to_grayscale(tensor)
    _, edges = KF.canny(gray, low_threshold=0.08, high_threshold=0.25)

    # Convert to PIL and save
    edge_np = (edges.squeeze().cpu().numpy() * 255).astype('uint8')
    Image.fromarray(edge_np, mode='L').save(output_path)


# ── Color-to-Style Mapping ─────────────────────────────────────────────

def dominant_hue_category(colors):
    """Classify dominant colors into a hue category.

    Returns: 'warm', 'cool', 'green', 'purple', 'neutral'
    """
    if not colors:
        return 'neutral'

    r, g, b = colors[0]  # Most dominant color

    # Convert to HSV for hue analysis
    t = torch.tensor([[[r / 255.0, g / 255.0, b / 255.0]]]).permute(0, 2, 1).unsqueeze(-1)
    hsv = KC.rgb_to_hsv(t.clamp(1e-6, 1))
    hue = hsv[0, 0, 0, 0].item()  # radians 0-2pi
    sat = hsv[0, 1, 0, 0].item()

    if sat < 0.15:
        return 'neutral'  # desaturated

    # Hue ranges (radians): R=0, Y=1.05, G=2.09, C=3.14, B=4.19, M=5.24
    if hue < 1.2 or hue > 5.5:
        return 'warm'     # red/orange/yellow
    elif 1.8 < hue < 3.0:
        return 'green'
    elif 3.0 < hue < 4.5:
        return 'cool'     # blue/cyan
    elif 4.5 < hue < 5.5:
        return 'purple'
    return 'neutral'


def map_to_cinema_style(analysis):
    """Map frame analysis → cinema style name."""
    hue_cat = dominant_hue_category(analysis['dominant_colors'])
    sat = analysis['saturation']

    if sat > 0.5:
        if hue_cat == 'warm':
            return 'anime'
        elif hue_cat == 'purple':
            return 'cyberpunk'
        return 'psychedelic'
    elif sat < 0.2:
        return 'noir'

    style_map = {
        'warm': 'fantasy',
        'cool': 'cinematic',
        'green': 'nature',
        'purple': 'cyberpunk',
        'neutral': 'cinematic',
    }
    return style_map.get(hue_cat, 'cinematic')


def map_to_environment(analysis):
    """Map frame analysis → environment name."""
    brightness = analysis['brightness']

    if brightness < 0.3:
        envs = ['black-waters', 'eye', 'neon-cathedral']
    elif brightness < 0.6:
        envs = ['water', 'forest', 'pianist', 'swarm']
    else:
        envs = ['starry', 'earth', 'space', 'solar-ascension']

    # Pick based on saturation sub-index
    sat = analysis['saturation']
    idx = int(sat * (len(envs) - 1))
    return envs[min(idx, len(envs) - 1)]


def map_to_preset(analysis):
    """Map frame analysis → DreamWave preset."""
    density = analysis['edge_density']

    if density > 0.15:
        return 'subtle'
    elif density > 0.05:
        return 'medium'
    else:
        return 'transcendent'


def map_to_mood(analyses):
    """Auto-detect overall mood from aggregate frame analyses."""
    if not analyses:
        return 'epic'

    avg_brightness = sum(a['brightness'] for a in analyses) / len(analyses)
    avg_saturation = sum(a['saturation'] for a in analyses) / len(analyses)
    brightness_var = sum((a['brightness'] - avg_brightness) ** 2 for a in analyses) / len(analyses)

    # High variance + high saturation = chaotic
    if brightness_var > 0.03 and avg_saturation > 0.35:
        return 'chaotic'
    # Dark
    if avg_brightness < 0.3:
        return 'dark'
    # Bright
    if avg_brightness > 0.55:
        return 'bright'
    # Low saturation + moderate brightness = intimate
    if avg_saturation < 0.25:
        return 'intimate'
    return 'epic'
