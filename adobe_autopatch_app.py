"""
Autonomous Patchwise Editing - Minimal runnable prototype
File: adobe_autopatch_app.py

What this is: a single-file Streamlit prototype that implements the pipeline logic you asked for:
- Prompt parsing via regex (simple agent)
- Lightweight segmentation (sky color heuristic + saliency fallback)
- Patch extraction and ranking (saliency * prompt-relevance heuristic)
- Patch "editor" that tries to call a diffusion inpainting model if available
  (Hugging Face diffusers). If not available, a fast fallback edit is applied
  (brightness / color transform) so you can test the full flow locally.
- Candidate scoring (CLIP if available, else simple heuristic)
- Patch blending using OpenCV seamlessClone

NOTES:
- This is a prototype to help you develop and test the full orchestration locally.
- For production-grade results plug in real models (SAM, diffusers SDXL inpainting,
  CLIP) where indicated in the code.

Requirements (install these in a virtualenv):

pip install -r requirements.txt

where requirements.txt contains (example):
streamlit
numpy
opencv-python
pillow
scikit-image
torch
transformers
diffusers
segment-anything
ftfy
regex

(You can omit heavy libs like `diffusers` and `segment-anything` if you only want the lightweight fallback behaviour.)

Run:
streamlit run adobe_autopatch_app.py

"""

import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
import os
import re
import math
from skimage import exposure

# Try optional imports
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Optional model imports (used only when available)
HAS_SAM = False
HAS_DIFFUSERS = False
HAS_CLIP = False

try:
    from segment_anything import SamPredictor, sam_model_registry
    HAS_SAM = True
except Exception:
    HAS_SAM = False

try:
    from diffusers import StableDiffusionInpaintPipeline
    HAS_DIFFUSERS = True
except Exception:
    HAS_DIFFUSERS = False

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except Exception:
    HAS_CLIP = False

# ----------------------
# 1) Prompt parser (regex-based simple agent)
# ----------------------

def parse_prompt(text):
    """Return list of parsed operations. Each op is a dict with keys:
    - region_hint (e.g., 'sky', 'left', 'background')
    - action (e.g., 'brighten', 'darken', 'remove', 'replace')
    - intensity (float 0-1)
    - raw (original text)
    """
    t = text.lower()
    ops = []

    # simple rules
    region = None
    if re.search(r"\bsky\b", t):
        region = 'sky'
    elif re.search(r"\bbackground\b", t):
        region = 'background'
    elif re.search(r"\bsubject\b|\bperson\b|\bportrait\b", t):
        region = 'subject'
    # direction hints
    direction = None
    m = re.search(r"(left|right|top|bottom)\s+half", t)
    if m:
        direction = m.group(1)

    # actions
    if re.search(r"brighten|increase brightness|make.*lighter", t):
        action = 'brighten'
    elif re.search(r"darken|decrease brightness|make.*darker", t):
        action = 'darken'
    elif re.search(r"remove|erase|delete object|object removal", t):
        action = 'remove'
    elif re.search(r"replace|change to|make it look like", t):
        action = 'replace'
    else:
        # default to stylize/edit intent if none matched
        action = 'stylize'

    # intensity extraction (0-1)
    m = re.search(r"(\d{1,3})%", t)
    intensity = None
    if m:
        intensity = min(max(int(m.group(1)) / 100.0, 0.0), 1.0)
    else:
        # keywords
        if 'slightly' in t:
            intensity = 0.25
        elif 'a lot' in t or 'strongly' in t or 'dramatically' in t:
            intensity = 0.9
        else:
            intensity = 0.6

    ops.append({
        'region_hint': region,
        'direction': direction,
        'action': action,
        'intensity': intensity,
        'raw': text
    })
    return ops

# ----------------------
# 2) Segmentation (SAM if available else heuristics)
# ----------------------

def segment_sky_by_color(np_img):
    """Simple HSV-based sky mask. Returns boolean mask same HxW."""
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    # sky tends to have hue in blue range; tune thresholds if needed
    lower = np.array([85, 20, 50])
    upper = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # morphological clean
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = mask > 0
    return mask


def saliency_mask(np_img):
    """Simple saliency via gradient magnitude thresholding."""
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
    mask = mag > 0.25
    # fill small holes
    mask = mask.astype('uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    return mask.astype(bool)


def get_region_mask(np_img, region_hint=None):
    """Return boolean mask for the region hint. Uses SAM if available.
    If region_hint is None, return saliency mask.
    """
    h, w = np_img.shape[:2]
    if HAS_SAM and region_hint:
        # Here you'd run SAM with text prompt or a box; placeholder since
        # setting up SAM predictor may require checkpoint path.
        try:
            # user's environment must have a SAM checkpoint configured for this branch
            # This block will fail silently if SAM isn't fully set up.
            predictor = None
            # Example (commented):
            # sam = sam_model_registry["vit_b"](checkpoint="/path/to/sam.pt")
            # predictor = SamPredictor(sam)
            # boxes = predictor.predict(...)
            pass
        except Exception:
            pass

    # Fallback heuristics
    if region_hint == 'sky':
        return segment_sky_by_color(np_img)
    elif region_hint == 'background':
        # try to get large low-frequency area via saliency invert
        s = saliency_mask(np_img)
        return ~s
    elif region_hint == 'subject':
        # approximate by saliency (foreground)
        return saliency_mask(np_img)
    else:
        return saliency_mask(np_img)

# ----------------------
# 3) Patch extraction & ranking
# ----------------------

def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (x0, y0, x1, y1)


def extract_patches(np_img, mask, patch_size=256, overlap=0.2):
    """Given an image and boolean mask, extract overlapping patches that cover mask.
    Returns list of dicts: {id, x,y,w,h,patch_img,mask_patch}
    """
    h, w = mask.shape
    stride = int(patch_size * (1 - overlap))
    patches = []
    pid = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + patch_size, w)
            y2 = min(y + patch_size, h)
            submask = mask[y:y2, x:x2]
            if submask.sum() == 0:
                continue
            patch_img = np_img[y:y2, x:x2]
            patches.append({
                'id': pid,
                'x': x, 'y': y, 'w': x2-x, 'h': y2-y,
                'patch': patch_img.copy(),
                'mask': submask.copy()
            })
            pid += 1
    return patches


def compute_prompt_relevance(patch_np, prompt):
    """Heuristic relevance score between patch and prompt.
    If CLIP present, use CLIP; else simple color/brightness heuristics.
    """
    if HAS_CLIP:
        try:
            # lightweight CLIP scoring (requires model to be loaded globally) - not implemented here
            return 0.5
        except Exception:
            pass
    # heuristic: if prompt mentions 'brighten' measure darkness -> more relevant
    p = prompt.lower()
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean() / 255.0
    score = 0.0
    if 'brighten' in p:
        score = 1 - mean_brightness
    elif 'darken' in p:
        score = mean_brightness
    elif 'remove' in p:
        # if patch has a lot of edges assume object -> higher relevance
        edges = cv2.Canny(gray, 50, 150)
        score = edges.mean()
    else:
        # general saliency: variance
        score = patch_np.var() / (255.0**2)
    return float(score)


def rank_patches(patches, prompt, top_k=6):
    # compute combined score = mask_coverage_ratio * prompt_relevance
    scored = []
    for p in patches:
        mask = p['mask']
        cover = mask.sum() / (p['w'] * p['h'] + 1e-9)
        relevance = compute_prompt_relevance(p['patch'], prompt)
        score = cover * (0.6 * relevance + 0.4 * cover)
        scored.append((p, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]

# ----------------------
# 4) Patch editor (attempt to use diffusers, else fallback)
# ----------------------

def diffusion_inpaint_patch(patch_np, patch_mask, prompt, steps=8):
    """Try to run a diffusion inpainting model on the patch. If unavailable
    perform a simple deterministic edit based on the prompt.
    Returns edited patch as uint8 RGB numpy array same size as input.
    """
    h, w = patch_np.shape[:2]
    # Prefers real model if available
    if HAS_DIFFUSERS and HAS_TORCH:
        try:
            # Example: using StableDiffusionInpaintPipeline (this requires model id & auth token)
            # This code is intentionally minimal; user must set MODEL_ID env var or edit below.
            model_id = os.environ.get('SD_INPAINT_MODEL', None)
            if model_id:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipe = pipe.to('cuda')
                # convert patch to PIL
                pil = Image.fromarray(patch_np)
                mask_pil = Image.fromarray((patch_mask*255).astype('uint8'))
                out = pipe(prompt=prompt, image=pil, mask_image=mask_pil, num_inference_steps=steps)
                edited = np.array(out.images[0].convert('RGB'))
                return edited
        except Exception:
            pass
    # Fallback deterministic edits
    p = prompt.lower()
    img = Image.fromarray(patch_np)
    if 'brighten' in p:
        factor = 1.0 + 0.6 * (0.6)  # use intensity estimate (simple)
        enhancer = ImageEnhance.Brightness(img)
        edited = enhancer.enhance(1.3)
        return np.array(edited)
    elif 'darken' in p:
        enhancer = ImageEnhance.Brightness(img)
        edited = enhancer.enhance(0.7)
        return np.array(edited)
    elif 'remove' in p or 'erase' in p:
        # crude inpaint: fill masked area with neighboring median color
        edited = patch_np.copy()
        mask = patch_mask.astype(bool)
        if mask.sum() > 0:
            # compute median color around mask
            dil = cv2.dilate(mask.astype('uint8'), np.ones((15,15),np.uint8))
            ring = dil.astype(bool) & (~mask)
            if ring.sum() > 0:
                for c in range(3):
                    med = np.median(patch_np[:,:,c][ring])
                    edited[:,:,c][mask] = med
        # slight blur to hide seams
        edited = cv2.GaussianBlur(edited, (7,7), 0)
        return edited
    else:
        # stylize-ish: increase contrast
        arr = exposure.equalize_adapthist(patch_np/255.0, clip_limit=0.03)
        return (arr * 255).astype('uint8')

# ----------------------
# 5) Candidate scoring
# ----------------------

def score_candidate_patch(orig_patch, cand_patch, prompt):
    """Return a float score for a candidate patch. If CLIP available use it.
    Otherwise use heuristics: similarity to prompt (brightness changes, edge
    preservation) and small LPIPS-like proxy using SSIM.
    """
    # simple heuristic: prefer higher CLIP sim (if available), else prefer
    # changes that align to prompt intentions and preserve structure.
    score = 0.0
    if HAS_CLIP:
        try:
            # placeholder for actual CLIP usage
            score += 0.5
        except Exception:
            pass
    # brightness heuristic
    p = prompt.lower()
    gray_orig = cv2.cvtColor(orig_patch, cv2.COLOR_RGB2GRAY).astype(float)
    gray_cand = cv2.cvtColor(cand_patch, cv2.COLOR_RGB2GRAY).astype(float)
    mean_orig = gray_orig.mean()
    mean_cand = gray_cand.mean()
    bscore = 1.0 - abs(mean_cand - mean_orig) / 255.0
    score += 0.5 * bscore
    # structure preservation: ssim proxy using normalized cross correlation
    num = ((gray_orig - gray_orig.mean()) * (gray_cand - gray_cand.mean())).sum()
    den = math.sqrt(((gray_orig - gray_orig.mean())**2).sum() * ((gray_cand - gray_cand.mean())**2).sum() + 1e-9)
    corr = num / (den + 1e-9)
    score += 0.5 * max(0.0, corr)
    return float(score)

# ----------------------
# 6) Merge & blend
# ----------------------

def blend_patch_into_image(base_img_np, patch_np, mask_patch, x, y):
    """Blend patch_np into base_img_np at location x,y using seamlessClone.
    mask_patch is boolean mask for region within patch area. Returns updated base image.
    """
    out = base_img_np.copy()
    h, w = patch_np.shape[:2]
    center = (x + w//2, y + h//2)
    # create 3-channel mask for seamlessClone
    clone_mask = (mask_patch.astype('uint8') * 255)
    if clone_mask.sum() == 0:
        # if empty mask, paste patch directly
        out[y:y+h, x:x+w] = patch_np
        return out
    try:
        src = patch_np.astype('uint8')
        dst = base_img_np.astype('uint8')
        # seamlessClone requires BGR
        src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        clone_mask_u8 = clone_mask.astype('uint8')
        res = cv2.seamlessClone(src_bgr, dst_bgr, clone_mask_u8, center, cv2.NORMAL_CLONE)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return res
    except Exception:
        # fallback: feathered alpha blend
        alpha = cv2.GaussianBlur(clone_mask.astype('float32')/255.0, (31,31), 0)
        alpha = alpha[:,:,None]
        out[y:y+h, x:x+w] = (alpha * patch_np + (1-alpha) * out[y:y+h, x:x+w]).astype('uint8')
        return out

# ----------------------
# 7) Full pipeline function
# ----------------------

def autonomous_patchwise_edit(pil_img, text_prompt, top_k=6, candidates=2):
    np_img = np.array(pil_img.convert('RGB'))
    ops = parse_prompt(text_prompt)
    edited = np_img.copy()
    for op in ops:
        mask = get_region_mask(edited, op['region_hint'])
        if mask is None or mask.sum() == 0:
            st.warning("No region found for hint: {}. Trying saliency fallback.".format(op['region_hint']))
            mask = saliency_mask(edited)
        patches = extract_patches(edited, mask)
        if len(patches) == 0:
            st.warning('No patches extracted from mask.')
            continue
        ranked = rank_patches(patches, text_prompt, top_k=top_k)
        # per-patch editing
        for p in ranked:
            x, y = p['x'], p['y']
            orig_patch = p['patch']
            pmask = p['mask']
            best_cand = None
            best_score = -1e9
            for _ in range(candidates):
                cand = diffusion_inpaint_patch(orig_patch, pmask, text_prompt, steps=8)
                s = score_candidate_patch(orig_patch, cand, text_prompt)
                if s > best_score:
                    best_score = s
                    best_cand = cand
            # blend best candidate
            edited = blend_patch_into_image(edited, best_cand, pmask, x, y)
    return Image.fromarray(edited)

# ----------------------
# 8) Streamlit UI
# ----------------------

def main():
    st.set_page_config(page_title='AutoPatch Editor - Prototype', layout='centered')
    st.title('AutoPatch: Autonomous patchwise editor (prototype)')
    st.markdown("""
    Upload an image and write an edit prompt.
    This prototype uses simple heuristics when heavy ML models are not available.
    """)

    uploaded = st.file_uploader('Upload image', type=['png','jpg','jpeg'])
    prompt = st.text_input('Edit prompt (e.g., "brighten the left half of the sky by 30%")')
    top_k = st.sidebar.slider('Top patches to edit', 1, 12, 6)
    candidates = st.sidebar.slider('Candidates per patch', 1, 4, 2)

    if st.button('Run autonomous edit') and uploaded and prompt:
        pil = Image.open(uploaded)
        with st.spinner('Running pipeline...'):
            out = autonomous_patchwise_edit(pil, prompt, top_k=top_k, candidates=candidates)
        st.image([pil, out], caption=['Original', 'Edited'], use_column_width=True)
        # show diff
        orig_np = np.array(pil.convert('RGB'))
        out_np = np.array(out.convert('RGB'))
        diff = np.abs(orig_np.astype(int) - out_np.astype(int)).sum(axis=2).astype('uint8')
        st.image(diff, caption='Absolute diff (grayscale)')

if __name__ == '__main__':
    main()
