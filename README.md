🖼️ AutoPatch: Autonomous Patchwise AI Editor (Adobe 2030 Prototype)
🌍 Overview

AutoPatch is a lightweight, mobile-first AI image editing prototype for 2030 — showing how intent-based editing (“brighten left sky”, “remove object”) can work efficiently on low-compute devices.
It merges prompt understanding, region segmentation, patchwise editing, and seamless blending — keeping human control while enabling smart automation.

🚀 Features

🧠 Prompt-driven edits — natural commands like “brighten the sky”

✂️ Patchwise processing — edits only relevant regions

🎨 Heuristic fallback — runs even without GPU/diffusion

🪄 Seamless blending — smooth merges via OpenCV or soft masks

⚙️ Optional diffusion models — plug SDXL Inpainting / Flux / Kandinsky

💻 Streamlit UI — lightweight web interface

🪶 Setup Instructions
1️⃣ Create and activate virtual environment

Windows

python -m venv venv
venv\Scripts\activate


Mac / Linux

python3 -m venv venv
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt


To enable full AI editing:

pip install torch torchvision torchaudio diffusers transformers accelerate

3️⃣ Run the app
streamlit run adobe_autopatch_app.py


Open http://localhost:8501
 in your browser.

🧠 How It Works

Prompt Parsing → Extract region & action from user text.

Segmentation → SAM or heuristic masks (sky, subject, background).

Patch Extraction → Divide editable regions into tiles.

Editing → Use diffusion/heuristic enhancement per patch.

Scoring → Choose best patch results via CLIP or brightness metrics.

Blending → Combine patches smoothly into the original image.

🧩 Model & Extensibility

Works out of the box (no downloads needed).

Supports:

Segment Anything (SAM)

Stable Diffusion XL Inpainting

CLIP

Set diffusion model (optional):

export SD_INPAINT_MODEL="stabilityai/stable-diffusion-2-inpainting"

❤️ Credits

Developed as part of Adobe 2030 AI Editor Prototype Challenge
Built using Python, Streamlit, OpenCV, Pillow, and Diffusers

Author: Hirday (IIT Dhanbad)
Prototype inspired by Adobe Firefly’s intent-based editing vision
