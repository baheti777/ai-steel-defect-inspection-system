# ============================================================
# STEEL SURFACE DEFECT ANALYZER — HuggingFace Spaces
# app.py — upload this + requirements.txt + best_model.pth
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import gradio as gr

# ── Device ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ───────────────────────────────────────────────────
CLASSES = [
    'crazing', 'inclusion', 'patches',
    'pitted_surface', 'rolled-in_scale', 'scratches'
]
NUM_CLASSES  = len(CLASSES)
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
IMG_SIZE     = 224

# ── Model path (HuggingFace — same folder as app.py) ─────────
MODEL_PATH = "best_model.pth"

# ── Load Model ───────────────────────────────────────────────
def load_model(model_path, num_classes, device):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, NUM_CLASSES, device)

# ── Transform ────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── GradCAM ──────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam - cam.min()
        cam     = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

gradcam = GradCAM(model, model.features[-1])

# ── Root Cause Knowledge Base ────────────────────────────────
ROOT_CAUSE_KB = {
    'crazing': {
        'description': 'Network of fine cracks on steel surface resembling a spider web.',
        'causes': [
            'Rapid and uneven cooling after rolling or heat treatment',
            'High residual thermal stress during quenching',
            'Improper cooling rate in the cooling bed',
        ],
        'process_checks': [
            'Inspect cooling bed temperature uniformity',
            'Review quenching water temperature and flow rate',
            'Check thermal gradient across the slab cross-section',
        ],
        'corrective_actions': [
            'Reduce quenching rate gradually',
            'Ensure uniform water spray distribution',
            'Preheat steel to reduce thermal shock',
        ]
    },
    'inclusion': {
        'description': 'Non-metallic particles (oxides, sulfides, silicates) embedded in steel matrix.',
        'causes': [
            'Insufficient ladle refining and deslagging',
            'Reoxidation during teeming or casting',
            'Slag entrapment from ladle or tundish',
        ],
        'process_checks': [
            'Review ladle refining time and slag composition',
            'Inspect tundish lining for erosion',
            'Check argon stirring intensity and duration',
        ],
        'corrective_actions': [
            'Improve ladle deslagging practice',
            'Optimize argon purging to prevent reoxidation',
            'Use calcium treatment to globularize inclusions',
        ]
    },
    'patches': {
        'description': 'Irregular surface discoloration or scale patches on steel surface.',
        'causes': [
            'Uneven scale formation during reheating',
            'Inconsistent furnace atmosphere (oxidizing zones)',
            'Improper descaling leaving residual scale',
        ],
        'process_checks': [
            'Monitor furnace temperature profile across length',
            'Inspect descaler nozzle pressure and coverage',
            'Check furnace atmosphere oxygen levels',
        ],
        'corrective_actions': [
            'Optimize furnace heating schedule',
            'Increase descaling water pressure',
            'Ensure uniform slab heating before rolling',
        ]
    },
    'pitted_surface': {
        'description': 'Small craters or pits distributed across the steel surface.',
        'causes': [
            'Scale pressed into the surface during rolling',
            'Corrosion pits from improper storage or handling',
            'Gas porosity from casting reaching the surface',
        ],
        'process_checks': [
            'Inspect descaling effectiveness before each rolling pass',
            'Review slab storage conditions for moisture exposure',
            'Check mold powder consumption during casting',
        ],
        'corrective_actions': [
            'Improve descaling before finishing rolling pass',
            'Control storage humidity and apply rust inhibitor',
            'Optimize casting speed and mold oscillation',
        ]
    },
    'rolled-in_scale': {
        'description': 'Oxide scale folded into the steel surface during rolling.',
        'causes': [
            'Inadequate descaling before hot rolling passes',
            'Excessive furnace scale formation (high temp / long soak)',
            'Low descaling water pressure or blocked nozzles',
        ],
        'process_checks': [
            'Check descaler nozzle condition and pressure logs',
            'Review reheating furnace temperature and soaking time',
            'Inspect inter-pass descaling between rolling stands',
        ],
        'corrective_actions': [
            'Increase descaling pressure to ≥180 bar',
            'Reduce furnace soaking time at peak temperature',
            'Schedule regular descaler nozzle maintenance',
        ]
    },
    'scratches': {
        'description': 'Linear surface marks caused by mechanical contact or abrasion.',
        'causes': [
            'Damaged or worn roller table guides',
            'Foreign debris on roller surface',
            'Improper handling during transport or coiling',
        ],
        'process_checks': [
            'Inspect all roller surfaces and side guides for wear',
            'Check coiler mandrel and pinch roll condition',
            'Review material handling equipment for sharp edges',
        ],
        'corrective_actions': [
            'Replace or resurface worn roller guides',
            'Clear debris from roller tables before rolling',
            'Improve coiling tension control to avoid slippage',
        ]
    }
}

RPN = {
    'inclusion':       10,
    'rolled-in_scale':  9,
    'crazing':          8,
    'pitted_surface':   7,
    'patches':          5,
    'scratches':        4,
}

# ── Analysis Function ─────────────────────────────────────────
def analyze(uploaded_image):
    if uploaded_image is None:
        return None, "Please upload an image."

    x = val_transform(uploaded_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()

    idx    = int(np.argmax(probs))
    defect = IDX_TO_CLASS[idx]
    conf   = probs[idx]

    # GradCAM overlay
    cam = gradcam.generate(x)
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
    ) / 255.0

    img_resized = np.array(uploaded_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heatmap     = plt.cm.jet(cam_resized)[:, :, :3]
    overlay     = np.clip(0.5 * img_resized + 0.5 * heatmap, 0, 1)
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    # GradCAM-based severity
    activated_area = (cam_resized > 0.5).sum()
    area_ratio     = activated_area / (IMG_SIZE * IMG_SIZE)
    area_pct       = round(area_ratio * 100, 2)

    if area_ratio > 0.30:
        severity = "🔴 Critical"
    elif area_ratio > 0.15:
        severity = "🟠 Moderate"
    else:
        severity = "🟢 Minor"

    warning = ""
    if conf < 0.70:
        warning = "\n⚠️  UNCERTAIN PREDICTION — Manual inspection recommended.\n"

    info    = ROOT_CAUSE_KB[defect]
    causes  = "\n".join([f"  • {c}" for c in info['causes']])
    checks  = "\n".join([f"  • {c}" for c in info['process_checks']])
    actions = "\n".join([f"  • {a}" for a in info['corrective_actions']])
    prob_bars = "\n".join([
        f"  {IDX_TO_CLASS[i]:>18s} | {'█' * int(probs[i] * 20):<20} {probs[i]:.2%}"
        for i in range(NUM_CLASSES)
    ])

    report = f"""
{'='*56}
  DEFECT DETECTED  :  {defect.replace('_', ' ').upper()}
  CONFIDENCE       :  {conf:.2%}
  SEVERITY         :  {severity}
  DEFECT AREA      :  {area_pct}% of surface
  RISK (RPN)       :  {RPN[defect]} / 10
{'='*56}{warning}
📋 DESCRIPTION
  {info['description']}

🔍 PROBABLE ROOT CAUSES
{causes}

🔧 PROCESS PARAMETERS TO CHECK
{checks}

✅ RECOMMENDED CORRECTIVE ACTIONS
{actions}

📊 ALL CLASS PROBABILITIES
{prob_bars}
"""
    return overlay_pil, report


# ── Launch ────────────────────────────────────────────────────
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(
        type="pil",
        label="Upload Steel Surface Image"
    ),
    outputs=[
        gr.Image(
            type="pil",
            label="GradCAM Overlay — Where Model Looked"
        ),
        gr.Textbox(
            label="Full Metallurgical Analysis Report",
            lines=38
        )
    ],
    title="🔬 Steel Surface Defect Analyzer",
    description=(
        "Upload a steel surface image to detect defects, "
        "visualize where the model focused using GradCAM, "
        "assess severity based on defect area, and get "
        "metallurgical root cause analysis with corrective actions."
    ),
    flagging_mode='never',
    theme=gr.themes.Soft()
)

demo.launch()
