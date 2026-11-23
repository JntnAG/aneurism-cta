import os
import sys
import glob
import math
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from dataclasses import dataclass

try:
    import dicomsdl as dicom
except ImportError:
    import pydicom as dicom

from src.model import Model

CLASSES = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

def normalize(img, modality=None):
    img = img.astype(np.float32)
    p1, p99 = 0, 600
    img = np.clip(img, p1, p99)
    normalized = (img - p1) / ((p99 - p1) * 0.2)
    return normalized

def load_dicom_series(series_path, crop_size=448):
    dicom_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_path}")
    
    slices = []
    slice_position = []
    crop = (512 - crop_size) // 2

    for i, f in enumerate(dicom_files):
        try:
            ds = dicom.open(f)
            pos = getattr(ds, 'ImagePositionPatient', None)
            if pos is not None and len(pos) >= 3:
                pos = float(pos[2])
            else:
                pos = float(getattr(ds, 'InstanceNumber', i))
            img = ds.pixelData(0)
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            if slope is not None and intercept is not None:
                img = img * float(slope) + float(intercept)
            ds.close()
        except Exception:
            import pydicom
            pyds = pydicom.dcmread(f)
            pos = getattr(pyds, 'ImagePositionPatient', None)
            if pos is not None and len(pos) >= 3:
                pos = float(pos[2])
            else:
                pos = float(getattr(pyds, 'InstanceNumber', i))
            img = pyds.pixel_array
            slope = getattr(pyds, 'RescaleSlope', 1.0)
            intercept = getattr(pyds, 'RescaleIntercept', 0.0)
            img = img * float(slope) + float(intercept)

        slice_position.append(pos)
        img = normalize(img)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR_EXACT)
        if crop > 0:
            img = img[crop:-crop, crop:-crop]
        slices.append(img)

    idx_sorted = [x[0] for x in sorted(enumerate(slice_position), key=lambda x: x[1])]
    images = np.stack([slices[idx] for idx in idx_sorted], axis=0)

    # Statistical normalization
    images = images - 2.5
    images = torch.from_numpy(images)

    # 2.5D slab (left, center, right)
    left = torch.roll(images, 1, 0)
    right = torch.roll(images, -1, 0)
    images = torch.stack([left, images, right], dim=1) # (N, 3, H, W)
    return images

def run_inference(series_dir, model_dir, folds, output_csv, device="cuda:0"):
    print(f"Loading DICOM series from: {series_dir}")
    images = load_dicom_series(series_dir, crop_size=448)
    print(f"Loaded volume shape: {images.shape}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_fold_probs = []

    for fold in folds:
        fold_path = os.path.join(model_dir, f"fold-{fold}")
        if not os.path.exists(fold_path):
            print(f"Skipping fold-{fold} (not found)")
            continue
        print(f"Running inference with fold-{fold}...")

        with open(os.path.join(fold_path, "config_stage2.pkl"), "rb") as f:
            transformer_config = pickle.load(f)
        if not hasattr(transformer_config, '_experts_implementation_internal'):
            transformer_config._experts_implementation_internal = None

        model = Model(encoder_name="convnext_base.dinov3_lvd1689m", pool="gem", transformer_config=transformer_config)
        weights_path = os.path.join(fold_path, "weights_inference.pth")
        weights = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(weights, strict=True)
        model.to(device)
        model.eval()

        batch_size = 32
        features_list = []
        batches = math.ceil(len(images) / batch_size)

        with torch.no_grad():
            for b in range(batches):
                start = b * batch_size
                end = start + batch_size
                batch = images[start:end].to(device)
                feat = model.forward_encoder(batch)
                features_list.append(feat)

            features = torch.cat(features_list, dim=0).unsqueeze(0)
            mask = torch.ones(1, features.shape[1], dtype=features.dtype, device=features.device)
            logits = model.forward_transformer(features, mask)
            probs = logits[0].sigmoid().cpu().numpy()
            all_fold_probs.append(probs)

    if not all_fold_probs:
        raise RuntimeError("No predictions generated.")

    # Ensemble average across folds
    mean_probs = np.mean(all_fold_probs, axis=0)

    series_id = os.path.basename(os.path.normpath(series_dir))
    df = pd.DataFrame([{
        "SeriesInstanceUID": series_id,
        **{CLASSES[i]: float(mean_probs[i]) for i in range(len(CLASSES))}
    }])

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to: {output_csv}")
    print("\n" + "="*60)
    print(" 8th PLACE SOLUTION - RSNA PREDICTION RESULTS")
    print("="*60)
    for i, c in enumerate(CLASSES):
        prob = mean_probs[i]
        bar = "█" * int(prob * 30)
        print(f"  {c:<45} {prob:6.4f}  {bar}")
    print("="*60)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8th Place RSNA Aneurysm Inference")
    parser.add_argument("-i", "--input", default="sample_data/series/case_001", help="DICOM series directory")
    parser.add_argument("-m", "--model_dir", default="model", help="Path to models directory")
    parser.add_argument("-f", "--folds", default="0,1,2,5", help="Comma-separated folds to ensemble")
    parser.add_argument("-o", "--output", default="results/predictions.csv", help="Output CSV path")
    parser.add_argument("-d", "--device", default="cuda:0", help="Execution device (cuda:0 or cpu)")
    args = parser.parse_args()

    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    run_inference(args.input, args.model_dir, folds, args.output, device=args.device)
