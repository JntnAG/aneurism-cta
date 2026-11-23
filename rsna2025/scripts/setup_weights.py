import os
import shutil
from pathlib import Path

# General portable directories (user home cache and current project logs)
cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "tomoon33"
workspace_logs = Path(__file__).resolve().parent.parent / "logs"

mappings = [
    (
        cache_dir / "nnunet-vessel-grouping-da7" / "versions" / "1",
        workspace_logs / "nnUNet_results" / "Dataset003_VesselGrouping" / "RSNA2025Trainer_moreDAv7__nnUNetResEncUNetMPlans__3d_fullres",
    ),
    (
        cache_dir / "nnunet-da3-sklr-ep800" / "versions" / "1",
        workspace_logs / "nnUNet_results" / "Dataset001_VesselSegmentation" / "nnUNetTrainerSkeletonRecall_more_DAv3_ep800__nnUNetResEncUNetMPlans__3d_fullres",
    ),
    (
        cache_dir / "nnunet-da6-sklr-w3-tv07" / "versions" / "1",
        workspace_logs / "nnUNet_results" / "Dataset001_VesselSegmentation" / "RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres",
    ),
    (
        cache_dir / "nnunet-da6-1-sklr-tv07" / "versions" / "1",
        workspace_logs / "nnUNet_results" / "Dataset001_VesselSegmentation" / "RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres",
    ),
    (
        cache_dir / "251013-prev6-1-ex-dav6w3-e25-w01-005-1-s128-256" / "versions" / "1" / "251013-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256",
        workspace_logs / "train" / "runs" / "251013-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256",
    ),
]

for src, dst in mappings:
    print(f"Syncing from {src} to {dst}...")
    if not src.exists():
        print(f"Warning: source does not exist: {src}")
        continue
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    items = list(dst.rglob("*"))
    print(f"Done: {dst.name} ({len(items)} items)")

print("ALL WEIGHTS SYNCED SUCCESSFULLY!")
