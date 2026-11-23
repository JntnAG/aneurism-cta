import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Vessel-aware Aneurysm Detection Inference Runner")
    parser.add_argument("--scan_dir", default="sample_data/crop_0.4", help="Directory with cropped CTA scans (.nii.gz)")
    parser.add_argument("--edt_dir", default="sample_data/crop_0.4_vessel_edt", help="Directory with vessel EDT scans (.nii.gz)")
    parser.add_argument("--output_dir", default="results", help="Directory where results will be stored")
    parser.add_argument("--checkpoint", default="final", help="Checkpoint name (default: final)")
    parser.add_argument("--config", default="configs/deform/decoder_only_no_rec_pe_edt.yaml", help="Config file")
    parser.add_argument("--threshold", default="0.95", help="Confidence threshold for postprocessing NIfTI filtering")
    parser.add_argument("--num_gpus", default="1", help="Number of GPUs to use")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.resolve()
    python_exe = project_dir / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = "python"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_dir) + os.pathsep + env.get("PYTHONPATH", "")

    print("=" * 70)
    print(" VESSEL-AWARE MULTI-SCALE DEFORMABLE 3D ATTENTION DETECTION")
    print("=" * 70)
    print(f"Scans Dir   : {args.scan_dir}")
    print(f"EDT Dir     : {args.edt_dir}")
    print(f"Output Dir  : {args.output_dir}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Threshold   : {args.threshold}")
    print("=" * 70)

    # Step 1: Run neural network inference
    print("\n[Step 1/2] Running 3D multi-scale deformable attention model inference...")
    cmd_infer = [
        str(python_exe), "src/train_net.py",
        "--num-gpus", str(args.num_gpus),
        "--config-file", args.config,
        "--eval-only",
        "MODEL.WEIGHTS", args.checkpoint,
        "DATA.DIR.VAL.SCAN_DIR", args.scan_dir,
        "DATA.DIR.VAL.VESSEL_DIR", args.edt_dir,
        "OUTPUT_DIR", args.output_dir
    ]
    ret = subprocess.run(cmd_infer, cwd=str(project_dir), env=env)
    if ret.returncode != 0:
        print("Inference step failed.")
        sys.exit(ret.returncode)

    # Step 2: Postprocessing CSV detections to NIfTI masks
    print("\n[Step 2/2] Generating thresholded NIfTI candidate detection masks...")
    cmd_post = [
        str(python_exe), "src/postprocess/csv_to_nifti.py",
        "--config-file", args.config,
        "POSTPROCESS.CHECKPOINT", args.checkpoint,
        "POSTPROCESS.THRESHOLD", str(args.threshold),
        "DATA.DIR.VAL.SCAN_DIR", args.scan_dir
    ]
    subprocess.run(cmd_post, cwd=str(project_dir), env=env)
    print("\n" + "=" * 70)
    print(" INFERENCE AND POSTPROCESSING COMPLETED SUCCESSFULLY!")
    print(f" Detection CSV and NIfTI files are stored in: {args.output_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
