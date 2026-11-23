# 🔬 Manual de Uso: Vessel-Aware 3D Aneurysm Detection

## 1. Descripción del Modelo
Este proyecto implementa la arquitectura publicada en **MICCAI 2024** (*Ceballos-Arroyo et al.*) para detección de aneurismas intracraneales utilizando atención deformable multi-escala en 3D (`PARQ_Deformable_R`) combinada con codificación posicional basada en la Transformada de Distancia Euclidiana (EDT) de la vasculatura cerebral.

- **Repositorio**: `alceballosa/deform-aneurysm-detection`
- **Commit fijado**: `42cd3ba431938093305d509a44a493fb2ea6e7c0`
- **Checkpoint**: `model_weights/decoder_only_no_rec_pe_edt/model_final.pth`
- **SHA-256 verificado**: `0582bb7964960275a41f28484763a65d99176dcea3a607bb2c6912e3b067f75d` (154,762,479 bytes)

---

## 2. Configuración y Entorno Independiente
- **Entorno virtual**: `.venv/` (Python 3.12 con PyTorch CUDA 12.8 para RTX 5070 Ti).
- **Variables de entorno**: `.env`
- **Dependencias**: `requirements.txt` (incluye `detectron2`, `torchio==0.20.2`, `edt`, `yacs`, `fvcore`, `nibabel`, `SimpleITK`).

Para activar el entorno:
```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Preparación de Datos de Entrada
Para procesar un nuevo escaneo CTA, se requieren dos volúmenes NIfTI:
1. **Escaneo CTA re-muestreado a 0.4 mm** (carpeta `crop_0.4/`): e.g. `scan_1.nii.gz`
2. **Transformada de distancia euclidiana de la máscara vascular** (carpeta `crop_0.4_vessel_edt/`): e.g. `scan_1.nii.gz`

---

## 4. Ejecución de la Inferencia

### Opción A: Script PowerShell (Recomendado en Windows)
```powershell
.\run_inference.ps1
```

### Opción B: Script Python
```powershell
.\.venv\Scripts\python.exe run_inference.py --scan_dir sample_data/crop_0.4 --edt_dir sample_data/crop_0.4_vessel_edt --output_dir results --threshold 0.1
```

### Opción C: Comando Detectron2 Original
```powershell
$env:PYTHONPATH = ".;$env:PYTHONPATH"
.\.venv\Scripts\python.exe src/train_net.py --num-gpus 1 --config-file configs/deform/decoder_only_no_rec_pe_edt.yaml --eval-only MODEL.WEIGHTS final DATA.DIR.VAL.SCAN_DIR sample_data/crop_0.4 DATA.DIR.VAL.VESSEL_DIR sample_data/crop_0.4_vessel_edt OUTPUT_DIR results
.\.venv\Scripts\python.exe src/postprocess/csv_to_nifti.py --config-file configs/deform/decoder_only_no_rec_pe_edt.yaml POSTPROCESS.CHECKPOINT final POSTPROCESS.THRESHOLD 0.1 DATA.DIR.VAL.SCAN_DIR sample_data/crop_0.4
```

---

## 5. Estructura de la Carpeta Results
Tras la ejecución, la carpeta `results/` contiene:
- `results/sample_data/decoder_only_no_rec_pe_edt/inference_final/predict.csv`: Contiene todos los candidatos detectados con sus coordenadas 3D (`coordX, coordY, coordZ`), dimensiones (`d, h, w`) y probabilidad calculada por el modelo.
- `results/sample_data/decoder_only_no_rec_pe_edt/inference_final/nifti/<scan>.nii.gz`: Máscara 3D en formato NIfTI donde las cajas delimitadoras de los candidatos detectados por encima del umbral tienen valores de intensidad proporcionales a su confianza.
