# 🧬 Manual de Uso: GLIA-Net (Global Localization based Intracranial Aneurysm Network)

## 1. Descripción del Modelo
GLIA-Net es una red de segmentación tridimensional para aneurismas intracraneales en imágenes Angio-TC (CTA) publicada en **Cell Press Patterns** (*Bo et al.*). Emplea un mecanismo dual:
1. **Red de Posicionamiento Global**: Genera mapas de atención globales que guían la localización anatómica.
2. **Red de Segmentación Local**: Segmenta con alta resolución y precisión sub-voxel sobre parches 3D 96x96x96 con ventana deslizante.

- **Repositorio**: `MeteorsHub/GLIA-Net`
- **Checkpoint oficial**: `exp/GLIA-Net/1/ckpt/checkpoint-0245700.pt`
- **SHA-256 verificado**: `e466c0c13280c02f98a73c405d9ec1404d0cbed4fd8b8fa226affcab1f548bfb` (12,900,161 bytes)

---

## 2. Configuración y Entorno Independiente
- **Entorno virtual**: `.venv/` (Python 3.12 con PyTorch CUDA 12.8 para RTX 5070 Ti).
- **Variables de entorno**: `.env`
- **Dependencias**: `requirements.txt` (`medpy`, `colorlog`, `tensorboardX`, `SimpleITK`, `PyYAML`, `scikit-image`, `scipy`).

Para activar el entorno:
```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Preparación de Datos de Entrada
Soporta archivos individuales o carpetas en formato **NIfTI** (`.nii`, `.nii.gz`) o series **DICOM** (`.dcm`).
- Ejemplo incluido: `sample_data/case_001.nii.gz`

---

## 4. Ejecución de la Inferencia

### Opción A: Script PowerShell (Recomendado en Windows)
```powershell
.\run_inference.ps1
```

### Opción B: Script Python
```powershell
.\.venv\Scripts\python.exe inference.py -c inference_GLIA-Net -i sample_data/case_001.nii.gz -t nii -o results -d 0
```
Parámetros disponibles:
- `-c`: Nombre de la configuración (`inference_GLIA-Net`).
- `-i`: Archivo o carpeta de entrada.
- `-t`: Tipo de archivo (`nii` o `dcm`).
- `-o`: Carpeta de destino de resultados.
- `-d`: Dispositivo de cómputo (`0` para GPU CUDA, o `cpu`).
- `-b`: Guardar máscara binaria (`true` / `false`).
- `-p`: Guardar mapa de probabilidades flotante (`true` / `false`).

---

## 5. Estructura de la Carpeta Results
Tras la inferencia, se genera en `results/`:
- `results/<caso>.nii.gz`: Máscara 3D en formato NIfTI donde `1` corresponde a la segmentación del aneurisma y `0` al tejido circundante.
- Si se activa el refinamiento opcional:
  ```powershell
  .\.venv\Scripts\python.exe refine_segmentation.py -i results -o results/refined
  ```
