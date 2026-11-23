# 🧠 Manual de Uso: 8th Place Solution - RSNA Intracranial Aneurysm Detection

## 1. Descripción del Modelo
Este proyecto implementa la arquitectura ganadora del 8º lugar en el desafío Kaggle **RSNA Intracranial Aneurysm Detection** (*Konrad Habel*). Es un modelo híbrido en dos etapas:
1. **Etapa 1 (Extractor de características espaciales)**: Backbone `convnext_base.dinov3_lvd1689m` pre-entrenado con DINOv3 y pooling GeM (Generalized Mean) evaluando cortes 2.5D de 448x448 píxeles.
2. **Etapa 2 (Modelo de secuencia temporal)**: Transformer `DeBERTa-v2` que modela la continuidad anatómica longitudinal a lo largo de toda la serie axial del paciente.

- **Repositorio**: `KonradHabel/rsna`
- **Pesos oficiales**: Kaggle `khabel/rsna-2025-weights-4xbase` (`fold-0`, `fold-1`, `fold-2`, `fold-5`)
- **Salida**: Probabilidad global de presencia de aneurisma + 13 probabilidades por localización anatómica arterial.

---

## 2. Configuración y Entorno Independiente
- **Entorno virtual**: `.venv/` (Python 3.12 con PyTorch CUDA 12.8 para RTX 5070 Ti).
- **Variables de entorno**: `.env`
- **Dependencias**: `requirements.txt` (`albumentations`, `dicomsdl`, `timm>=1.0.20`, `torch-ema`, `transformers`, `opencv-python`).

Para activar el entorno:
```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Preparación de Datos de Entrada
El modelo procesa directamente **series DICOM** axiales de tomografía computarizada (archivos `.dcm` con etiquetas DICOM estándar como `ImagePositionPatient`, `PixelSpacing`, etc.).
- Ejemplo incluido: `sample_data/series/case_001/*.dcm` (64 cortes axiales).

---

## 4. Ejecución de la Inferencia

### Opción A: Script PowerShell (Recomendado en Windows)
Ejecuta la inferencia en conjunto (ensemble) con los 4 folds:
```powershell
.\run_inference.ps1
```

### Opción B: Script Python
```powershell
.\.venv\Scripts\python.exe run_inference.py -i sample_data/series/case_001 -m model -f 0,1,2,5 -o results/predictions.csv -d cuda:0
```

---

## 5. Estructura de la Carpeta Results
Tras la inferencia, se genera `results/predictions.csv` con la siguiente información:
- `SeriesInstanceUID`: Identificador de la serie DICOM analizada.
- `Left Infraclinoid Internal Carotid Artery`
- `Right Infraclinoid Internal Carotid Artery`
- `Left Supraclinoid Internal Carotid Artery`
- `Right Supraclinoid Internal Carotid Artery`
- `Left Middle Cerebral Artery`
- `Right Middle Cerebral Artery`
- `Anterior Communicating Artery`
- `Left Anterior Cerebral Artery`
- `Right Anterior Cerebral Artery`
- `Left Posterior Communicating Artery`
- `Right Posterior Communicating Artery`
- `Basilar Tip`
- `Other Posterior Circulation`
- `Aneurysm Present` (Probabilidad global)
