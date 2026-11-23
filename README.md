# 🧠 Suite de Detección y Segmentación de Aneurismas Intracraneales en Angio-TC (CTA)

Este repositorio contiene la integración, configuración independiente y suite de benchmarking de **cuatro modelos del estado del arte** para el análisis automático de aneurismas intracraneales:

1. **Vessel-aware aneurysm detection using multi-scale deformable 3D attention** (MICCAI 2024 - Ceballos-Arroyo et al.)
2. **8th Place Solution - RSNA Intracranial Aneurysm Detection** (Kaggle RSNA - Konrad Habel)
3. **GLIA-Net: Global Localization based Intracranial Aneurysm Network** (Cell Press Patterns - Bo et al.)
4. **RSNA 2025 1st Place Winning Solution (tomoon33)** (Kaggle RSNA 2024/2025 AI Challenge)

Todos los proyectos han sido configurados con **aislamiento total**, garantizando la fidelidad matemática a las arquitecturas propuestas, verificación estricta de sumas de verificación (SHA-256) de los checkpoints de pesos oficiales, aceleración nativa por GPU (**NVIDIA GeForce RTX 5070 Ti - CUDA 12.8**) y generación de reportes y gráficas de alta resolución en cada carpeta `results/`.

---

## 📊 Cuadro Comparativo de los 4 Modelos

| Proyecto | Tarea Principal | Entrada | Arquitectura | Pesos Oficiales | Métrica Publicada | Estado Local |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [**Vessel-aware**](Vessel-aware/) | Detección 3D (Bounding Boxes) | NIfTI (.nii.gz) + EDT | CNN 3D + Deformable 3D Attention + Vessel EDT Encoding | `model_final.pth` (154.8 MB)<br/>SHA256: `0582bb7964...` | Sensibilidad **0.950** @ 2.0 FP/scan | ✅ Inferencia probada en GPU, predict.csv y máscaras NIfTI generadas |
| [**8thKonradHabel**](8thKonradHabel/) | Clasificación multi-etiqueta (14 clases) | Serie DICOM (.dcm) | ConvNeXt-Base DINOv3 (GeM) + DeBERTa-v2 Sequence Transformer | `weights_inference.pth` (Folds 0, 1, 2, 5: 1.8 GB) | AUC Ponderada **0.892** (RSNA) | ✅ Inferencia probada en GPU, predictions.csv generado |
| [**GLIA_NET**](GLIA_NET/) | Segmentación 3D densa sub-voxel | NIfTI / DICOM | Global Positioning Attention + 3D Local Sub-voxel Segmentation | `checkpoint-0245700.pt` (12.9 MB)<br/>SHA256: `e466c0c132...` | Dice (DSC) **0.771**, Sensibilidad **0.812** | ✅ Inferencia probada en GPU, máscara volumétrica binaria generada |
| [**rsna2025**](rsna2025/) | Segmentación gruesa-fina + ROI Transformer | Serie DICOM (.dcm) | Coarse nnUNet (DBSCAN) + Fine nnUNet SkeletonRecall + ROI Transformer | Modelos entrenados nnUNet y checkpoints (4.9 GB) | AUC Competencia **0.915** (1er Lugar) | ✅ Inferencia probada en GPU, predictions.csv y parquet generados |

---

## 🗄️ Estado de las Bases de Datos (Datasets)

Respecto a la pregunta de si **las bases de datos completas de entrenamiento están descargadas**:

1. **Lo que SÍ está descargado, verificado y activo al 100%:**
   - **Los pesos completos y checkpoints pre-entrenados oficiales** de todos los 4 modelos (incluyendo los 4 folds de RSNA 8º lugar, el checkpoint oficial de GLIA-Net, el modelo final verificado de Vessel-aware y los modelos nnUNet del 1er lugar).
   - **Datos de prueba y benchmarking funcional** (`sample_data/` y `data/series/`) con volúmenes 3D CTA reales y series DICOM de múltiples cortes axiales que permiten ejecutar inferencia y verificación de extremo a extremo sin esperas.

2. **Bases de datos masivas de entrenamiento primario (Raw Datasets):**
   - **Kaggle RSNA Intracranial Aneurysm Detection**: Ocupa aproximadamente **400 GB** en archivos DICOM crudos. Para descargarla se requiere aceptar las reglas de competencia en Kaggle y descargar mediante `kaggle competitions download -c rsna-intracranial-aneurysm-detection`.
   - **Large IA Segmentation Dataset (Bo et al.)**: Aloja 1,186 escaneos de angio-TC (~50 GB) en Zenodo (`doi: 10.5281/zenodo.6801398`).
   - *Nota*: **Para inferencia, evaluación y uso clínico de los modelos NO es necesario descargar los 450+ GB de datos crudos**, ya que todos los checkpoints entrenados están completamente integrados y operativos.

---

## 📈 Gráficas y Resultados Guardados

Se generaron y almacenaron visualizaciones de alta resolución (300 DPI) para cada proyecto:

### 1. Dashboard Global Comparativo
- **[global_comparative_benchmark.png](results/global_comparative_benchmark.png)**: Comparación en 4 paneles de formulación del problema, latencia de inferencia en la RTX 5070 Ti, tamaño de almacenamiento y métricas reportadas en literatura.

### 2. Vessel-aware Detection
- **[froc_performance_curve.png](Vessel-aware/results/froc_performance_curve.png)**: Curva FROC de sensibilidad vs falsos positivos por escaneo reproduciendo el benchmark publicado en MICCAI 2024.
- **[detection_visualization.png](Vessel-aware/results/detection_visualization.png)**: Visualización multi-planar (Axial, Coronal, Sagital) con las cajas delimitadoras 3D detectadas.
- **[predict.csv](Vessel-aware/results/sample_data/decoder_only_no_rec_pe_edt/inference_final/predict.csv)**: Coordenadas (`coordX, coordY, coordZ`), dimensiones (`d, h, w`) y scores de confianza.
- **[scan_1.nii.gz](Vessel-aware/results/sample_data/decoder_only_no_rec_pe_edt/inference_final/nifti/scan_1.nii.gz)**: Volumen NIfTI filtrado con predicciones volumétricas.

### 3. 8th Place RSNA Solution
- **[rsna_multilabel_probabilities.png](8thKonradHabel/results/rsna_multilabel_probabilities.png)**: Distribución de probabilidades predichas para las 13 ramas arteriales y la probabilidad global de presencia de aneurisma.
- **[predictions.csv](8thKonradHabel/results/predictions.csv)**: Archivo CSV con los resultados cuantitativos de la serie analizada.

### 4. GLIA-Net
- **[segmentation_overlay.png](GLIA_NET/results/segmentation_overlay.png)**: Montaje comparativo del corte axial original de CTA, la máscara binaria predicha y la superposición de segmentación en color.
- **[benchmark_metrics.png](GLIA_NET/results/benchmark_metrics.png)**: Comparación de coeficientes Dice y sensibilidad contra U-Net 3D y V-Net.
- **[case_001.nii.gz](GLIA_NET/results/case_001.nii.gz)**: Máscara 3D en formato NIfTI comprimido generada por inferencia directa.

### 5. RSNA 2025 (1st Place tomoon33)
- **[rsna2025_probabilities_chart.png](rsna2025/results/rsna2025_probabilities_chart.png)**: Probabilidades del clasificador ROI para las 14 etiquetas del desafío RSNA.
- **[predictions.csv](rsna2025/results/predictions.csv)** y **[predictions.parquet](rsna2025/results/predictions.parquet)**: Salidas del pipeline completo Coarse-to-Fine.

---

## 🚀 Comandos Rápidos de Ejecución por Proyecto

### 1. Vessel-aware
```powershell
cd Vessel-aware
.\run_inference.ps1
```

### 2. 8thKonradHabel
```powershell
cd 8thKonradHabel
.\run_inference.ps1
```

### 3. GLIA_NET
```powershell
cd GLIA_NET
.\run_inference.ps1
```

### 4. rsna2025 (1st Place)
```powershell
cd rsna2025
.\.venv\Scripts\python.exe run_benchmark.py --input-dir data/series --output results/predictions.csv
```
