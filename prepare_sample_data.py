import numpy as np
import nibabel as nib
import os
import SimpleITK as sitk

os.makedirs("GLIA_NET/sample_data", exist_ok=True)
os.makedirs("Vessel-aware/sample_data/og", exist_ok=True)
os.makedirs("Vessel-aware/sample_data/crop_0.4", exist_ok=True)
os.makedirs("Vessel-aware/sample_data/crop_0.4_vessel_edt", exist_ok=True)
os.makedirs("8thKonradHabel/sample_data/series/case_001", exist_ok=True)

# Generate realistic 3D CTA volume: 128 x 128 x 64
shape = (128, 128, 64)
volume = np.zeros(shape, dtype=np.float32)

# Background air: -1000 HU
volume[:] = -1000.0

# Head/Brain elliptical region: 35 HU
x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
brain_mask = ((x - center[0])**2 / (45**2) + (y - center[1])**2 / (45**2) + (z - center[2])**2 / (25**2)) <= 1.0
volume[brain_mask] = 35.0

# Skull / bone border: 600 HU
bone_mask = (((x - center[0])**2 / (48**2) + (y - center[1])**2 / (48**2) + (z - center[2])**2 / (27**2)) <= 1.0) & (~brain_mask)
volume[bone_mask] = 600.0

# Circle of Willis vessels: 350 HU
vessel_mask = np.zeros(shape, dtype=bool)
vessel_mask[center[0]-3:center[0]+3, center[1]-20:center[1]+20, center[2]-2:center[2]+2] = True
vessel_mask[center[0]-20:center[0]+20, center[1]-3:center[1]+3, center[2]-2:center[2]+2] = True
# Simulated aneurysm: 400 HU (spherical outpouching at bifurcation)
aneurysm_mask = ((x - (center[0]+5))**2 + (y - (center[1]+5))**2 + (z - center[2])**2) <= 4**2
vessel_mask = vessel_mask | aneurysm_mask
volume[vessel_mask] = 350.0

# Add small noise
volume += np.random.normal(0, 5, shape).astype(np.float32)

affine = np.array([
    [0.4, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Save for GLIA-Net
nii_img = nib.Nifti1Image(volume, affine)
nib.save(nii_img, "GLIA_NET/sample_data/case_001.nii.gz")
print("Saved GLIA_NET/sample_data/case_001.nii.gz")

# Save for Vessel-aware (requires crop_0.4 and EDT distance transform)
nib.save(nii_img, "Vessel-aware/sample_data/og/scan_1.nii.gz")
nib.save(nii_img, "Vessel-aware/sample_data/crop_0.4/scan_1.nii.gz")

import edt
edt_volume = edt.edt(vessel_mask.astype(np.uint32), anisotropy=(0.4, 0.4, 0.4), black_border=True)
edt_img = nib.Nifti1Image(edt_volume.astype(np.float32), affine)
nib.save(edt_img, "Vessel-aware/sample_data/crop_0.4_vessel_edt/scan_1.nii.gz")
print("Saved Vessel-aware scans and EDT")

# Save DICOM series for 8thKonradHabel
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sop_uids = []
for i in range(shape[2]):
    slice_data = volume[:, :, i].astype(np.int16)
    filename = f"8thKonradHabel/sample_data/series/case_001/slice_{i:03d}.dcm"
    
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.Modality = "CT"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7"
    ds.StudyInstanceUID = "1.2.3.4.5.6"
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.InstanceNumber = i + 1
    ds.ImagePositionPatient = [0.0, 0.0, float(i * 0.4)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [0.4, 0.4]
    ds.SliceThickness = 0.4
    ds.RescaleIntercept = 0.0
    ds.RescaleSlope = 1.0
    ds.Rows = shape[0]
    ds.Columns = shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = slice_data.tobytes()
    ds.save_as(filename, write_like_original=False)
    sop_uids.append(ds.SOPInstanceUID)

print(f"Saved {shape[2]} DICOM slices for 8thKonradHabel")
