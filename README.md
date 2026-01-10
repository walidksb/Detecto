# 📌 Project Title
3D Crack Localization via AI and Multi-View Reconstruction

# 🧠 Motivation
Manual inspection of structures is subjective and difficult to quantify, while traditional 2D crack detection methods lack spatial context. This project proposes a 3D inspection pipeline that combines AI-based crack detection with geometric reconstruction to enable spatial localization and analysis of surface defects.

# 🎯 Objectives
- Detect surface cracks using deep learning
- Reconstruct inspected surfaces in 3D
- Localize cracks directly in 3D space
- Fuse evidence from multiple views
- Robustly fuse evidence from multiple views

# System Overview
### Architecture
![Alt text](images/system_architechture.png)

### Point Cloud Results
![Alt text](images/crack_detection_3d_point_cloud.png)
3D point cloud with crack localization obtained by projecting 2D crack predictions into the reconstructed geometry.

### Multi-view Point Cloud Results
![Alt text](images/multiview_crack_detection_3d_point_cloud.png)
Multi-view 3D point cloud with crack localization obtained by projecting 2D crack predictions into the reconstructed geometry.

# Report Example ".json" file
```
{
  "scene_status": "DAMAGED",
  "num_detected_cracks": 6,
  "inspection_parameters": {
    "confidence_threshold": 0.6,
    "min_crack_length": 1.0,
    "min_cluster_points": 10
  },
  "cracks": [
    {
      "id": 0,
      "length": 3.66,
      "num_points": 213,
      "center_3d": [
        10.057,
        5.434,
        15.151
      ]
    },
    {
      "id": 1,
      "length": 2.26,
      "num_points": 76,
      "center_3d": [
        3.055,
        5.303,
        15.335
      ]
    },
    ....
```
# Modules
- **Reconstruction:** COLMAP-based multi-view 3D reconstruction
- **Detection:** U-Net-based crack segmentation on 2D images
- **Fusion:** Projection of 2D predictions into 3D space and multi-view aggregation 
- **Reporting** A generated report .json showing the cracks details

# Methodology
1. Multi-view images are captured
2. COLMAP reconstructs geometry and camera poses
3. Crack detection is applied per image using a pretrained AI model
4. Predictions are projected into 3D
5. Multi-view fusion builds a 3D crack heatmap
6. Reporting

# Scope & Limitations
* Offline inspection (not real-time)
* Requires sufficient image overlap
* Detection accuracy depends on training data

- contact : walid.kesbi@gmail.com
- Developed by Walid.
