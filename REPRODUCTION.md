# 🛠 Technical Setup and Reproduction Guide

To reproduce the results obtained in the AI Cup 2025, specific patches are required for the `nnssl` and `nnUNet` frameworks. This guide outlines the environment configuration and the source code modifications.

---

## 🏗 Environment Configuration

### 1. Clone Repositories
```sh
git clone git@github.com:Sebastian-0912/NTU_CVPDL_2025_FINAL.git
git clone git@github.com:MIC-DKFZ/nnssl.git
git clone git@github.com:TaWald/nnUNet.git

# Install in editable mode
cd nnssl && pip install -e . && cd ..
cd nnUNet && pip install -e . && cd ..
```

### 2. Required Source Code Patches

Due to specific requirements for 3D medical image segmentation on heart CT/MRI data, the following patches must be applied to the cloned libraries.

#### **Patch 1: `nnssl` - Spark Trainer Modification**
Target File: `nnssl/src/nnssl/training/nnsslTrainer/masked_image_modeling/SparkTrainer.py`

- **Line 3**: Add `import dataclasses`
- **Line 131**: Inside `class SparkMAETrainer(BaseMAETrainer)`, add:
  ```python
  "nnssl_adaptation_plan": dataclasses.asdict(self.adaptation_plan)
  ```
- **Line 260**: In `SparkMAETrainer5epBS10`, change `total_batch_size` to 1 if memory is limited:
  ```python
  self.total_batch_size = 1
  ```

#### **Patch 2: `nnUNet` - Experiment Planner Fix**
Target File: `nnUNet/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py`

- **Lines 457-463**: Ensure `lowres_spacing` is handled as a NumPy array:
  ```python
  lowres_spacing = np.array(lowres_spacing)
  if np.any((max_spacing / lowres_spacing) > 2):
      max_spacing = np.max(lowres_spacing)
      mask = (max_spacing / lowres_spacing) > 2
      lowres_spacing[mask] *= spacing_increase_factor
  else:
      lowres_spacing *= spacing_increase_factor
  ```

#### **Patch 3: `nnUNet` - Prediction Logic**
Target File: `nnUNet/nnunetv2/inference/predict_from_raw_data.py`

- **Lines 104-111**: Update the architecture building logic to disable deep supervision during inference:
  ```python
  network = trainer_class.build_network_architecture(
      architecture_class_name=configuration_manager.network_arch_class_name,
      arch_init_kwargs=configuration_manager.network_arch_init_kwargs,
      arch_init_kwargs_req_import=configuration_manager.network_arch_init_kwargs_req_import,
      num_input_channels=num_input_channels,
      num_output_channels=plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
      enable_deep_supervision=False
  )
  ```

#### **Patch 4: `nnUNet` - Residual Encoder Support**
Target File: `nnUNet/nnunetv2/training/nnUNetTrainer/pretraining/pretrainedTrainer.py`

- Add support for `Primus` and `ResEncL` classes and handle missing `patch_size` gracefully by providing a default (e.g., `(96, 160, 160)`).

---

## 🏃 Execution Steps

Once the environment is patched, run the following scripts in order:

1. **Dataset Analysis**:
   ```sh
   python3 analyze_dataset.py
   ```
2. **MAE Pre-training**:
   ```sh
   python3 MAE.py
   ```
3. **Fine-tuning**:
   ```sh
   python3 FT.py
   ```
4. **Visualization**:
   ```sh
   python3 visualize_data.py
   python3 pipeline_generation.py
   ```
