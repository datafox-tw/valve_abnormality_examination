# aicup25
### datapath

```
aicup25/
├── train/
│   ├── imagesTr/
│   │   ├── patient0001.nii.gz
│   │   ├── ...
│   │   └── patient0050.nii.gz
│   └── labelsTr/
│       ├── patient0001_gt.nii.gz
│       ├── ...
│       └── patient0050_gt.nii.gz
├── test/
│   ├── patient0051.nii.gz
│   ├── ...
│   └── patient0100.nii.gz
├── MAE.py
├── FT.py
├── analyze_dataset.py
├── visualize_data.py
├── pipeline_generation.py

├── requirements.txt
└── readme.md
├── visualization_output/
│   ├── dataset_analysis_report.txt
│   ├── intensity_histogram.png
│   ├── orthographic_slices.png
│   ├── orthographic_slices_with_labels.png
│   ├── pipeline_overview.png
│   └── 3d_surface.png


```

### Download and Installation
1. **Clone the repository:**

    ```sh
    git clone git@github.com:kuanlee2001/aicup25.git
    
    git clone git@github.com:MIC-DKFZ/nnssl.git
    cd nnssl
    pip install -e .

    git clone git@github.com:TaWald/nnUNet.git
    cd nnUNet
    pip install -e .

    cd aicup25
    ```

2. **Fix Source code:**

    ```sh
    nnssl/src/nnssl/training/nnsslTrainer/masked_image_modeling/SparkTrainer.py

    line 3
    (add)
    import dataclasses

    line 131 (class SparkMAETrainer(BaseMAETrainer))
    (add)
    "nnssl_adaptation_plan": dataclasses.asdict(self.adaptation_plan)

    line 260 (class SparkMAETrainer5epBS10(SparkMAETrainer5ep))
    (fix)
    # self.total_batch_size = 10
    self.total_batch_size = 1

    &&

    nnUNet/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py
    line 457-463
    (fix)
    lowres_spacing = np.array(lowres_spacing)  # Ensure it's a NumPy array
    if np.any((max_spacing / lowres_spacing) > 2):
        # lowres_spacing = np.array(lowres_spacing)  # Ensure it's a NumPy array
        max_spacing = np.max(lowres_spacing)
        mask = (max_spacing / lowres_spacing) > 2
        lowres_spacing[mask] *= spacing_increase_factor
    else:
        lowres_spacing *= spacing_increase_factor

    &&
    
    nnUNet/nnunetv2/inference/predict_from_raw_data.py
    line 104-111
    (fix)
            network = trainer_class.build_network_architecture(
            architecture_class_name=configuration_manager.network_arch_class_name,
            arch_init_kwargs=configuration_manager.network_arch_init_kwargs,
            arch_init_kwargs_req_import=configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels=num_input_channels,
            num_output_channels=plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

    &&
    
    nnUNet/nnunetv2/training/nnUNetTrainer/pretraining/pretrainedTrainer.py
    line 80
    (fix)
                    # input_patch_size=self.configuration_manager.patch_size,  # Set in plan to pt_recommended_patchsize

    line 330
    (fix)
            # input_patch_size: tuple[int, int, int],

    line 366
    (add)
            elif architecture_class_name in ["PrimusS", "PrimusM", "PrimusL", "PrimusB", "ResEncL"]:
                if arch_init_kwargs is None:
                    arch_init_kwargs = {}

                # 防護 2: 嘗試讀取 patch_size，如果讀不到，給予預設值
                # 注意：這裡假設是 (128, 128, 128)，這對大多數心臟分割任務是安全的
                if 'patch_size' in arch_init_kwargs:
                    input_patch_size = arch_init_kwargs['patch_size']
                else:
                    # 給一個「合理」的預設值以防止 Crash
                    # 如果你的模型其實是 64x64x64 或其他尺寸，可以在這裡修改
                    input_patch_size = (96, 160, 160) 
                    print(f"Warning: patch_size missing in plans. Using default: {input_patch_size}")
                network = get_network_from_name(
                    architecture_class_name, ...)
                ...
    
    ```
3. **Excute:**

    ```sh
    python3 MAE.py
    python3 FT.py
    analyze_dataset.py
    visualize_data.py
    pipeline_generation.py
    ```
analyze_dataset.py：整合整個資料庫的統計數據並繪製成圖檔
visualize_data.py：分析單一資料並繪製切片狀態與整個心臟建模
pipeline_generation.py：生成nnUNet的訓練流程描述