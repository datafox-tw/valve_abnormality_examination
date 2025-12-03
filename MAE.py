import os
import shutil
import subprocess
import json
import torch
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.adaptation_planning.adaptation_plan import DynamicArchitecturePlans, ArchitecturePlans

# -------------------------------
# 1. 設定 nnSSL 路徑變數
# -------------------------------
def setup_nnssl_paths():
    """
    設定 nn-SSL 所需的環境變數。
    """
    base_dir = os.path.abspath("./nnssl_workdir")

    os.environ["nnssl_raw"] = os.path.join(base_dir, "nnssl_raw")
    os.environ["nnssl_preprocessed"] = os.path.join(base_dir, "nnssl_preprocessed")
    os.environ["nnssl_results"] = os.path.join(base_dir, "nnssl_results")

    os.makedirs(os.environ["nnssl_raw"], exist_ok=True)
    os.makedirs(os.environ["nnssl_preprocessed"], exist_ok=True)
    os.makedirs(os.environ["nnssl_results"], exist_ok=True)

    print(f"✅ nnssl_raw = {os.environ['nnssl_raw']}")
    print(f"✅ nnssl_preprocessed = {os.environ['nnssl_preprocessed']}")
    print(f"✅ nnssl_results = {os.environ['nnssl_results']}")


# -------------------------------
# 2. 準備資料結構
# -------------------------------
def prepare_dataset(task_id, task_name, train_dir, test_dir):
    dataset_name = f"Dataset{task_id:03d}_{task_name}"
    dataset_root = os.path.join(os.environ["nnssl_raw"], dataset_name)
    target_images_tr = os.path.join(dataset_root, "imagesTr")
    target_labels_tr = os.path.join(dataset_root, "labelsTr")
    target_images_ts = os.path.join(dataset_root, "imagesTs")

    os.makedirs(target_images_tr, exist_ok=True)
    os.makedirs(target_labels_tr, exist_ok=True)
    os.makedirs(target_images_ts, exist_ok=True)

    print("📦 Copying training data...")
    for f in os.listdir(os.path.join(train_dir, "imagesTr")):
        shutil.copy(os.path.join(train_dir, "imagesTr", f), target_images_tr)
    for f in os.listdir(os.path.join(train_dir, "labelsTr")):
        shutil.copy(os.path.join(train_dir, "labelsTr", f), target_labels_tr)

    print("📦 Copying test data...")
    for f in os.listdir(test_dir):
        if f.endswith(".nii.gz"):
            shutil.copy(os.path.join(test_dir, f), target_images_ts)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "label_1": 1, "label_2": 2, "label_3": 3},
        "numTraining": len(os.listdir(target_images_tr)),
        "file_ending": ".nii.gz",
        "name": task_name,
    }

    with open(os.path.join(dataset_root, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"✅ dataset.json created at: {dataset_root}")


# -------------------------------
# 3. Shell 指令執行工具
# -------------------------------
def run_command(command):
    print(f"\n🚀 Running: {command}\n")
    env = os.environ.copy()  # 把當前環境變數傳給 subprocess
    process = subprocess.Popen(command, shell=True, text=True, env=env)
    process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")



# -------------------------------
# 4. 主流程
# -------------------------------
def main():
    TASK_ID = 1
    TASK_NAME = "AICUP_Cardiac"
    TRAIN_DATA_DIR = "./train"
    TEST_DATA_DIR = "./test"

    print("=== Step 1: 設定 nnSSL 環境變數 ===")
    setup_nnssl_paths()

    print("\n=== Step 2: 建立資料結構 ===")
    prepare_dataset(TASK_ID, TASK_NAME, TRAIN_DATA_DIR, TEST_DATA_DIR)

    DATASET_DIR = "./nnssl_workdir/nnssl_raw/Dataset001_AICUP_Cardiac"
    LABELS_DIR = os.path.join(DATASET_DIR, "labelsTr")
    IMAGES_TR_DIR = os.path.join(DATASET_DIR, "imagesTr")
    IMAGES_TS_DIR = os.path.join(DATASET_DIR, "imagesTs")
    PRETRAIN_JSON_PATH = os.path.join(DATASET_DIR, "pretrain_data.json")

    def create_subject_dict(image_file, label_file=None):
        session_id = "session_0"
        img_entry = {
            "image_path": image_file,
            "name": os.path.basename(image_file),
            "modality": "CT",
            "associated_masks": {"anatomy_mask": label_file, "anonymization_mask": None}
        }
        subject_id = os.path.basename(image_file).split("_")[0]
        return {
            "subjects": {
                subject_id: {
                    "subject_info": None,
                    "sessions": {
                        session_id: {
                            "session_info": None,
                            "images": [img_entry]
                        }
                    }
                }
            }
        }

    # -----------------------
    # 修改 labelsTr 檔名 (_gt -> _0000)
    # -----------------------
    for f in os.listdir(LABELS_DIR):
        if f.endswith("_gt.nii.gz"):
            old_path = os.path.join(LABELS_DIR, f)
            new_name = f.replace("_gt.nii.gz", "_0000.nii.gz")
            new_path = os.path.join(LABELS_DIR, new_name)
            os.rename(old_path, new_path)

    # -----------------------
    # 修改 imagesTr 檔名 (加上 _0000)
    # -----------------------
    for f in os.listdir(IMAGES_TR_DIR):
        if f.endswith(".nii.gz") and not f.endswith("_0000.nii.gz"):
            old_path = os.path.join(IMAGES_TR_DIR, f)
            new_name = f.replace(".nii.gz", "_0000.nii.gz")
            new_path = os.path.join(IMAGES_TR_DIR, new_name)
            os.rename(old_path, new_path)

    # -----------------------
    # 建立 JSON 結構（修正版：以 subject_id 配對）
    # -----------------------
    pretrain_data = {
        "collection_index": 0,
        "collection_name": "Dataset001_AICUP_Cardiac",
        "datasets": {}  # 必須用 str(dataset_index) 作 key
    }

    # ============================================================
    # 🟩 Train dataset (dataset_index 0)
    # ============================================================
    train_dataset = {"dataset_index": 0, "split": "train", "subjects": {}}

    # 建立 label dict (key = subject_id, value = label 路徑)
    label_dict = {}
    for lbl_file in sorted(os.listdir(LABELS_DIR)):
        if not lbl_file.endswith(".nii.gz"):
            continue
        subject_id = lbl_file.split("_")[0]
        label_path = os.path.join(LABELS_DIR, lbl_file)
        label_dict[subject_id] = label_path

    # 以 subject_id 配對影像與標籤
    for img_file in sorted(os.listdir(IMAGES_TR_DIR)):
        if not img_file.endswith(".nii.gz"):
            continue
        subject_id = img_file.split("_")[0]
        img_path = os.path.join(IMAGES_TR_DIR, img_file)
        lbl_path = label_dict.get(subject_id, None)  # 找不到就設為 None（理論上不會發生）
        subject_dict = create_subject_dict(img_path, lbl_path)
        train_dataset["subjects"].update(subject_dict["subjects"])

    pretrain_data["datasets"]["0"] = train_dataset  # 用 "0" 作 key

    # ============================================================
    # 🟦 Test dataset (dataset_index 1, no labels)
    # ============================================================
    test_dataset = {"dataset_index": 1, "split": "test", "subjects": {}}

    for img_file in sorted(os.listdir(IMAGES_TS_DIR)):
        if not img_file.endswith(".nii.gz"):
            continue
        img_path = os.path.join(IMAGES_TS_DIR, img_file)
        subject_dict = create_subject_dict(img_path, None)
        test_dataset["subjects"].update(subject_dict["subjects"])

    pretrain_data["datasets"]["1"] = test_dataset  # 用 "1" 作 key

    # ============================================================
    # 💾 儲存 JSON
    # ============================================================
    with open(PRETRAIN_JSON_PATH, "w") as f:
        json.dump(pretrain_data, f, indent=4)

    print(f"✅ pretrain_data.json created at {PRETRAIN_JSON_PATH}")

    print("\n=== Step 3: 資料規劃與前處理 ===")
    run_command(f"nnssl_plan_and_preprocess -d {TASK_ID} ")



    # === Step 4: 自監督預訓練 (nnssl) ===
    run_command(f"nnssl_train {TASK_ID} onemmiso -tr SparkMAETrainer5epBS10 -p nnsslPlans -num_gpus 1 -device cuda")

    pretrained_model_path = "/home/ai2lab/Desktop/aicup25/nnssl_workdir/nnssl_results/Dataset001_AICUP_Cardiac/SparkMAETrainer5epBS10__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth"
    print(f"✅ 預訓練完成: {pretrained_model_path}")

    # # === Step 5: 有監督微調 (Fine-tuning) ===
    # # 1️⃣ 建 downstream network
    # arch_kwargs = DynamicArchitecturePlans(
    #     n_stages=6,
    #     features_per_stage=[32, 64, 128, 256, 320, 480],
    #     conv_op=torch.nn.Conv3d,
    #     kernel_sizes=[(3,3,3)]*6,
    #     strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
    #     n_blocks_per_stage=[1,3,4,6,6,8],
    #     n_conv_per_stage_decoder=[1,1,1,1,1],
    #     conv_bias=True,
    #     norm_op=torch.nn.InstanceNorm3d,
    #     norm_op_kwargs={"eps": 1e-5, "affine": True},
    #     dropout_op=None,
    #     dropout_op_kwargs=None,
    #     nonlin=torch.nn.LeakyReLU,
    #     nonlin_kwargs={"inplace": True},
    # )
    # arch_plans = ArchitecturePlans(arch_class_name="ResidualEncoderUNet", arch_kwargs=arch_kwargs)

    # network = get_network_from_plans(
    #     arch_class_name=arch_plans.arch_class_name,
    #     arch_kwargs=arch_plans.arch_kwargs.serialize(),
    #     arch_kwargs_req_import=arch_plans.arch_kwargs_requiring_import,
    #     input_channels=1,
    #     output_channels=2,
    #     allow_init=False,
    #     deep_supervision=False,
    # )

    # # 2️⃣ 載入 MAE encoder 權重
    # pretrained_ckpt = torch.load(pretrained_model_path, map_location="cuda")
    # encoder_keys = {k:v for k,v in pretrained_ckpt.items() if k.startswith("encoder.")}
    # network.encoder.load_state_dict(encoder_keys, strict=False)

    # # 3️⃣ 再用 nnSSL trainer 訓練
    # # 這裡用正常 trainer 取代 CLI
    # trainer_cmd = f"nnssl_train {TASK_ID} onemmiso -tr SparkMAETrainer5epBS10 -p nnsslPlans -num_gpus 1 -device cuda"
    # run_command(trainer_cmd)


    # print("\n=== Step 6: 推論 (Inference) ===")
    # inference_dir = "./inference_results"
    # os.makedirs(inference_dir, exist_ok=True)
    # run_command(
    #     f"nnssl_predict -d {TASK_ID} "
    #     f"-i {os.path.join(os.environ['nnssl_raw'], f'Dataset{TASK_ID:03d}_{TASK_NAME}', 'imagesTs')} "
    #     f"-o {inference_dir} -f all -c 3d_fullres"
    # )

    # print("\n✅ 全流程完成！")
    # print(f"🧩 預訓練模型: {os.path.dirname(pretrained_model_path)}")
    # print(f"🧩 微調結果: {os.environ['nnssl_results']}")
    # print(f"📂 推論結果: {inference_dir}")


if __name__ == "__main__":
    main()
