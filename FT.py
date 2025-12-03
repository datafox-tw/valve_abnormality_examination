import os
import json
import shutil
import subprocess

# =====================================
# ✳️ 路徑設定（請依實際情況修改這些變數）
# =====================================
TRAIN_IMAGES = "/home/ai2lab/Desktop/aicup25/train/imagesTr"
TRAIN_LABELS = "/home/ai2lab/Desktop/aicup25/train/labelsTr"
TEST_IMAGES = "/home/ai2lab/Desktop/aicup25/test"
OUTPUT_RAW = "/home/ai2lab/Desktop/aicup25/nnUNet_raw"
TASK_ID = 3
DATASET_NAME = f"Dataset{TASK_ID:03d}_AICUP_Cardiac_FT"
OUTPUT_BASE = os.path.join(OUTPUT_RAW, DATASET_NAME)

PRETRAINED_CHECKPOINT = (
    "/home/ai2lab/Desktop/aicup25/nnssl_workdir/nnssl_results/"
    "Dataset001_AICUP_Cardiac/SparkMAETrainer5epBS10__nnsslPlans__onemmiso/"
    "fold_all/checkpoint_final.pth"
)
def run_command(command):
    print(f"\n🚀 Running: {command}\n")
    env = os.environ.copy()  # 把當前環境變數傳給 subprocess
    process = subprocess.Popen(command, shell=True, text=True, env=env)
    process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")
# =====================================
# ⚙️ 設定環境變數
# =====================================
env_vars = {
    "nnUNet_raw": OUTPUT_RAW,
    "nnUNet_preprocessed": "/home/ai2lab/Desktop/aicup25/nnUNet_preprocessed",
    "nnUNet_results": "/home/ai2lab/Desktop/aicup25/nnUNet_results",
    "nnssl_raw": "/home/ai2lab/Desktop/aicup25/nnssl_workdir/nnssl_raw",
    "nnssl_preprocessed": "/home/ai2lab/Desktop/aicup25/nnssl_workdir/nnssl_preprocessed",
    "nnssl_results": "/home/ai2lab/Desktop/aicup25/nnssl_workdir/nnssl_results",
}
os.environ.update(env_vars)
print("✅ Environment variables set.")

# =====================================
# 🧱 建立資料夾結構
# =====================================
imagesTr = os.path.join(OUTPUT_BASE, "imagesTr")
labelsTr = os.path.join(OUTPUT_BASE, "labelsTr")
imagesTs = os.path.join(OUTPUT_BASE, "imagesTs")
os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)
os.makedirs(imagesTs, exist_ok=True)
print(f"✅ Created dataset folders under {OUTPUT_BASE}")

# =====================================
# 📦 複製檔案
# =====================================
print("📦 Copying training images...")
for f in sorted(os.listdir(TRAIN_IMAGES)):
    if f.endswith(".nii.gz"):
        shutil.copy(os.path.join(TRAIN_IMAGES, f), imagesTr)

print("📦 Copying training labels...")
for f in sorted(os.listdir(TRAIN_LABELS)):
    if f.endswith(".nii.gz"):
        shutil.copy(os.path.join(TRAIN_LABELS, f), labelsTr)

print("📦 Copying test images...")
for f in sorted(os.listdir(TEST_IMAGES)):
    if f.endswith(".nii.gz"):
        shutil.copy(os.path.join(TEST_IMAGES, f), imagesTs)

# =====================================
# ✏️ 檔名修正：符合 nnUNet 命名規範
# =====================================
print("✏️ Renaming files to comply with nnUNet format...")
# 影像
for f in os.listdir(imagesTr):
    if f.endswith(".nii.gz") and f.endswith("__0001.nii.gz"):
        old = os.path.join(imagesTr, f)
        new = os.path.join(imagesTr, f.replace("__0001.nii.gz", "_0001.nii.gz"))
        os.rename(old, new)
    if  f.endswith("_0000.nii.gz") or f.endswith("_0001.nii.gz"):
        pass
    else:
        os.remove(os.path.join(imagesTr, f))
    
# 標籤
for f in os.listdir(labelsTr):
    if f.endswith("_gt.nii.gz"):
        old = os.path.join(labelsTr, f)
        new = os.path.join(labelsTr, f.replace("_gt.nii.gz", ".nii.gz"))
        os.rename(old, new)
print("✅ Renaming done.")

# =====================================
# 🧾 建立 dataset.json (符合新版 nnU-Net v2 規範)
# =====================================
num_train = len([f for f in os.listdir(imagesTr) if f.endswith("_0000.nii.gz")])

dataset_json = {
    "channel_names": {
        "0": "CT",
        "1": "HE"
    },
    "labels": {
        "background": 0,
        "myocardium": 1,
        "left_atrium": 2,
        "left_ventricle": 3
    },
    "numTraining": num_train,
    "file_ending": ".nii.gz",
    "name": "AICUP_Cardiac",
    # "numChannels": 2,
}

json_path = os.path.join(OUTPUT_BASE, "dataset.json")
with open(json_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"✅ dataset.json created at {json_path} (numTraining={num_train})")

# =====================================
# 🔧 前處理 (Plan & Preprocess)
# =====================================
print("🔧 Running nnUNetv2_plan_and_preprocess ...")

run_command(f"nnUNetv2_plan_and_preprocess -d {TASK_ID} --no_pp")
print("✅ Preprocessing completed.")

# =====================================
# 🧠 微調 (Fine-tuning using pretrained encoder)
# =====================================
print("🚀 Starting fine-tuning ...")

run_command(f"nnUNetv2_preprocess_like_nnssl -d {TASK_ID} -n TTT -pc {PRETRAINED_CHECKPOINT} -am like_pretrained")


run_command(f"nnUNetv2_train_pretrained {TASK_ID} 3d_fullres all -p ptPlans__TTT____Spacing__1.00_1.00_1.00___Norm__Z_Z")
print("✅ Fine-tuning finished.")

# =====================================
# 🧪 推論 (Inference)
# =====================================
import os
import shutil

# 設定你的測試資料夾路徑
folder_path = "./nnUNet_raw/Dataset003_AICUP_Cardiac_FT/imagesTs"
# 設定正確的副檔名 (根據 dataset.json，通常是 .nii.gz)
extension = ".nii.gz"

files = os.listdir(folder_path)
print(f"Found {len(files)} files.")

# for f in files:
#     if f.endswith(extension) and not f.endswith(f"_0000{extension}"):
#         # 建構舊路徑與新路徑
#         old_path = os.path.join(folder_path, f)
        
#         # 插入 _0000
#         new_name = f.replace(extension, f"_0000{extension}")
#         new_path = os.path.join(folder_path, new_name)
        
#         print(f"Renaming: {f} -> {new_name}")
#         os.rename(old_path, new_path)

# print("Done!")

print("🧪 Running inference on test images …")
inference_output_dir = os.path.join("./inference_results", DATASET_NAME)
os.makedirs(inference_output_dir, exist_ok=True)

run_command("nnUNetv2_predict \
  -i ./nnUNet_raw/Dataset003_AICUP_Cardiac_FT/imagesTs \
  -o ./inference_results/Dataset003_AICUP_Cardiac_FT \
  -d 2 \
  -tr PretrainedTrainer \
  -p ptPlans \
  -c TTT____Spacing__1.00_1.00_1.00___Norm__Z__3d_fullres \
  -f all ")

print(f"✅ Inference done. Outputs in {inference_output_dir}")

