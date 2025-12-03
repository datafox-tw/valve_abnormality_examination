import numpy as np
import nibabel as nib
import os
from glob import glob
from typing import List

# =================================================================
# 🛑 必需要修改的路徑設定 🛑
# 請將此路徑替換為您的 nnUNet 數據集根目錄
# 範例: nnUNet_raw/DatasetXXX_YourTask/
BASE_DIR = "nnUNet_raw/Dataset002_AICUP_Cardiac_FT"
# =================================================================

def global_histogram_equalization_3d(volume: np.ndarray) -> np.ndarray:
    """
    對整個 3D 體積執行全局直方圖均衡化 (Global HE)。

    Args:
        volume (np.ndarray): 3D 圖像的 NumPy 陣列。
        
    Returns:
        np.ndarray: 均衡化後且數據類型與輸入相同的 3D 圖像陣列。
    """
    
    # 記錄原始數據類型，以便輸出時保持一致
    original_dtype = volume.dtype
    
    # 為了計算整數 bin，需要確保數據是整數類型
    # 這裡假設 CT 數據通常是 int16
    if not np.issubdtype(volume.dtype, np.integer):
        # 如果是 float，暫時轉換為 int16 進行 bin 計算
        volume_int = volume.astype(np.int16)
    else:
        volume_int = volume
        
    flat_volume = volume_int.flatten()
    
    # 1. 計算直方圖的範圍和 bin 數量
    # 醫學圖像通常有負值，需要計算真實的 min/max
    min_val = np.min(flat_volume)
    max_val = np.max(flat_volume)
    bins = int(max_val - min_val) + 1
    
    # 計算直方圖 (Histogram)
    hist, _ = np.histogram(flat_volume, bins=bins, range=(min_val, max_val))
    
    # 2. 計算累積分佈函數 (CDF)
    cdf = hist.cumsum()
    
    # 3. 處理 CDF 中的 0 值 (避免除以零)，並構建映射函數 (LUT)
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_m.min()
    total_pixels = volume_int.size
    
    # 標準的 HE 映射公式，將 cdf 映射到 0 到 bins-1 範圍
    lut_normalized = (cdf_m - cdf_min) / (total_pixels - cdf_min)
    lut_scaled = np.round(lut_normalized * (bins - 1))
    lut = np.ma.filled(lut_scaled, 0)
    
    # 4. 應用映射函數
    # 原始像素值轉為 bin 索引: index = value - min_val
    bin_indices = (volume_int - min_val).astype(int)
    
    # 使用 LUT 進行映射
    equalized_volume = lut[bin_indices]
    
    # 5. 將結果轉換回原始的數據類型
    equalized_volume = equalized_volume.astype(original_dtype)
    
    return equalized_volume


def process_directory_for_multichannel(folder_name: str):
    """
    處理指定文件夾中的 *_0000 檔案，執行 HE，並將結果以 *_0001 儲存回原文件夾。
    
    Args:
        folder_name (str): 待處理的輸入文件夾名 ('imagesTr' 或 'imagesTs')
    """
    input_dir = os.path.join(BASE_DIR, folder_name)
    
    print(f"\n======== 正在處理 {folder_name}，準備建立 _0001 HE 通道 ========")
    
    # 查找所有原始 CT 檔案 (通道 0000)
    original_ct_files: List[str] = glob(os.path.join(input_dir, "*_0000.nii.gz"))
    
    if not original_ct_files:
        print(f"🔴 警告: 在路徑 {input_dir} 中未找到任何 *\_0000.nii.gz 檔案。請檢查 BASE_DIR 設定是否正確。")
        return
    
    print(f"🟢 找到 {len(original_ct_files)} 個 _0000 檔案...")
    
    for file_path in original_ct_files:
        try:
            filename = os.path.basename(file_path)
            
            # 將檔名尾碼 _0000 替換為 _0001
            new_filename = filename.replace("_0000.nii.gz", "_0001.nii.gz")
            output_path = os.path.join(input_dir, new_filename)
            
            # 檢查 _0001 檔案是否已經存在，避免重複處理
            if os.path.exists(output_path):
                 print(f"  -> {new_filename} 已存在，跳過。")
                 continue
            
            print(f"  -> 處理檔案: {filename} -> 儲存為 {new_filename}")
            
            # 1. 讀取 CT 影像
            nifti_img = nib.load(file_path)
            volume_data = nifti_img.get_fdata()
            affine_matrix = nifti_img.affine
            
            # 2. 執行 3D 全局直方圖均衡化
            # 使用 float32 確保 HE 運算的數值穩定性
            equalized_volume = global_histogram_equalization_3d(volume_data.astype(np.float32))
            
            # 3. 儲存結果為 _0001 通道
            equalized_nifti = nib.Nifti1Image(equalized_volume, affine_matrix)
            nib.save(equalized_nifti, output_path)

        except Exception as e:
            print(f"❌ 處理檔案 {filename} 時發生錯誤: {e}")

# =================================================================
# 主執行區塊
# =================================================================
if __name__ == "__main__":
    
    if not os.path.exists(BASE_DIR):
        print(f"❌ 錯誤: 基本路徑 {BASE_DIR} 不存在。請檢查並修改 BASE_DIR 變數。")
    else:
        # 1. 處理 imagesTr (訓練數據)
        process_directory_for_multichannel("imagesTr")
        
        # 2. 處理 imagesTs (測試/驗證數據)
        process_directory_for_multichannel("imagesTs")
        
        print("\n🎉 所有檔案處理完成。")
        print("下一步請記得更新 dataset.json 以納入 '0001' 通道，並重新運行 nnU-Net 預處理。")