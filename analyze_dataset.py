import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def analyze_dataset():
    train_images_dir = './train/imagesTr'
    train_labels_dir = './train/labelsTr'
    output_dir = './visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'dataset_analysis_report.txt')

    image_files = sorted(glob.glob(os.path.join(train_images_dir, '*.nii.gz')))
    
    # Storage for metrics
    spacings = []
    shapes = []
    intensity_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0} # Accumulate total voxels per class
    
    print(f"Starting analysis of {len(image_files)} files...")
    
    with open(report_path, 'w') as report:
        report.write("Dataset Analysis Report\n")
        report.write("=======================\n\n")
        
        for idx, img_path in enumerate(image_files):
            basename = os.path.basename(img_path)
            # Infer label path: patientXXXX.nii.gz -> patientXXXX_gt.nii.gz
            label_name = basename.replace('.nii.gz', '_gt.nii.gz')
            label_path = os.path.join(train_labels_dir, label_name)
            
            try:
                # Load Image
                img = nib.load(img_path)
                data = img.get_fdata()
                header = img.header
                spacing = header.get_zooms()
                
                # Metrics
                spacings.append(spacing)
                shapes.append(data.shape)
                intensity_stats['min'].append(data.min())
                intensity_stats['max'].append(data.max())
                intensity_stats['mean'].append(data.mean())
                intensity_stats['std'].append(data.std())
                
                # Load Label
                if os.path.exists(label_path):
                    lbl = nib.load(label_path)
                    lbl_data = lbl.get_fdata()
                    unique, counts = np.unique(lbl_data, return_counts=True)
                    for u, c in zip(unique, counts):
                        if int(u) in label_counts:
                            label_counts[int(u)] += c
                else:
                    print(f"Warning: Label not found for {basename}")

                # Progress log
                msg = f"[{idx+1}/{len(image_files)}] {basename}: Shape={data.shape}, Spacing={spacing}"
                print(msg)
                report.write(msg + "\n")
                
            except Exception as e:
                err_msg = f"Error processing {basename}: {e}"
                print(err_msg)
                report.write(err_msg + "\n")

    # --- Aggregation and Plotting ---
    spacings = np.array(spacings)
    shapes = np.array(shapes)
    
    # 1. Spacing Distribution
    plt.figure(figsize=(10, 6))
    plt.boxplot([spacings[:, 0], spacings[:, 1], spacings[:, 2]], labels=['X Spacing', 'Y Spacing', 'Z Spacing'])
    plt.title('Voxel Spacing Distribution')
    plt.ylabel('mm')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'dataset_stats_spacing.png'))
    plt.close()
    
    # 2. Intensity Stats
    mins = np.array(intensity_stats['min'])
    maxs = np.array(intensity_stats['max'])
    means = np.array(intensity_stats['mean'])
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.boxplot(mins)
    plt.title('Min Intensity (HU)')
    
    plt.subplot(1, 3, 2)
    plt.boxplot(means)
    plt.title('Mean Intensity (HU)')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(maxs)
    plt.title('Max Intensity (HU)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_stats_intensity.png'))
    plt.close()
    
    # 3. Label Balance
    classes = list(label_counts.keys())
    counts = list(label_counts.values())
    total_voxels = sum(counts)
    percentages = [c / total_voxels * 100 for c in counts]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['gray', 'red', 'green', 'blue'])
    plt.title('Total Class Volume Distribution')
    plt.xlabel('Class ID')
    plt.ylabel('Total Voxels')
    plt.xticks(classes, ['0 (Bg)', '1 (Muscle)', '2 (Valve)', '3 (Calc)'])
    plt.yscale('log') # Log scale because background dominates
    
    # Add text labels
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{pct:.1f}%',
                 ha='center', va='bottom')
                 
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'dataset_stats_labels.png'))
    plt.close()

    # Final Report append
    with open(report_path, 'a') as report:
        report.write("\n\nAggregate Statistics\n")
        report.write("--------------------\n")
        report.write(f"Mean Spacing: {np.mean(spacings, axis=0)}\n")
        report.write(f"Mean Shape: {np.mean(shapes, axis=0)}\n")
        report.write(f"Global Intensity Range: {np.min(mins)} to {np.max(maxs)}\n")
        report.write("Class Distribution:\n")
        for cls, count in label_counts.items():
            report.write(f"  Class {cls}: {count} voxels ({count/total_voxels*100:.2f}%)\n")
            
    print(f"\nAnalysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    analyze_dataset()
