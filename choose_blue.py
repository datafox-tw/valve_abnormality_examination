import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_nifti_file(filepath):
    """Loads a NIfTI file and returns data and affine."""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header

def plot_intensity_histogram(data, output_path):
    """Plots and saves the intensity distribution histogram."""
    plt.figure(figsize=(10, 6))
    
    # Flatten data and remove background (0) if needed, 
    # but for CT, 0 is a valid value. 
    # Usually we might want to exclude air which is very low (~ -1000 HU) 
    # if we only care about tissue, but the request asks for general distribution.
    # Let's plot the whole range inside the body usually.
    
    # Simple downsample for speed if needed, but modern matplotlib handles millions ok.
    flat_data = data.flatten()
    
    plt.hist(flat_data, bins=100, color='skyblue', alpha=0.7, log=True)
    plt.title('Voxel Intensity Distribution (HU)')
    plt.xlabel('Intensity (Hounsfield Units)')
    plt.ylabel('Count (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved intensity histogram to {output_path}")

def plot_orthographic_slices(data, output_path):
    """Plots middle slices for Axial, Coronal, and Sagittal planes."""
    shape = data.shape
    # Find center indices
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    
    slice_0 = data[cx, :, :] # Sagittal approx
    slice_1 = data[:, cy, :] # Coronal approx
    slice_2 = data[:, :, cz] # Axial approx
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show slices with gray colormap
    # Rotate if necessary for better viewing, but raw storage is often isotropic or close
    axes[0].imshow(np.rot90(slice_0), cmap='gray')
    axes[0].set_title(f'Sagittal (x={cx})')
    axes[0].axis('off')
    
    axes[1].imshow(np.rot90(slice_1), cmap='gray')
    axes[1].set_title(f'Coronal (y={cy})')
    axes[1].axis('off')
    
    axes[2].imshow(np.rot90(slice_2), cmap='gray')
    axes[2].set_title(f'Axial (z={cz})')
    axes[2].axis('off')
    
    plt.suptitle('Orthographic Slices (Mid-volume)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved orthographic slices to {output_path}")

def plot_orthographic_slices_with_labels(data, label_data, output_path):
    """Plots middle slices with label overlay."""
    shape = data.shape
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    
    slices = [
        (data[cx, :, :], label_data[cx, :, :], f'Sagittal (x={cx})'),
        (data[:, cy, :], label_data[:, cy, :], f'Coronal (y={cy})'),
        (data[:, :, cz], label_data[:, :, cz], f'Axial (z={cz})')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define colors for labels: 0=Background (Transparent), 1=Red, 2=Green, 3=Blue
    # Using a listed colormap for labels is easier, or just masking.
    from matplotlib.colors import ListedColormap
    # Create a colormap: 0:clear, 1:red, 2:green, 3:blue
    cmap_labels = ListedColormap(['none', 'red', 'green', 'blue'])
    
    for i, (sl_img, sl_lbl, title) in enumerate(slices):
        # Background image
        axes[i].imshow(np.rot90(sl_img), cmap='gray')
        
        # Overlay labels
        # Mask out 0 (background)
        lbl_show = np.rot90(sl_lbl)
        masked_lbl = np.ma.masked_where(lbl_show == 0, lbl_show)
        
        # We need to map 1, 2, 3 to colors. 
        # imshow with cmap handles values directly if we set limits.
        axes[i].imshow(masked_lbl, cmap=cmap_labels, vmin=0, vmax=3, alpha=0.5, interpolation='nearest')
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Label 1 (Muscle)'),
        Patch(facecolor='green', edgecolor='green', label='Label 2 (Valve)'),
        Patch(facecolor='blue', edgecolor='blue', label='Label 3 (Calcification)')
    ]
    fig.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('Orthographic Slices with Labels', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved labeled slices to {output_path}")

def plot_3d_surface(label_data, output_path):
    """Generates 3D surface rendering of labels."""
    print("Generating 3D surface..." )
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['r', 'g', 'b']
    labels = [1, 2, 3]
    names = ['Label 1', 'Label 2', 'Label 3']
    
    has_plot = False
    
    for lbl, color, name in zip(labels, colors, names):
        # Create a binary mask for the current label
        mask = (label_data == lbl)
        print(name, np.sum(mask))
        # Marching cubes requires some volume. Check if label exists.
        if np.sum(mask) < 50:
            print(f"Skipping {name}: not enough voxels.")
            continue
        continue
            
        try:
            # Use spacing to correct aspect ratio if possible, but for simple vis just index space is okay usually.
            # But let's try to grab spacing if we pass it, or just assume isotropic for now or fix axis later.
            # marching_cubes returns verts, faces, normals, values
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            
            mesh = Poly3DCollection(verts[faces], alpha=0.3)
            mesh.set_facecolor(color)
            ax.add_collection3d(mesh)
            has_plot = True
            
        except Exception as e:
            print(f"Could not extract surface for {name}: {e}")
    
    if has_plot:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Set limits
        shape = label_data.shape
        ax.set_xlim(0, shape[0])
        ax.set_ylim(0, shape[1])
        ax.set_zlim(0, shape[2])
        
        plt.title("3D Surface Rendering of Labels")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved 3D surface plot to {output_path}")
    else:
        print("No surfaces generated to plot.")
    plt.close()


def main():
    # Configuration
    train_images_dir = './train/imagesTr'
    output_dir = './visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find files
    files = sorted(glob.glob(os.path.join(train_images_dir, '*.nii.gz')))
    if not files:
        print(f"No .nii.gz files found in {train_images_dir}")
        return

    print(f"Found {len(files)} training images.")
    
    # Process the first file as a representative sample
    for id_v in range(0,50):
        sample_file = files[id_v]
        print(f"\nProcessing sample file: {sample_file}")
        
        try:
            data, affine, header = load_nifti_file(sample_file)
            
            # 1. Dataset Characteristics Report
            spacing = header.get_zooms()
            print("\n--- Dataset Characteristics (Sample) ---")
            print(f"Image Shape: {data.shape}")
            print(f"Voxel Spacing (mm): {spacing}")
            print(f"Intensity Range: Min={data.min():.2f}, Max={data.max():.2f}")
            print(f"Mean Intensity: {data.mean():.2f}")
            print(f"Std Dev: {data.std():.2f}")
            
            # 2. Intensity Distribution
            hist_path = os.path.join(output_dir, f'intensity_histogram-{id_v}.png')
            plot_intensity_histogram(data, hist_path)
            
            # 3. 3D Geometry Visualization (Orthographic Slices)
            slices_path = os.path.join(output_dir, f'orthographic_slices-{id_v}.png')
            plot_orthographic_slices(data, slices_path)

            # 4. Label Processing
            # Assuming label filename structure: patientXXXX.nii.gz -> patientXXXX_gt.nii.gz in labelsTr
            basename = os.path.basename(sample_file) # patient0001.nii.gz
            label_filename = basename.replace('.nii.gz', '_gt.nii.gz')
            label_path = os.path.join('./train/labelsTr', label_filename)
            
            if os.path.exists(label_path):
                print(f"\nProcessing label file: {label_path}")
                label_data, _, _ = load_nifti_file(label_path)
                
                # Label Statistics
                unique_labels = np.unique(label_data)
                print(f"Unique Labels Found: {unique_labels}")
                
                # Overlay Plot
                lbl_slices_path = os.path.join(output_dir, f'orthographic_slices_with_labels-{id_v}.png')
                plot_orthographic_slices_with_labels(data, label_data, lbl_slices_path)
                
                # 3D Surface Plot
                surface_path = os.path.join(output_dir, f'3d_surface_rendering-{id_v}.png')
                plot_3d_surface(label_data, surface_path)
                
            else:
                print(f"Label file not found at {label_path}")
            
            print("\nVisualization completed successfully.")
        
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
