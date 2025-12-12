import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline():
    # Setup the figure for IEEE Column width (approx 3.5 inches) or Page width (7 inches)
    # Using a wider format for clarity
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Style constants
    box_color = '#E6E6E6'       # Light Gray for data
    module_color = '#DAE8FC'    # Light Blue for modules
    edge_color = '#6C8EBF'      # Darker blue border
    arrow_props = dict(facecolor='black', width=0.05)
    font_size = 10
    title_size = 11
    
    # === STAGE 1: S3D Pre-training (Top Row) ===
    # Frame
    ax.add_patch(patches.Rectangle((0.2, 3.2), 11.6, 2.6, fill=False, linestyle='--', color='gray'))
    ax.text(0.4, 5.5, "Stage 1: S3D-Style Self-Supervised Pre-Training", fontsize=title_size, weight='bold')

    # Blocks
    # Unlabeled CT
    ax.add_patch(patches.Rectangle((0.5, 3.8), 1.5, 1.0, facecolor=box_color, edgecolor='black'))
    ax.text(1.25, 4.3, "Unlabeled\nCT Volume", ha='center', va='center', fontsize=font_size)

    # Masking
    ax.arrow(2.0, 4.3, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((2.5, 3.8), 1.2, 1.0, facecolor=module_color, edgecolor=edge_color))
    ax.text(3.1, 4.3, "Masking\n(75%)", ha='center', va='center', fontsize=font_size)

    # Sparse Encoder
    ax.arrow(3.7, 4.3, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((4.2, 3.8), 1.8, 1.0, facecolor='#D5E8D4', edgecolor='#82B366')) # Greenish for emphasis
    ax.text(5.1, 4.3, "Sparse Encoder\n(nnU-Net Backbone)\n[SparseConv + Norm]", ha='center', va='center', fontsize=font_size)

    # Densification
    ax.arrow(6.0, 4.3, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((6.5, 3.8), 1.5, 1.0, facecolor=module_color, edgecolor=edge_color))
    ax.text(7.25, 4.3, "Mask Token\nDensification", ha='center', va='center', fontsize=font_size)

    # Decoder
    ax.arrow(8.0, 4.3, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((8.5, 3.8), 1.2, 1.0, facecolor=module_color, edgecolor=edge_color))
    ax.text(9.1, 4.3, "Decoder", ha='center', va='center', fontsize=font_size)

    # Reconstruction
    ax.arrow(9.7, 4.3, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((10.2, 3.8), 1.4, 1.0, facecolor=box_color, edgecolor='black'))
    ax.text(10.9, 4.3, "Reconstructed\nVolume", ha='center', va='center', fontsize=font_size)

    # === WEIGHT TRANSFER ARROW ===
    # Draw a curved arrow from Stage 1 Encoder to Stage 2 Encoder
    ax.annotate("Weight Transfer",
                xy=(5.1, 2.2), xycoords='data',
                xytext=(5.1, 3.8), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="#B85450", lw=2, linestyle="dashed"),
                ha='center', va='bottom', color="#B85450", weight='bold')

    # === STAGE 2: Fine-Tuning (Bottom Left) ===
    # Frame
    ax.add_patch(patches.Rectangle((0.2, 0.2), 6.5, 2.8, fill=False, linestyle='--', color='gray'))
    ax.text(0.4, 2.7, "Stage 2: nnU-Net Fine-Tuning", fontsize=title_size, weight='bold')

    # Labeled CT
    ax.add_patch(patches.Rectangle((0.5, 1.2), 1.2, 1.0, facecolor=box_color, edgecolor='black'))
    ax.text(1.1, 1.7, "Labeled\nAI CUP Data", ha='center', va='center', fontsize=font_size)

    # Encoder (Fine-tune)
    ax.arrow(1.7, 1.7, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((2.2, 1.2), 1.5, 1.0, facecolor='#D5E8D4', edgecolor='#82B366'))
    ax.text(2.95, 1.7, "Encoder\n(Fine-Tuning)", ha='center', va='center', fontsize=font_size)

    # Decoder / Seg Head
    ax.arrow(3.7, 1.7, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((4.2, 1.2), 1.5, 1.0, facecolor=module_color, edgecolor=edge_color))
    ax.text(4.95, 1.7, "Decoder &\nSeg Head", ha='center', va='center', fontsize=font_size)
    
    # Loss info
    ax.text(2.95, 0.8, "(Dice + CE Loss)", ha='center', va='top', fontsize=9, fontstyle='italic', color='gray')

    # Raw Output
    ax.arrow(5.7, 1.7, 0.5, 0, **arrow_props)
    
    # === STAGE 3: Post-Processing (Bottom Right) ===
    # Frame
    ax.add_patch(patches.Rectangle((6.9, 0.2), 4.9, 2.8, fill=False, linestyle='--', color='gray'))
    ax.text(7.1, 2.7, "Stage 3: Post-Processing", fontsize=title_size, weight='bold')

    # Raw Mask Input
    ax.add_patch(patches.Rectangle((7.2, 1.2), 1.2, 1.0, facecolor=box_color, edgecolor='black'))
    ax.text(7.8, 1.7, "Raw\nPredictions", ha='center', va='center', fontsize=font_size)

    # Post Process Block
    ax.arrow(8.4, 1.7, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((8.9, 1.2), 1.6, 1.0, facecolor='#FFE6CC', edgecolor='#D79B00'))
    ax.text(9.7, 1.7, "3D Morph Filter\n(Keep Largest)", ha='center', va='center', fontsize=font_size)

    # Final Output
    ax.arrow(10.5, 1.7, 0.5, 0, **arrow_props)
    ax.add_patch(patches.Rectangle((11.0, 1.2), 0.7, 1.0, facecolor='#F8CECC', edgecolor='#B85450'))
    ax.text(11.35, 1.7, "Final\nMask", ha='center', va='center', fontsize=font_size, rotation=90)

    # Save
    plt.tight_layout()
    plt.savefig('visualization_output/pipeline_overview.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('visualization_output/pipeline_overview.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_pipeline()