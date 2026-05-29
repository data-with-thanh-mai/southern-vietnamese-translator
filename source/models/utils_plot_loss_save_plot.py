# Gồm hàm lưu biểu đồ, vẽ biểu đồ train, val loss
import matplotlib.pyplot as plt
def save_plot(filename, folder_path="outputs/figures", dpi=300):
    """
    Utility function to save matplotlib plots safely and crisply.
    
    Args:
    - filename (str): Name of the image file (e.g., 'Error_Chart.png')
    - folder_path (str): Directory to save the image (Default: 'outputs/figures')
    - dpi (int): Image resolution (Default: 300 - standard for printing)
    """
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"📸 Successfully saved plot to: {full_path}")
    return full_path


def plot_training_loss(train_losses, val_losses, filename='Loss_Curve.png'):
    """
    Vẽ biểu đồ so sánh Train Loss và Val Loss qua các Epoch.
    Rất quan trọng để chứng minh mô hình có bị Overfitting hay không.
    """
    plt.figure(figsize=(10, 6))
    
    # Vẽ 2 đường đồ thị
    plt.plot(train_losses, label='Train Loss', color='blue', marker='o', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', marker='s', linewidth=2)
    
    plt.title('BIỂU ĐỒ HỘI TỤ (TRAINING & VALIDATION LOSS)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Mức độ sai lệch (Loss)', fontsize=12, fontweight='bold')
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Gọi lại hàm save_plot đã viết ở trên để lưu ảnh
    save_plot(filename)
