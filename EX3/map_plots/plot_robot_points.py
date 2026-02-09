import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\TRACK_LOG.TXT"

def main():
    print(f"Loading data from: {FILE_NAME}")
    data = np.loadtxt(FILE_NAME, delimiter=',')
    
    x = data[:, 0]
    y = data[:, 1]
    wall_type = data[:, 4].astype(int)  # 0=white wall, 1=black line, 2=corner
    
    # Separate points by wall type
    white_wall = wall_type == 0
    black_line = wall_type == 1
    corner = wall_type == 2
    
    print(f"Loaded {len(x)} points")
    print(f"  White wall points: {np.sum(white_wall)}")
    print(f"  Black line points: {np.sum(black_line)}")
    print(f"  Corner points: {np.sum(corner)}")
    
    plt.figure(figsize=(10, 8))
    
    # Plot white wall points (type 0) in blue
    plt.scatter(x[white_wall], y[white_wall], s=10, c='blue', alpha=0.7, label='White wall (0)')
    
    # Plot black line points (type 1) in red
    plt.scatter(x[black_line], y[black_line], s=15, c='red', alpha=0.8, label='Black line (1)')
    
    # Plot corner points (type 2) in green with larger markers
    plt.scatter(x[corner], y[corner], s=50, c='green', marker='*', alpha=1.0, label='Corner (2)')
    
    plt.title("Raw Robot Points", fontsize=14)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
