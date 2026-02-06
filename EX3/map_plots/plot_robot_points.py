import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\SLAM_CLEANED.TXT"
# FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\map_plots\\SLAM_PATH.TXT"
# FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\RAW_LAP_ONLY.TXT"
FILE_NAME = "C:\\CS\\Robots_mobile\\Mobile_Robots\\EX3\\SLAM_CLEANED.TXT"

def main():
    print(f"Loading data from: {FILE_NAME}")
    data = np.loadtxt(FILE_NAME, delimiter=',')
    
    x = data[:, 0]
    y = data[:, 1]
    # t = data[:, 2]  # ignored

    # // x is the third col and y is the forth col
    # x = data[:, 2]
    # y = data[:, 3]
    
    print(f"Loaded {len(x)} points")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=10, c='blue', alpha=0.7)
    plt.title("Raw Robot Points", fontsize=14)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
