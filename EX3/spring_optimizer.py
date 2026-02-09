import numpy as np
import matplotlib.pyplot as plt

# ==================================================================
# 1. LOAD DATA (PRESERVE ORIGINAL FORMAT)
# ==================================================================
file_path = r'C:\Users\isrgd\robots\Mobile_Robots\EX3\TRACK_LOG.TXT'

try:
    # Read the file as a raw matrix
    data = np.genfromtxt(file_path, delimiter=',')
    
    # We assume the last column (index 4) contains the markers
    # Format: [Col0, Col1, Col2, Col3, Marker]
    types = data[:, 4]
    
    is_anchor = (types == 1)
    print(f"Loaded {len(data)} rows. Preserving all columns.")

except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# ==================================================================
# 2. FIND TRIM BOUNDARIES
# ==================================================================
anchor_indices = np.where(is_anchor)[0]

if len(anchor_indices) < 2:
    print("Warning: Less than 2 anchors found. Saving full file.")
    start_idx = 0
    end_idx = len(data) - 1
else:
    # Identify the very first and very last anchor index
    start_idx = anchor_indices[0]
    end_idx = anchor_indices[-1]
    print(f"Trimming Data: Keeping rows {start_idx} to {end_idx}")

# ==================================================================
# 3. TRIM & SAVE
# ==================================================================
# Slice the entire matrix (all columns) based on the indices
trimmed_data = data[start_idx:end_idx+1]

# Save exactly what was read, just fewer rows
np.savetxt('SLAM_CLEANED.TXT', trimmed_data, delimiter=',', fmt='%.2f')

print(f"Saved {len(trimmed_data)} rows to SLAM_CLEANED.TXT")

# ==================================================================
# 4. VISUALIZATION (Sanity Check)
# ==================================================================
# We assume Col 0 and Col 1 are Robot X,Y for plotting purposes
rx = data[:, 0]
ry = data[:, 1]
final_rx = trimmed_data[:, 0]
final_ry = trimmed_data[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Original
ax1.plot(rx, ry, color='gray', alpha=0.5, label='Original')
ax1.scatter(rx[start_idx], ry[start_idx], c='green', s=50, label='Start Cut')
ax1.scatter(rx[end_idx], ry[end_idx], c='red', s=50, label='End Cut')
ax1.set_title(f"Original ({len(data)} points)")
ax1.legend()

# Plot 2: Trimmed
ax2.plot(final_rx, final_ry, color='blue', linewidth=2, label='Trimmed')
# If there are columns 2 and 3 (Walls), plot them just in case
if data.shape[1] >= 4:
    ax2.scatter(trimmed_data[:, 2], trimmed_data[:, 3], color='black', s=1, alpha=0.3, label='Col 2/3 Data')

ax2.set_title(f"Trimmed Result ({len(trimmed_data)} points)")
ax2.legend()

for ax in [ax1, ax2]:
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()