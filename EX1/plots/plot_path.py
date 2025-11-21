import matplotlib.pyplot as plt

# Load the robot path file: each line is "xmm,ymm,thetam"
xs = []
ys = []
thetas = []

with open("ROUTE.TXT") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Remove null bytes if they appear due to NXT file corruption
        line = line.replace('\x00', '')
            
        parts = line.split(",")
        if len(parts) != 3:
            continue
        
        xmm = int(parts[0])
        ymm = int(parts[1])
        thetam = int(parts[2])
        
        # Convert back to real units
        x = xmm / 10000.0      # cm
        y = ymm / 10000.0      # cm
        theta = thetam / 100000.0  # radians
        
        xs.append(x)
        ys.append(y)
        thetas.append(theta)

# Plot path
plt.figure(figsize=(7,5))
plt.plot(xs, ys, marker='o', markersize=2, linewidth=1)

# Mark Start
plt.plot(xs[0], ys[0], 'go', label='Start')
# Mark End
plt.plot(xs[-1], ys[-1], 'ro', label='End')

plt.title("Robot Odometry Path")
plt.xlabel("X position (cm)")
plt.ylabel("Y position (cm)")
plt.grid(True)
plt.axis("equal")  # keep aspect ratio correct

plt.show()
