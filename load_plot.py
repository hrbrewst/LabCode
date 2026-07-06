import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

csv_folder = "/Users/haleybrewster/Desktop/weight"
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

print("Folder exists?", os.path.exists(csv_folder))
print("CSV files found:", csv_files)

results = []

for file in csv_files:
    df = pd.read_csv(file)

    x_position = df["X [m]"]

    # Final displacement from starting position
    displacement_x = x_position.iloc[-1] - x_position.iloc[0]

    # If forward motion is negative and you want positive displacement, use:
    # displacement_x = -(x_position.iloc[-1] - x_position.iloc[0])

    filename = os.path.basename(file)

    # Extract load value from filename
    # Example filenames: load_1.csv, robot_2.5kg.csv, weight_0.75.csv
    numbers = re.findall(r"\d+\.?\d*", filename)

    if len(numbers) > 0:
        load = float(numbers[0])
    else:
        print("Could not find load value in filename:", filename)
        continue

    results.append((load, displacement_x, filename))

# Sort by load
results.sort(key=lambda x: x[0])

loads = [r[0] for r in results]
displacements = [r[1] for r in results]
labels = [r[2] for r in results]

print("\nLoad and displacement results:")
for load, displacement, filename in results:
    print(f"{filename}: load = {load}, displacement = {displacement:.4f} m")

plt.figure(figsize=(8, 6))
plt.plot(loads, displacements, marker="o")

plt.xlabel("Load")
plt.ylabel("Displacement in X-direction [m]")
plt.title("Displacement vs Load")
plt.grid(True)
plt.tight_layout()
plt.show()
