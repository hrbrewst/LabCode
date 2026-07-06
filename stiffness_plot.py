import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

csv_folder = "/Users/haleybrewster/Desktop/stiffness"
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

print("Folder exists?", os.path.exists(csv_folder))
print("CSV files found:", csv_files)

results = []

for file in csv_files:
    df = pd.read_csv(file)

    x_position = df["X [m]"]

    # Final displacement from starting position
    displacement_x = x_position.iloc[-1] - x_position.iloc[0]


    filename = os.path.basename(file)

    # Extract load value from filename
    numbers = re.findall(r"\d+\.?\d*", filename)

    if len(numbers) > 0:
        stiffness = float(numbers[0])
    else:
        print("Could not find load value in filename:", filename)
        continue

    results.append((stiffness, displacement_x, filename))

# Sort by stiffness
results.sort(key=lambda x: x[0])

stiffness = [r[0] for r in results]
displacements = [r[1] for r in results]
labels = [r[2] for r in results]

print("\nStiffness and displacement results:")
for load, displacement, filename in results:
    print(f"{filename}: stiffness = {stiffness}, displacement = {displacement:.4f} m")
    
plt.figure(figsize=(8, 6))
plt.plot(stiffness, displacements, marker="o")
    
plt.xlabel("Stiffness")
plt.ylabel("Displacement in X-direction [m]")
plt.title("Displacement vs Stiffness")
plt.grid(True)
plt.tight_layout()
plt.show()
