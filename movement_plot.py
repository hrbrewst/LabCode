import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

csv_folder = "/Users/haleybrewster/Desktop/stiffness"
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

print("Folder path:", csv_folder)
print("Folder exists?", os.path.exists(csv_folder))

dt = 0.01  # seconds per time step

print("CSV files found:")
print(csv_files)

plt.figure(figsize=(10, 6))

# Y position of the robot/baseplate
#    y_position = df["Y [m]"]
    # Distance moved from starting position
 #   movement_y = y_position - y_position.iloc[0]
# Z position of the robot/baseplate
  #  z_position = df["Z [m]"]
    # Distance moved from starting position
   # movement_z = z_position - z_position.iloc[0]

legend_items = []

plt.figure(figsize=(10, 6))

for file in csv_files:
    df = pd.read_csv(file)

    time = df.index * dt

    x_position = df["X [m]"]
    movement_x = x_position - x_position.iloc[0]

    label = os.path.basename(file).replace(".csv", "")

    line, = plt.plot(time, movement_x, label=label)

    final_value = movement_x.iloc[-1]
    legend_items.append((final_value, line, label))

# Sort legend by final movement value, highest at top
legend_items.sort(key=lambda item: item[0], reverse=True)

sorted_lines = [item[1] for item in legend_items]
sorted_labels = [f"{item[2]}: {item[0]:.3f} m" for item in legend_items]

plt.xlabel("Time [s]")
plt.ylabel("Displacement in X-direction [m]")
plt.title("Movement vs. Time with Varying Surface stiffness")

plt.legend(sorted_lines, sorted_labels, title="Stiffness")

plt.grid(True)
plt.tight_layout()
plt.show()
