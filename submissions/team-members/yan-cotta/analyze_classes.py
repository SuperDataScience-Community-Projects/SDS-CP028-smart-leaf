import os
import matplotlib.pyplot as plt

def count_images_per_class(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path) and class_name != "Invalid":
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = num_images
    return class_counts

# Run the analysis
dataset_dir = "dataset"
class_counts = count_images_per_class(dataset_dir)
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

# Verify with Info.txt
expected_counts = {
    "Corn___Common_Rust": 1192, "Corn___Gray_Leaf_Spot": 513, "Corn___Healthy": 1162, "Corn___Northern_Leaf_Blight": 985,
    "Potato___Early_Blight": 1000, "Potato___Healthy": 152, "Potato___Late_Blight": 1000,
    "Rice___Brown_Spot": 613, "Rice___Healthy": 1488, "Rice___Leaf_Blast": 977, "Rice___Neck_Blast": 1000,
    "Wheat___Brown_Rust": 902, "Wheat___Healthy": 1116, "Wheat___Yellow_Rust": 924
}
print("\nVerification with Info.txt:")
for class_name, count in class_counts.items():
    expected = expected_counts.get(class_name, 0)
    print(f"{class_name}: {count} images (Expected: {expected})")

# Visualize
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45, ha="right")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
