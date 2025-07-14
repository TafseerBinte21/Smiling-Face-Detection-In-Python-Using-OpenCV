import cv2
import os
import matplotlib.pyplot as plt

# Load classifiers
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
lbp_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

# Folder containing images
folder = "sample_faces"
for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Haar detection
    faces_haar = haar_cascade.detectMultiScale(gray, 1.3, 5)

    # LBP detection
    faces_lbp = lbp_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles on copies
    img_haar = img.copy()
    img_lbp = img.copy()

    for (x, y, w, h) in faces_haar:
        cv2.rectangle(img_haar, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in faces_lbp:
        cv2.rectangle(img_lbp, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert to RGB
    img_haar_rgb = cv2.cvtColor(img_haar, cv2.COLOR_BGR2RGB)
    img_lbp_rgb = cv2.cvtColor(img_lbp, cv2.COLOR_BGR2RGB)

    # Show comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img_haar_rgb)
    axs[0].set_title(f"Haar Cascade - {filename}")
    axs[0].axis("off")

    axs[1].imshow(img_lbp_rgb)
    axs[1].set_title(f"LBP Cascade - {filename}")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
