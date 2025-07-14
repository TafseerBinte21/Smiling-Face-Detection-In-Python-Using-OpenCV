# Smiling-Face-Detection-In-Python-Using-OpenCV

## Overview

This project demonstrates how to detect faces and smiles in images using OpenCV's pre-trained Haar cascade classifiers. The program detects faces in an image and then checks if the person is smiling by analyzing the mouth region.

The project highlights the following key features:
- **Face detection** using Haar cascades.
- **Smile detection** inside detected faces.
- Visualization of detected faces and smiles with green colored bounding boxes.
- Face with no smiles is with blue colored bounding boxes.
- Display of results using `matplotlib` for compatibility across environments.

Smiling face - 
<img width="1874" height="995" alt="fd" src="https://github.com/user-attachments/assets/e7f574d0-9aa6-4437-ba38-6bead829b0ae" />

Not Smiling face -
<img width="1449" height="830" alt="notfd" src="https://github.com/user-attachments/assets/7fc2b83d-e442-4977-b0cf-0fe285b1e5ae" />


## How it works
1. Load pre-trained Haar cascades for face and smile detection.
2. For each image, detect faces.
3. For each detected face, detect smiles within the face region.
4. Draw rectangles around faces and smiles, and annotate smiling faces.
5. Display the processed images.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- Matplotlib (`matplotlib`)

Install dependencies with:

```bash
pip install opencv-python matplotlib
