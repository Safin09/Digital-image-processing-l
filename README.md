
# Digital Image Processing - Assignment 1

## **Objective**
This assignment focuses on fundamental image processing tasks using OpenCV and Matplotlib in Python. The two main tasks are:

1. Extracting a **100×100** region from the center of an image.
2. Converting the image to **HSV color space** and visualizing the Hue, Saturation, and Value components in grayscale.

---

## **Requirements**
Ensure you have the following installed in your Google Colab or local Python environment:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- PIL (Pillow)
- Google Colab (if using Google Drive for image storage)

To install missing dependencies, run:

```sh
pip install opencv-python numpy matplotlib pillow
```

## **Usage Instructions**
### **Step 1: Clone the Repository**
#### **Create a separate repository for this assignment and upload the .ipynb file. Then, clone it using:**

```sh
git clone https://github.com/your-username/DIP-Assignment-1.git
cd DIP-Assignment-1
```

## **Step 2: Open and Run in Google Colab**
### **If using Google Colab, upload the .ipynb file and make sure your image is in Google Drive. Then, mount your drive:**

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow  # Use only in Google Colab

def extract_center_region(image_path, size=(100, 100)):
    """Extracts a 100x100 region from the center of an image."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None
    
    h, w = image.shape[:2]
    x = w // 2 - size[0] // 2
    y = h // 2 - size[1] // 2
    cropped_image = image[y:y+size[1], x:x+size[0]]
    
    cv2_imshow(cropped_image)  # Display in Google Colab
    return cropped_image

def convert_to_hsv(image_path):
    """Converts an image to HSV and displays Hue, Saturation, and Value channels."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Display using Matplotlib in grayscale
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(h, cmap='gray')
    axes[0].set_title("Hue")
    axes[0].axis("off")
    
    axes[1].imshow(s, cmap='gray')
    axes[1].set_title("Saturation")
    axes[1].axis("off")
    
    axes[2].imshow(v, cmap='gray')
    axes[2].set_title("Value")
    axes[2].axis("off")
    
    plt.show()
    return hsv_image

# Example usage
image_path = "/content/drive/MyDrive/UITS/Eight Semester/Digital Image Processing/Assigments/Class 1/pizza.jpg"
cropped_image = extract_center_region(image_path)
hsv_image = convert_to_hsv(image_path)
```

## **Expected Output**
#### **Task 1: Displays a 100x100 cropped image from the center.**
#### **Task 2: Displays three grayscale images representing:**
* Hue component
* Saturation component
* Value component
