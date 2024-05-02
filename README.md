
**Project: Aircraft Docking Guidance System**

This project implements an aircraft docking guidance system using deep learning techniques. The system leverages Roboflow for data labeling, YOLOv5 for model training, PyTorch for model deployment, and ONNX conversion for efficient inference on a Raspberry Pi.

**Getting Started**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Naumu77/AIENHANCEDPlaneDockingSystem.git
   ```


**Data Preparation**

Can go through my data through Roboflow [https://app.roboflow.com/plane-ggvsu/avdgs/1]
or can access trained data in this repository [https://github.com/Naumu77/AIENHANCEDPlaneDockingSystem/tree/main/Data/datatrain2]

1. **Getting started to Roboflow Project:**

   - Sign up for a free Roboflow account at [https://roboflow.com/](https://roboflow.com/).
   - Create a new project for your aircraft docking dataset.

2. **Label Your Data:**

   - Upload your aircraft docking images and videos to your Roboflow project.
   - Use Roboflow's annotation tools to label bounding boxes around the aircraft in each image or frame. Label the appropriate class (e.g., "aircraft").

3. **Export Labeled Data:**

   - Once labeling is complete, export the labeled dataset in a format compatible with YOLOv5. Roboflow typically provides options for YOLO,COCO, VOC, and custom formats.

**Model Training**

1. **Download YOLOv5:**

   ```bash
   git clone https://github.com/ultralytics/yolov5
   ```

2. **Modify YOLOv5 Configuration (Optional):**

   - You may choose to adjust the YOLOv5 configuration file (`data/custom_data.yaml`) to fine-tune the model for your specific dataset and requirements. This could involve modifying parameters like class names, anchor boxes, or learning rate. Refer to the YOLOv5 documentation for guidance on configuration options.

3. **Train the Model:**

   ```bash
   cd yolov5
   python train.py --data data/aircraft.yaml --img-size 640 --batch-size 8 --epochs 100 --weights yolov5s.pt  # Adjust hyperparameters as needed
   ```

   - Replace `yolov5s.pt` with a pre-trained YOLOv5 weight file if you want to use transfer learning (recommended). You can download pre-trained weights from the Ultralytics website.

**Model Deployment**

1. **Convert Model to ONNX:**

   ```bash
   python export.py --weights runs/train/exp1/weights/best.pt --img-size 640 --export onnx --outfile aircraft_docking_guidance.onnx
   ```

   - Replace `runs/train/exp1/weights/last.pt` with the path to your trained model weights.
   after training is completed the best.pt and last.pt are created automatically

2.**For my Project model named "plane.pt" have converted to "plane.onnx":** 
you can view in this repository [https://github.com/Naumu77/AIENHANCEDPlaneDockingSystem/tree/main/models]
       ready to be deployed to the raspberry pi 
**Deploy to Raspberry Pi:**
To ensure smooth deployment on a Raspberry Pi, it's crucial to undergo the conversion process from PyTorch to ONNX format. This conversion optimizes compatibility and efficiency for the Pi's architecture. By converting the model beforehand, you streamline the deployment process and enhance performance on the Raspberry Pi platform.

   - Transfer the `plane.onnx` file to your Raspberry Pi.
   - Use a framework like TensorFlow Lite or PyTorch Mobile to load and run inference on the ONNX model. Refer to the documentation of your chosen framework for Raspberry Pi deployment instructions.

 
**Additional Notes**

training notebook [https://colab.research.google.com/drive/1XmuTuzO1GCwCaeicX6bLFZTk26zEGU5t#scrollTo=GV1Al5NZbwc9]
conversion pytorch to onnx notebook [https://colab.research.google.com/drive/1kbLkAc1AvEfSCQ3A17cNwATZbnVNp9nC]
roboflow labeling [https://app.roboflow.com/plane-ggvsu/avdgs/1]
