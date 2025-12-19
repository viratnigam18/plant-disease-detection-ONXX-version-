# AGROSCAN – Plant Disease Detection using AI (ONNX)

AGROSCAN is an AI-based system that detects **plant leaf diseases from images**, helping farmers and agricultural practitioners identify crop health issues at an early stage.

This project is built as part of an **AI/ML hackathon**, focusing on **model training, experimentation, and ONNX-based inference**.

---

## Dataset
- **Source:** Kaggle – Plant Disease Dataset  
- **Link:** https://www.kaggle.com/datasets/emmarex/plantdisease  
- **Details:** Images of healthy and diseased leaves of:
  - Tomato
  - Potato
  - Bell Pepper

---

## Model & Training
- **Model Type:** Convolutional Neural Network (CNN)
- **Input Image Size:** 128 × 128
- **Epochs:** 30
- **Training Accuracy:** 95.25%
- **Best Validation Accuracy:** 88.52%
- **Final Validation Loss:** 0.63

The CNN learns visual patterns such as leaf spots, discoloration, and texture changes to classify plant diseases.

---

## How to Train the Model

bash
python leaf_disease_detection.py


This generates:leaf_disease_model.h5

Convert the trained model to ONNX format:

python -m tf2onnx.convert --keras leaf_disease_model.h5 --output leaf_disease_model.onnx --opset 13




###############################################
To run the file use 

python onnx_run.py

## Note- please save image in OIP.jpeg format only 
## Use image from any source from below class only 

Pepper__bell___Bacterial_spot

Pepper__bell__healthy

Potato__Early_blight

Potato__healthy

Potato__Late_blight

Tomato__Target_Spot

Tomato__Tomato_mosaic_virus

Tomato__Tomato_YellowLeaf_Curl_Virus

Tomato_Bacterial_spot

Tomato_Early_blight

Tomato_healthy

Tomato_Late_blight

Tomato_Leaf_Mold

Tomato_Septoria_leaf_spot

Tomato_Spider_mites_Two_spotted_spider_mite



output format:

Predicted class : <Disease_Name>
Confidence      : XX.XX%


we have given some testing data feel free to use your own also 


## Cation- use only single leaf image with no obstruction like hand etc, for better accuracy .

Thank you
Virat Nigam 
