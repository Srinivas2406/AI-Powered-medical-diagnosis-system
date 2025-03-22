# AI-Powered Medical Diagnosis System  
This project predicts diseases (Diabetes, Liver Disease, Thyroid) using Machine Learning.  
## How to Run  
1. Clone the repository  
2. Install dependencies (`pip install -r requirements.txt`)  
3. Run `streamlit run app.py`  

#Folder Structure
📂 Project 
│── 📂 database/              # Contains CSV files  
│   ├── diabetes.csv  
│   ├── liver.csv  
│   ├── thyroid.csv  
│  
│── 📂 models/                # Contains trained model files  
│   ├── diabetes_model.pkl  
│   ├── liver_model.pkl  
│   ├── thyroid_model.pkl  
│  
│── app.py                    # Streamlit frontend  
│── train.py                   # Model training script 
