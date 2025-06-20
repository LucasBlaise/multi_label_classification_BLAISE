Project: Multi-Label Classification of Scientific Literature  
Author: Lucas Blaise  
Date: June 2025


Description:


This project applies a transformer-based model (BERT) to perform multi-label classification on scientific documents from the NASA SciX corpus. Each document (title + abstract) is classified into multiple keyword labels using a fine-tuned model.


Key Features:


- Dataset: adsabs/SciX_UAT_keywords (Hugging Face Datasets)
- Model: BERT-base (with sigmoid head for multi-label output)
- Training: 1 epoch, batch size of 8, max token length of 128
- Thresholding: 0.2 applied to sigmoid outputs for prediction
- Evaluation: Classification report with F1, precision, and recall metrics


Usage:


1. install all the required packages by running:
pip install -r requirements.txt

2. run the code 
