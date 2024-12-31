
# Hackathon Project Report  

**Project Title**: Sentiment Classification using BERT for Structured Text Data  
**Team Name**: ssudhanshu488  

## 1. Problem Statement  
With the exponential growth of textual data on social platforms, identifying the sentiment or classification of text is crucial for applications like content moderation, fake news detection, and targeted advertising. The task is to build an effective sentiment classification model that achieves high accuracy and generalizability, particularly for short text entries.  

---  

## 2. Objective  
The primary goal of this project is to design, train, and evaluate a machine learning model that can classify text entries based on their sentiment or class labels. The focus is on leveraging transformer-based architectures for state-of-the-art performance while ensuring the results are presented in a structured format.  

---  

## 3. Dataset  
- **Source**: The dataset provided includes 30,000 labeled samples for training.  
- **Test Set**: A separate `test.tsv` file is used for evaluation.  
- **Structure**:  
  - **Text**: Input text for classification.  
  - **Label**: Ground truth label for each entry (e.g., sentiment or category).  
- **Preprocessing**:  
  - Tokenization using `bert-base-uncased`.  
  - Normalization of text by truncating/padding to a maximum sequence length of 512 tokens.  

---  

## 4. Methodology  
### 4.1 Model Architecture  
- **Pretrained Model**: BERT (`bert-base-uncased`) from Hugging Face Transformers.  
- **Fine-tuning**: The BERT model is fine-tuned on the training dataset for the specific classification task.  

### 4.2 Training Process  
- **Framework**: Hugging Face Transformers library.  
- **Parameters**:  
  - Batch size: 8  
  - Epochs: 2  
  - Optimizer: AdamW  
  - Learning rate: Automatically adjusted.  
- **Training Dataset Size**: 30,000 samples.  
- **Evaluation Strategy**: Performed after every epoch.  

### 4.3 Evaluation Metrics  
The following metrics were used to evaluate model performance:  
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- AUC-ROC: For both binary and multiclass scenarios.  

---  

## 5. Results  
### Training Dataset Metrics  
- **Accuracy**: 99.52%  
- **Precision**: 99.52%  
- **Recall**: 99.51%  
- **F1-Score**: 99.52%  
- **AUC-ROC**: 99.99%  

### Test Dataset Metrics  
- **Accuracy**: 99.85%  
- **Precision**: 99.85%  
- **Recall**: 99.85%  
- **F1-Score**: 99.85%  
- **AUC-ROC**: 99.99%  

---  

## 6. Output  
The model generates structured predictions for the test dataset as required:  
- **Input Format**: `test.tsv`  
- **Output Format**: Saved as `result.txt`:  
```json
{
    ["Trump's new policy reviewed", 1],
    ["Shocking claims about health!", 0]
}
```  

---  

## 7. Challenges  
- Fine-tuning a transformer-based model on large datasets posed memory constraints.  
- Handling class imbalances in the dataset.  
- Optimizing hyperparameters for improved AUC-ROC performance.  

---  

## 8. Future Improvements  
- **Data Augmentation**: Adding synthetic data for underrepresented classes to handle imbalances.  
- **Multi-language Support**: Expanding the model for multilingual datasets.  
- **Optimization**: Experimenting with lightweight transformer models (e.g., DistilBERT) for faster inference.  
- **Explainability**: Adding SHAP or LIME to explain predictions.  

---  

## 9. Tools & Technologies  
- **Programming Language**: Python  
- **Libraries**: Hugging Face Transformers, scikit-learn, Pandas, NumPy, PyTorch  
- **Other Tools**: Weights & Biases (W&B) for experiment tracking and visualization.  

---  

## 10. Conclusion  
This project demonstrates the effectiveness of transformer-based models like BERT in achieving state-of-the-art performance for text classification tasks. The structured outputs and evaluation metrics highlight the robustness of the approach, making it applicable to real-world scenarios.  

--- 
Here is the GitHub link of the project - https://github.com/ssudhanshu488/Jagriti/
