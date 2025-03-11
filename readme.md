# Paddy Disease Classification

This project focuses on classifying paddy diseases using Convolutional Neural Networks (CNN) and Transformer models. The project includes training and evaluation scripts, as well as pre-trained models and training history visualizations.

## Project Structure

```
.gitattributes
CNN_Classification.csv
CNN_confusion_matrix_with_metadata.png
CNN_confusion_matrix_without_metadata.png
CNN_eval.ipynb
CNN_train_with_metadata.ipynb
CNN_train_without_metadata.ipynb
CNN_training_history_with_metadata.png
CNN_training_history_without_metadata.png
paddy_disease_densenet_with_metadata.pth
paddy_disease_densenet_without_metadata.pth
paddy_disease_transformer_with_metadata.pth
paddy_disease_transformer_without_metadata.pth
train.csv
Transformer_Classification.csv
...
```

## Files and Directories

- `CNN_eval.ipynb`: Jupyter notebook for evaluating the CNN models.
- `CNN_train_with_metadata.ipynb`: Jupyter notebook for training the CNN model with metadata.
- `CNN_train_without_metadata.ipynb`: Jupyter notebook for training the CNN model without metadata.
- `CNN_Classification.csv`: CSV file containing CNN classification results.
- `CNN_confusion_matrix_with_metadata.png`: Confusion matrix for CNN model trained with metadata.
- `CNN_confusion_matrix_without_metadata.png`: Confusion matrix for CNN model trained without metadata.
- `CNN_training_history_with_metadata.png`: Training history for CNN model trained with metadata.
- `CNN_training_history_without_metadata.png`: Training history for CNN model trained without metadata.
- `paddy_disease_densenet_with_metadata.pth`: Pre-trained DenseNet model with metadata.
- `paddy_disease_densenet_without_metadata.pth`: Pre-trained DenseNet model without metadata.
- `paddy_disease_transformer_with_metadata.pth`: Pre-trained Transformer model with metadata.
- `paddy_disease_transformer_without_metadata.pth`: Pre-trained Transformer model without metadata.
- `train.csv`: CSV file containing training data.
- `Transformer_Classification.csv`: CSV file containing Transformer classification results.

## Models Used

### CNN Models

1. **DenseNet121**:
   - Used for feature extraction from images.
   - Trained with and without metadata.
   - Pre-trained models: `paddy_disease_densenet_with_metadata.pth`, `paddy_disease_densenet_without_metadata.pth`.

### Transformer Models

1. **DeiT (Data-efficient Image Transformers)**:
   - Used for image classification.
   - Trained with and without metadata.
   - Pre-trained models: `paddy_disease_transformer_with_metadata.pth`, `paddy_disease_transformer_without_metadata.pth`.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- timm (for Transformer models)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/paddy_classification.git
   cd paddy_classification
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### Training the CNN Model

1. To train the CNN model with metadata, open and run the `CNN_train_with_metadata.ipynb` notebook:
   ```sh
   jupyter notebook CNN_train_with_metadata.ipynb
   ```

2. To train the CNN model without metadata, open and run the `CNN_train_without_metadata.ipynb` notebook:
   ```sh
   jupyter notebook CNN_train_without_metadata.ipynb
   ```

#### Training the Transformer Model

1. To train the Transformer model with metadata, open and run the `Transformer_train_with_metadata.ipynb` notebook:
   ```sh
   jupyter notebook Transformer_train_with_metadata.ipynb
   ```

2. To train the Transformer model without metadata, open and run the `Transformer_train_without_metadata.ipynb` notebook:
   ```sh
   jupyter notebook Transformer_train_without_metadata.ipynb
   ```

#### Making Predictions

1. To make predictions using the CNN models, open and run the `CNN_eval.ipynb` notebook, the results will be stored in `CNN_Classification.csv`:
   ```sh
   jupyter notebook CNN_eval.ipynb
   ```

2. To make predictions using the Transformer models, open and run the `Transformer_eval.ipynb` notebook, the results will be stored in `Transformer_Classification.csv`:
   ```sh
   jupyter notebook Transformer_eval.ipynb
   ```

### Note

- Ensure that the images to be tested are placed in the `test_images` folder in the same directory.

### Results

The results of the classification are stored in the CSV files and visualized using confusion matrices and training history plots.