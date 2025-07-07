# Word2Vec From Scratch 🧠💬

This project implements a Word2Vec model (CBOW architecture) **from scratch using PyTorch**. It includes a full training pipeline, dataset preprocessing, and a simple app to visualize word embeddings.

## 🚀 Features
- Custom implementation of Word2Vec (CBOW)
- PyTorch model training and saving
- Data preparation pipeline (raw → processed)
- Embedding visualization interface via `APP.py`
- Modular project structure for easy experimentation

## 📁 Project Structure

```

WORD2VEC_FROM_SCRATCH/
├── data/                                   # Contains all data-related files
│   ├── external/                          
│   │   └── .gitkeep                        
│   ├── interim/                            
│   │   ├── .gitkeep                        
│   │   ├── id2word                         # Mapping of word IDs to actual words
│   │   └── word2id                         # Mapping of words to their IDs
│   ├── processed/                        
│   │   ├── .gitkeep                        
│   │   ├── train_dataset                   # Processed training dataset
│   │   └── val_dataset                     # Processed validation dataset
│   ├── raw/                              
│   │   ├── .gitkeep                        
│   │   └── raw_dataset.pkl                 # Raw dataset in pickle format
│   └── notebooks/                          
│       ├── .gitkeep                        
│       └── word2vec_notes.ipynb            # Notebook for Word2Vec experiments
│
├── reports/                               
│   └── figures/                            
│       ├── .gitkeep                        
│       ├── 1.png                           # Example visualization 1
│       ├── 2.png                           # Example visualization 2
│       ├── 3.png                           # Example visualization 3
│       ├── 4.png                           # Example visualization 4
│       └── 5.png                           # Example visualization 5
│
├── src/                                    # Source code directory
│   ├── __init__.py                         
│   ├── .gitignore                          
│   ├── APP.py                              # Main application/entry point
│   ├── README.md                           # Project documentation
│   ├── data/                               
│   │   └── make_dataset.py                 # Script for data processing
│   ├── helpers/
│   │   └── helpers.py                      # Helper/utility functions
│   ├── models/                             
│   │   ├── __init__.py                     # Makes models a Python package
│   │   ├── .gitkeep                       
│   │   ├── config.yaml                     # Model configuration file
│   │   ├── Model.py                        # Main model implementation
│   │   └── trainer.py                      # Training logic
│   └── Utils/                             
│       ├── __init__.py                    
│       ├── .gitkeep                        
│       ├── constants.py                    # Project constants
│       ├── dataloader.py                   # Data loading utilities
│       └── make_dataset.py                 # Alternative dataset creation script
│
├── requirements.txt                        # Python dependencies
├── Train.py                                # Main training script
└── word2vec_model.pth                      # Trained Word2Vec model weights

```




App interface samples:
## [▶️ Watch Demo Video]
[[https://github.com/
](https://github.com/user-attachments/assets/15b09d29-0a72-4c6d-b3de-63e09cbf5dbc)](https://github.com/user-attachments/assets/15b09d29-0a72-4c6d-b3de-63e09cbf5dbc)



## 🔧 Installation

```bash
git clone https://github.com/yourusername/word2vec_from_scratch.git
cd word2vec_from_scratch
pip install -r requirements.txt
```

## 🏁 Running the Project
```bash
 > ⚠️ you need to edit config file and constant file before running train.py
```
### Train the model

```bash
python Train.py
```

### Launch the App (e.g., Streamlit)

```bash
python -m streamlit run App.py

> ⚠️ You may need to adjust the paths in the scripts depending on your working directory.
> ⚠️ You may need to adjust the tokenizer function to use a different dataset

## 📊 Model

- Architecture: CBOW Word2Vec
- Framework: PyTorch
- EMBED_DIMENSION, window size(CBOW_N_WORDS),MIN_WORD_FREQUENCY,MAX_SEQUENCE_LENGTH  – configurable in `config.py`

## 📂 Dataset

- Preprocessed text data is located in `data/processed`
- Raw text file: `data/raw/raw_dataset.pkl`
- Vocabulary mappings: `data/interim/word2id`, `id2word`

## ✅ TODOs / Future Improvements

- Add skip-gram option
- Visualize embeddings with t-SNE or PCA
- Add evaluation metrics
- Add schedule learning rate 

## 🤝 References:
   https://medium.com/data-science/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0


## 📄 License

MIT License

---

*Made with ❤️ by [Ahmed Shalaby]*
