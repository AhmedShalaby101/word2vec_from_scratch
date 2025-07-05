# Word2Vec From Scratch 🧠💬

This project implements a Word2Vec model (skip-gram architecture) **from scratch using PyTorch**. It includes a full training pipeline, dataset preprocessing, and a simple app to visualize word embeddings.

## 🚀 Features
- Custom implementation of Word2Vec (skip-gram)
- PyTorch model training and saving
- Data preparation pipeline (raw → processed)
- Embedding visualization interface via `APP.py`
- Modular project structure for easy experimentation

## 📁 Project Structure

```
word2vec_from_scratch/
├── APP.py                  # Front-end interface to explore embeddings
├── Train.py                # Script to train the model
├── word2vec_model.pth      # Saved PyTorch model
├── requirements.txt        # Python dependencies
├── README.md               # Project README
├── app_screen_shots/       # Screenshots for app usage
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed training/validation sets
│   ├── interim/            # Intermediate mappings (word2id, id2word)
│   └── external/           # External or third-party data (empty)
└── docs/                   # Reserved for documentation
```

## 📸 Screenshots

App interface samples (from `app_screen_shots/`):

![screenshot1](app_screen_shots/1.png)
![screenshot2](app_screen_shots/2.png)
![screenshot3](app_screen_shots/3.png)
![screenshot4](app_screen_shots/4.png)
![screenshot5](app_screen_shots/5.png)

## 🔧 Installation

```bash
git clone https://github.com/yourusername/word2vec_from_scratch.git
cd word2vec_from_scratch
pip install -r requirements.txt
```

## 🏁 Running the Project

### Train the model

```bash
python Train.py
```

### Launch the App (e.g., Streamlit)

```bash
streamlit run APP.py
```

> ⚠️ You may need to adjust the paths in the scripts depending on your working directory.

## 📊 Model

- Architecture: Skip-gram Word2Vec
- Framework: PyTorch
- Embedding size, window size, negative sampling – configurable in `Train.py`

## 📂 Dataset

- Preprocessed text data is located in `data/processed`
- Raw text file: `data/raw/raw_dataset.pkl`
- Vocabulary mappings: `data/interim/word2id`, `id2word`

## ✅ TODOs / Future Improvements

- Add CBOW option
- Visualize embeddings with t-SNE or PCA
- Add evaluation metrics

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

MIT License

---

*Made with ❤️ by [Your Name]*
