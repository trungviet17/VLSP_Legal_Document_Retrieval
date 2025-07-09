# VLSP_Legal_Document_Retrieval

## Structure 
```
📦 src
 ┣ 📂 benchmark
 ┃ ┗ 📜 cal_metric.py        # Calculate performance evaluation metrics
 ┃
 ┣ 📂 config
 ┃ ┣ 📜 default.yaml         # Default project configuration
 ┃ ┗ 📜 envconfig.py         # Manage API environment variables (OpenAI, Gemini, Groq)
 ┃
 ┣ 📂 core
 ┃ ┣ 📂 chunking
 ┃ ┃ ┗ 📜 baseline_chunking.py   # Chunking logic for splitting documents
 ┃ ┣ 📂 db
 ┃ ┃ ┗ 📜 qdrant.py              # Qdrant vector database connector
 ┃ ┗ 📂 retriever
 ┃   ┗ 📜 baseline_retriever.py  # Baseline retrieval logic
 ┃
 ┣ 📂 data                   # Data source 
 ┣ 📂 data_processing
 ┃ ┗ 📜 eda.ipynb            # Jupyter notebook for data analysis and exploration
 ┃
 ┗ 📂 utils
    ┣ 📜 model.py            # Factory for LLM and embedding models
    ┗ 📜 output_parser.py    # Parser for processing language model outputs
```


## How to run baseline 


1. Add data. in vlsp folder, must be add 2 file .json (legal_corpus.json and train.json)
```
cd src
mkdir data/vlsp 
```


2. Run baseline : 
```
python -m venv vlsp-env (python 3.11+)

pip install uv 
uv pip install -r requirements.txt

cd src/scripts 
chmod +x run_baseline.sh --base_path= <your_path> --device=<cuda / cpu>

./run_baseline.sh
```

