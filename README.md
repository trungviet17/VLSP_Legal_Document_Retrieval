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
 ┣ 📂 data                   # Data source 
 ┣ 📂 data_processing
 ┃ ┗ 📜 eda.ipynb            # Jupyter notebook for data analysis and exploration
 ┃
 ┗ 📂 utils
    ┣ 📜 model.py            # Factory for LLM and embedding models
    ┗ 📜 output_parser.py    # Parser for processing language model outputs
```