# VLSP Legal Document Retrieval

A comprehensive legal document retrieval system.
## 📁 Folder Structure

```
VLSP_Legal_Document_Retrieval/
├── src/                            # Source code directory
│   ├── main.py                     # Main application entry point
│   ├── llm/                        # Language Model implementations
│   │   ├── __init__.py            # LLM module exports
│   │   ├── base.py                # Abstract base class for LLMs
│   │   ├── openai_model.py        # OpenAI/LiteLLM implementation
│   │   ├── vllm_model.py          # VLLM implementation
│   │   └── gemini_model.py        # Google Gemini implementation
│   ├── embedding_models/           # Embedding model implementations
│   │   └── base.py                # Base embedding model class
│   ├── utils/                      # Utility modules
│   │   ├── configs.py             # Configuration management
│   │   ├── schema.py              # Data schemas and message types
│   │   ├── json_handler.py        # JSON processing utilities
│   │   └── logger.py              # Logging configuration
│   ├── prompts/                    # Prompt templates
│   │   └── template.py            # Prompt template definitions
│   ├── tests/                      # Test files
│   │   └── test_llm.py            # LLM functionality tests
│   └── benchmark/                  # Benchmarking tools
│       └── cal_metric.py          # Metric calculation utilities
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
├── config.yaml.example            # Configuration template
├── .python-version                 # Python version specification
├── .gitignore                      # Git ignore rules
└── LICENSE                         # Project license
```

## 🚀 How to Run the Code

### Setup Instructions

#### 1. Clone and Navigate to Project
```bash
cd VLSP_Legal_Document_Retrieval
```

#### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

uv sync

uv add -r requirements.txt
```

#### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

### Running the LLM Test

```bash
# Run the LLM test from project root
python -m src.tests.test_llm
```


### License

This project is licensed under the terms specified in the `LICENSE` file.

