#!/bin/bash

# Default values

BASE_PATH="/Users/uncpham/Repo/VLSP_Legal_Document_Retrieval"
run_pipeline() {
    cmd="python $BASE_PATH/src/pipeline/baseline.py"
    
    if [ -n "$BASE_PATH" ]; then
        cmd="$cmd base_path=$BASE_PATH"
    fi
    
    # Add data path overrides if specified
    if [ -n "$CORPUS_PATH" ]; then
        cmd="$cmd data.corpus_path=$CORPUS_PATH"
    fi
    if [ -n "$TRAIN_PATH" ]; then
        cmd="$cmd data.train_path=$TRAIN_PATH"
    fi
    if [ -n "$RESULT_PATH" ]; then
        cmd="$cmd output.result_path=$RESULT_PATH"
    fi

    # Add database and chunking overrides
    if [ -n "$COLLECTION_NAME" ]; then
        cmd="$cmd db.collection_name=$COLLECTION_NAME"
    fi
    if [ -n "$MAX_TOKENS" ]; then
        cmd="$cmd chunking.max_tokens=$MAX_TOKENS"
    fi

    # Add retrieval overrides
    if [ -n "$LIMIT" ]; then
        cmd="$cmd retrieval.limit=$LIMIT"
    fi
    if [ -n "$THRESHOLD" ]; then
        cmd="$cmd retrieval.threshold=$THRESHOLD"
    fi

    # Add embedding model overrides
    if [ -n "$EMBEDDING_MODEL" ]; then
        cmd="$cmd embedding.embedding_model=$EMBEDDING_MODEL"
    fi
    if [ -n "$VECTOR_SIZE" ]; then
        cmd="$cmd embedding.vector_size=$VECTOR_SIZE"
    fi

    echo "Running pipeline with the following command:"
    echo "$cmd"
    eval $cmd
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--base-path)
            if [ -z "$2" ]; then
                echo "Please provide a base path."
                exit 1
            fi
            BASE_PATH="$2"
            shift 2
            ;;
        -c|--corpus-path)
            if [ -z "$2" ]; then
                echo "Please provide a corpus path."
                exit 1
            fi
            CORPUS_PATH="$2"
            shift 2
            ;;
        -t|--train-path)
            if [ -z "$2" ]; then
                echo "Please provide a train path."
                exit 1
            fi
            TRAIN_PATH="$2"
            shift 2
            ;;
        -r|--result-path)
            if [ -z "$2" ]; then
                echo "Please provide a result path."
                exit 1
            fi
            RESULT_PATH="$2"
            shift 2
            ;;
        -n|--collection-name)
            if [ -z "$2" ]; then
                echo "Please provide a collection name."
                exit 1
            fi
            COLLECTION_NAME="$2"
            shift 2
            ;;
        -m|--max-tokens)
            if [ -z "$2" ]; then
                echo "Please provide a max tokens value."
                exit 1
            fi
            MAX_TOKENS="$2"
            shift 2
            ;;
        -l|--limit)
            if [ -z "$2" ]; then
                echo "Please provide a retrieval limit."
                exit 1
            fi
            LIMIT="$2"
            shift 2
            ;;
        -th|--threshold)
            if [ -z "$2" ]; then
                echo "Please provide a retrieval threshold."
                exit 1
            fi
            THRESHOLD="$2"
            shift 2
            ;;
        -e|--embedding-model)
            if [ -z "$2" ]; then
                echo "Please provide an embedding model."
                exit 1
            fi
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        -v|--vector-size)
            if [ -z "$2" ]; then
                echo "Please provide a vector size."
                exit 1
            fi
            VECTOR_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -b, --base-path          Set the base path (default: /workspace/thviet/trungnnv/vlsp-2025)."
            echo "  -c, --corpus-path        Set the corpus file path."
            echo "  -t, --train-path         Set the train file path."
            echo "  -r, --result-path        Set the result file path."
            echo "  -n, --collection-name    Set the Qdrant collection name."
            echo "  -m, --max-tokens         Set the maximum tokens for chunking."
            echo "  -l, --limit              Set the retrieval limit."
            echo "  -th, --threshold         Set the retrieval threshold."
            echo "  -e, --embedding-model    Set the embedding model name."
            echo "  -v, --vector-size        Set the embedding vector size."
            echo "  -h, --help               Show this help message."
            exit 0
            ;;
        *)
            echo "Invalid option: $1. Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Run the pipeline
run_pipeline

echo "Pipeline execution completed."