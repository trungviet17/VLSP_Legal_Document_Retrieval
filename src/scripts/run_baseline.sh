#!/bin/bash

# Default values

BASE_PATH="/workspace/thviet/trungnnv/vlsp-2025"
CONFIG_PATH="default"

run_pipeline() {
    cmd="python $BASE_PATH/src/pipeline/baseline.py --config-name=$CONFIG_PATH"
    
    if [ -n "$BASE_PATH" ]; then
        cmd="$cmd base_path=$BASE_PATH"
    fi
    if [ -n "$CORPUS_PATH" ]; then
        cmd="$cmd data.corpus_path=$CORPUS_PATH"
    fi
    if [ -n "$TRAIN_PATH" ]; then
        cmd="$cmd data.train_path=$TRAIN_PATH"
    fi
    if [ -n "$RESULT_PATH" ]; then
        cmd="$cmd output.result_path=$RESULT_PATH"
    fi
    if [ -n "$COLLECTION_NAME" ]; then
        cmd="$cmd db.collection_name=$COLLECTION_NAME"
    fi
    if [ -n "$DEVICE" ]; then
        cmd="$cmd embedding.device=$DEVICE"
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
        -d|--device)
            if [ -z "$2" ]; then
                echo "Please provide a device (e.g., cuda, cpu)."
                exit 1
            fi
            DEVICE="$2"
            shift 2
            ;;
        -cfg|--config-path)
            if [ -z "$2" ]; then
                echo "Please provide a config path (e.g., default, punc_chunking)."
                exit 1
            fi
            CONFIG_PATH="$2"
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
            echo "  -d, --device             Set the device (e.g., cuda, cpu)." 
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