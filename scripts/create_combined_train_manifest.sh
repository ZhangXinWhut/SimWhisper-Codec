#!/bin/bash
# -*- coding: utf-8 -*-
"""
Create LibriSpeech combined training manifest (simplest setup)
This script creates a single combined manifest for all training data
"""

set -e  # Exit on any error

# Configuration
LIBRISPEECH_ROOT="/root/autodl-tmp/dataset/LibriSpeech"
OUTPUT_DIR="/root/autodl-tmp/WhisperCodec/manifests/librispeech"
SCRIPT_DIR="/root/autodl-tmp/WhisperCodec/utils"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🎵 Creating LibriSpeech Combined Training Manifest"
echo "================================================="
echo "Input directory: $LIBRISPEECH_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if LibriSpeech directory exists
if [ ! -d "$LIBRISPEECH_ROOT" ]; then
    echo "❌ Error: LibriSpeech directory not found: $LIBRISPEECH_ROOT"
    exit 1
fi

# Check if script exists
SCRIPT_PATH="$SCRIPT_DIR/create_librispeech_manifest.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Create combined training manifest + dev/test
echo "🔄 Creating combined training manifest..."
python "$SCRIPT_PATH" \
    --librispeech_root "$LIBRISPEECH_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --combine_train \
    --train_subsets train-clean-100 train-clean-360 train-other-500 \
    --subsets dev-clean test-clean

echo ""
echo "✅ Combined manifest created successfully!"
echo "📁 Output files:"
ls -la "$OUTPUT_DIR"/*.jsonl

echo ""
echo "📊 Manifest Statistics:"
echo "  Training: $(wc -l < "$OUTPUT_DIR/librispeech_train_combined.jsonl") utterances"
echo "  Dev:      $(wc -l < "$OUTPUT_DIR/librispeech_dev_clean.jsonl") utterances"
echo "  Test:     $(wc -l < "$OUTPUT_DIR/librispeech_test_clean.jsonl") utterances"

echo ""
echo "🎯 Ready for WhisperCodec training!"
echo ""
echo "📋 Add this to your config YAML:"
echo "  train_ds_meta:"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_train_combined.jsonl"
echo "      weight: 1.0"
echo ""
echo "  val_ds_meta:"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_dev_clean.jsonl"
echo ""
echo "  test_ds_meta:"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_test_clean.jsonl"
echo ""
echo "✨ That's it! Simple 1-file training setup complete."
