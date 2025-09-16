#!/bin/bash
# -*- coding: utf-8 -*-
"""
Create LibriSpeech manifest files
This script creates manifest files for all LibriSpeech subsets
"""

set -e  # Exit on any error

# Configuration
LIBRISPEECH_ROOT="/root/autodl-tmp/dataset/LibriSpeech"
OUTPUT_DIR="/root/autodl-tmp/WhisperCodec/manifests/librispeech"
SCRIPT_DIR="/root/autodl-tmp/WhisperCodec/utils"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🎵 Creating LibriSpeech Manifest Files"
echo "====================================="
echo "LibriSpeech root: $LIBRISPEECH_ROOT"
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

# Choose manifest creation mode
echo "🔄 Creating manifest files..."
echo ""

# Option 1: Combined training manifest (recommended for simple setup)
echo "💡 Creating combined training manifest (recommended)..."
python "$SCRIPT_PATH" \
    --librispeech_root "$LIBRISPEECH_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --combine_train \
    --train_subsets train-clean-100 train-clean-360 train-other-500 \
    --subsets dev-clean test-clean

echo ""
echo "📝 Alternative: If you prefer separate training manifests, uncomment the following:"
echo "# python \"\$SCRIPT_PATH\" \\"
echo "#     --librispeech_root \"\$LIBRISPEECH_ROOT\" \\"
echo "#     --output_dir \"\$OUTPUT_DIR\" \\"
echo "#     --subsets dev-clean test-clean train-clean-100 train-clean-360 train-other-500"

echo ""
echo "✅ All LibriSpeech manifest files created successfully!"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "📋 Generated files:"
ls -la "$OUTPUT_DIR"/*.jsonl

echo ""
echo "🎯 Next steps:"
echo "1. You can now use these manifest files in your WhisperCodec training"
echo "2. Update your config files to point to these manifest paths"
echo "3. Run training with the new LibriSpeech data"
echo ""
echo "🎯 Configuration Examples:"
echo ""
echo "📋 Option 1: Combined training manifest (simpler setup):"
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
echo "📋 Option 2: Separate training manifests (more control):"
echo "  train_ds_meta:"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_train_clean_100.jsonl"
echo "      weight: 1.0"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_train_clean_360.jsonl"
echo "      weight: 1.0"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_train_other_500.jsonl"
echo "      weight: 0.5  # Lower weight for more challenging data"
echo ""
echo "  val_ds_meta:"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_dev_clean.jsonl"
echo ""
echo "  test_ds_meta:"
echo "    - manifest_filepath: $OUTPUT_DIR/librispeech_test_clean.jsonl"
