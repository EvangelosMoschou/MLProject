#!/bin/bash

# Build script for Team1 presentation
# Compiles the LaTeX file and organizes build artifacts

set -e  # Exit on error

echo "🔨 Building Team1.pdf..."

# Run XeLaTeX twice (first run for content, second for TOC/references)
echo "  First pass..."
xelatex -interaction=nonstopmode -output-directory=build Team1.tex > /dev/null

echo "  Second pass..."
xelatex -interaction=nonstopmode -output-directory=build Team1.tex > /dev/null

# Copy the PDF to the current directory
cp build/Team1.pdf .

echo "✅ Build complete! Team1.pdf is ready."
echo "📁 Build artifacts are in ./build/"

# Show file size
ls -lh Team1.pdf | awk '{print "📄 Size: " $5}'
