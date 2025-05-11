# Img-Comp
Image Compression using KMeans, Vector Quantization
A Flask web application that compresses images using Vector Quantization with K-Means clustering algorithm, reducing file size while preserving visual quality.

Key Features
Smart Compression: Uses K-Means clustering to reduce color space

Web Interface: User-friendly UI for uploading/downloading images

Metrics: Calculates compression ratio for quality assessment

Format Support: Handles JPG, PNG, and JPEG formats

Block Processing: Configurable block size for compression granularity

Technical Implementation
Core Technologies
Python 3

Flask (Web Framework)

Scikit-learn (K-Means implementation)

Pillow (Image processing)

NumPy (Matrix operations)

How It Works
Divides image into small blocks (default 4x4 pixels)

Uses K-Means to find most representative colors

Reconstructs image using cluster centroids

Calculates compression metrics
