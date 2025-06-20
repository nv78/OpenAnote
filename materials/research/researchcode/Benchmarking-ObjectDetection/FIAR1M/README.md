# FIAR1M Dataset

## Overview

FIAR1M is a large-scale dataset designed for object detection tasks in aerial imagery. It contains high-resolution images annotated with various object categories, making it suitable for benchmarking object detection algorithms in remote sensing and aerial image analysis.

## Dataset Contents

- **Images:** High-resolution aerial images.
- **Annotations:** Bounding box annotations for multiple object categories.
- **Format:** All data is under [datafiles directory](materials\research\researchcode\Benchmarking-ObjectDetection\FIAR1M\datafiles) split into training, validation, and test sets. All the images are in their respective directories, and annotations are provided in the CSV files (metadata.csv) with columns for image ID, bounding box coordinates, and class labels.
- **Original Annotations:** The original annotations are provided in the [original_annotations directory](materials\research\researchcode\Benchmarking-ObjectDetection\FIAR1M\original_annotations), which includes the original annotations in their original format.
- **Link to Original Dataset:** The original dataset can be accessed from the [FAIR1M HuggingFace repository](https://huggingface.co/datasets/blanchon/FAIR1M).

## Citation

**APA:**  
Sun, P., Li, H., Zhu, D., & Zhu, X. (2021). FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery. *arXiv preprint arXiv:2103.05569*. https://arxiv.org/abs/2103.05569

[FAIR1M Paper on arXiv](https://arxiv.org/abs/2103.05569)
