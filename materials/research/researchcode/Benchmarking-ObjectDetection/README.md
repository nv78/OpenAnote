# Underwater Image Datasets for Fine-Tuning

This repository utilizes three underwater image datasets for fine-tuning machine learning models related to object detection and segmentation in aquatic environments. Each dataset is curated for different aspects of underwater perception such as pollution detection, object classification, and semantic understanding.

---

## Datasets Used

### 1. COU (Classification of Underwater Trash)

- **Description**: The COU dataset contains underwater images from both closed-water (pool) and open-water (lake and ocean) environments. It includes 24 different classes of objects including marine debris, dive tools, and AUVs. This dataset is valuable for training robust classifiers and detectors in mixed aquatic settings.
- **Classes**:
  `unknown instances`, `scissors`, `plastic cup`, `metal rod`, `fork`, `bottle`, `soda can`, `case`, `plastic bag`, `cup`, `googles`, `flipper`, `loco`, `aqua`, `pip`, `snorkle`, `spoon`, `lure`, `screwdriver`, `car`, `tripod`, `ROV`, `knife`, `dive weight`
- **Source**: 
    - [COU Dataset on UMN Conservancy](https://conservancy.umn.edu/items/f4294f39-d0c7-491a-bc9c-9a72714d57e2)
    - [Dataset on Hugging Face](https://huggingface.co/datasets/anyaeross/COU) 
- **Reference Paper**: ["Image Dataset and Benchmark for Underwater Marine Debris Classification"](https://ieeexplore.ieee.org/document/9561454)

---

### 2. TrashCan 1.0

- **Description**: TrashCan 1.0 is an image dataset focused on detecting marine debris in underwater environments. It covers a diverse range of object types including animals, plants, ROVs, and various categories of trash such as fabric, plastic, metal, and rubber. This dataset is suitable for developing generalized models for underwater litter detection and segmentation.
- **Classes**:  
  `animal_crab`, `animal_eel`, `animal_etc`, `animal_fish`, `animal_shells`, `animal_starfish`, `plant`, `rov`, `trash_etc`, `trash_fabric`, `trash_fishing_gear`, `trash_metal`, `trash_paper`, `trash_plastic`, `trash_rubber`, `trash_wood`
- **Source**: 
    - [TrashCan 1.0 Dataset on UMN Conservancy](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) 
    - [Dataset on Hugging Face](https://huggingface.co/datasets/anyaeross/trashcan)
- **Reference Paper**: ["TrashCan: A Benchmark Dataset for Litter Detection in the Underwater Environment"](https://ieeexplore.ieee.org/document/9561455)

---

### 3. WQT and DG-YOLO

- **Description**: This dataset is part of the DG-YOLO benchmark for domain generalization in underwater object detection. It contains multiple underwater categories and is useful for evaluating model performance under distribution shifts. A hosted version is available on Hugging Face.
- **Classes**:  
  `echinus`, `holothurian`, `scallop`, `starfish`, `waterweeds`
- **Source**:
  - [DG-YOLO Paper on arXiv](https://arxiv.org/abs/2004.06333)
  - [Dataset on Hugging Face](https://huggingface.co/datasets/Francesco/underwater-objects-5v7p8)

- **Reference Paper**: ["Domain Generalization for Underwater Object Detection with DG-YOLO"](https://arxiv.org/abs/2004.06333)

---