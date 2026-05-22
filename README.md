# Open-Vocabulary 3D Scene Segmentation and Editing with Language-Aligned Gaussians

## Project Overview
The project focuses on developing an efficient, scalable, and flexible system for language-conditioned 3D scene manipulation and understanding. It bridges the gap between high-fidelity 3D graphics rendering and high-level natural language processing without requiring category-specific supervision.

## Team Members
- [Aaditya Joshi](https://github.com/AadityaX86)
- [Abhijeet K.C.](https://github.com/Abhijeet-KC)
- [Ankit Neupane](https://github.com/AnkitNeupane007)
- [Lijan Shrestha](https://github.com/Lijan09)

## Detailed Context & Motivation
- **The Core Framework (3DGS):** The project is built upon recent breakthroughs in 3D Gaussian Splatting (3DGS), a technique that utilizes millions of transparent, anisotropic Gaussian primitives to represent real-world environments calculated from multi-view images. While 3DGS stands out for its high speed, exceptional visual fidelity, and real-time novel view synthesis, it fundamentally lacks any form of semantic comprehension.
- **The Problem:** Traditional 3D graphics systems are optimized solely to minimize photometric errors. Because of this, individual Gaussian primitives are considered independent, unlabeled points defined merely by their geometric coherence, opacity, and radiance (color). The system does not naturally understand what it is rendering, meaning users cannot natively query or interact with specific objects (e.g., selecting "the chair" or "all the windows").
- **The Opportunity:** Concurrently, vision-language models have demonstrated an outstanding capability to map visual data and natural language into a shared high-dimensional feature space. This project leverages those capabilities to inject rich semantic attributes directly into the 3D structure, evolving 3DGS from a passive rendering pipeline into an active framework for intelligent scene inspection.

## System Architecture & Pipeline

### Multi-View Ingestion & 3D Reconstruction
The pipeline begins by processing a set of calibrated multi-view RGB images. It utilizes Structure-from-Motion (SfM) to estimate camera extrinsic parameters and construct an initial sparse 3D point cloud. These points are subsequently refined into anisotropic Gaussian ellipsoids that model the continuous volumetric properties, spatial extent, and radiance of the environment.

### Direct Language Embedding Registration
To imbue the scene with semantic meaning, each 3D Gaussian primitive is enriched with high-dimensional language-aligned embeddings. Images rendered via differentiable Gaussian rasterization are passed through pre-trained vision-language foundation models, specifically Contrastive Language-Image Pretraining (CLIP) and the Segment Anything Model (SAM). Semantic similarity with text prompts is optimized by propagating loss signals back through the 3D representation. To preserve memory efficiency and enable broad generalization, a learnable semantic codebook containing prototype embeddings is incorporated into the architecture.

### Natural Language Command Parsing
For real-time interactivity, a lightweight language parser driven by Bidirectional Encoder Representations from Transformers (BERT) and fine-tuned with Low-Rank Adaptation (LoRA) is deployed. When a user inputs a natural language command, this parser breaks it down into a structured action-target format, translating the textual request into explicit semantic embeddings for targeted object grounding.

### Interactive Scene Editing & Manipulation
The parsed textual embeddings locate and isolate the precise 3D primitives matching the user's intent. Once localized, users can interactively manipulate objects within the 3D environment in real-time through language or a GUI—enabling operations such as highlighting, deleting, scaling, rotating, or moving objects.

### Post-Removal Scene Completion (3D Inpainting)
Simply removing or zeroing out the opacity of a group of Gaussians exposes unconstructed backgrounds and creates empty holes or visual discontinuities in the 3D scene. To remedy this while maintaining visual photorealism, the project introduces a 3D semantic patch-based inpainting workflow. The system extracts the void boundaries, applies RANSAC to estimate the surrounding flat background planes, and patches the hole by copying and geometrically aligning semantically similar 3D Gaussians from known regions. By editing the 3D data directly instead of utilizing 2D neural diffusion generators, multi-view consistency from all camera angles is completely guaranteed by construction.

