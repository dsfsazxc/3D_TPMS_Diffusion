# 3D TPMS Structure Generation using Diffusion Models

This project implements a 3D conditional denoising diffusion probabilistic model (DDPM) to generate triply periodic minimal surface (TPMS)-like structures based on mechanical properties such as Young’s modulus (E) and volume fraction (VF). The generated structures are intended for mechanical optimization and additive manufacturing.

## Course: Advanced Topics in Artificial Intelligence  
Department of Mechanical Design and Robotics Engineering  
- 김의현 (25510065)  
- 지민정 (24510057)

---

## Repository Structure

---

## Project Summary

- **Objective**: Learn a generative model that creates 3D printable structures conditioned on mechanical targets (E, VF).
- **Architecture**: A 3D U-Net-based denoiser using sinusoidal time embeddings and spatially concatenated condition channels.
- **Training**: 1,000-step cosine noise schedule applied only to the structure channel. Conditions (VF, E) are broadcasted and concatenated.
- **Sampling**: Structures are sampled from Gaussian noise and progressively denoised using learned reverse dynamics.
- **Validation**: Sampled structures are voxelized, simulated using FEM, and compared against ground truth. Selected samples are also fabricated using 3D printing.

---

## How to Run

1. **Train the Model**
   - Open `1_diffusion_model_training.ipynb`
   - Configure dataset paths and hyperparameters
   - Execute training loop

2. **Generate Samples**
   - Open `2_3D-TPMS_sample.ipynb`
   - Load trained model weights
   - Specify condition values (VF, E)
   - Run sampling and visualize outputs

---

## Keywords

Diffusion model, 3D structure generation, conditional generation, topology optimization, voxel modeling, TPMS, additive manufacturing

---

## Acknowledgment

This project builds upon and extends the [TopoDiff](https://github.com/francoismaze/topodiff) framework, adapting it to 3D generation with scalar conditioning.
