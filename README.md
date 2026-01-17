# NTU Parallel Programming - Homework + Final Project

本資料夾收錄 NTU 平行程式設計課程的全部作業與期末專題，以下提供每個資料夾的內容重點，方便快速了解有哪些內容。  
This folder contains all homework submissions and the final project for NTU Parallel Programming. The list below summarizes each folder so others can quickly understand what is included.

## 專題與作業主題總覽 / Topic Overview

Pararrel programing (HW1-5, Final)
- HW1: Sokoban solver - C++ code, A* search with OpenMP parallelization
- HW2: SIFT Algorithm Parallelization with Hybrid MPI + OpenMP
- HW3: Mandelbulb Ray Marching CUDA Optimization
- HW4: GPU Miner Optimization
- HW5: N-body Simulation GPU parallel
- Final: Accelerating Rejection Sampling for Speculative Decoding

平行程式設計（作業一～五、期末專題）
- 作業一：Sokoban 解題器（C++，A* 搜尋 + OpenMP 平行化）
- 作業二：SIFT 演算法平行化（MPI + OpenMP 混合）
- 作業三：Mandelbulb Ray Marching CUDA 最佳化
- 作業四：GPU Miner 最佳化
- 作業五：N-body Simulation GPU 平行化
- 期末專題：Speculative Decoding 的 Rejection Sampling 加速

## 作業 / Homework

### `homework1/`
- 中文：作業一（C++）。主要檔案為 `hw1.cpp`，使用 `Makefile` 編譯，成果與說明在 `report.pdf`。  
- EN: Homework 1 (C++). Main file is `hw1.cpp`, build with `Makefile`, write-up in `report.pdf`.

### `homework2/`
- 中文：作業二（C++），包含影像處理與 SIFT 相關實作。主要檔案有 `hw2.cpp`、`sift.cpp`、`image.cpp` 與對應標頭檔，使用 `Makefile` 編譯，報告在 `report.pdf`。  
- EN: Homework 2 (C++), includes image processing and SIFT-related implementation. Main files are `hw2.cpp`, `sift.cpp`, `image.cpp` with headers; build via `Makefile`, report in `report.pdf`.

### `homework3/`
- 中文：作業三（CUDA）。主要檔案為 `hw3.cu`，使用 `Makefile` 編譯，報告在 `report.pdf`。  
- EN: Homework 3 (CUDA). Main file `hw3.cu`, build with `Makefile`, report in `report.pdf`.

### `homework4/`
- 中文：作業四（CUDA），包含 SHA-256 相關實作。主要檔案為 `hw4.cu`、`sha256.cu` 與 `sha256.h`，使用 `Makefile` 編譯，報告在 `report.pdf`。  
- EN: Homework 4 (CUDA) with SHA-256 related implementation. Main files `hw4.cu`, `sha256.cu`, `sha256.h`; build with `Makefile`, report in `report.pdf`.

### `homework5/`
- 中文：作業五（C++）。主要檔案為 `hw5.cpp`，使用 `Makefile` 編譯，報告在 `report.pdf`。  
- EN: Homework 5 (C++). Main file `hw5.cpp`, build with `Makefile`, report in `report.pdf`.

### `homeworrk1 bonus/`
- 中文：作業一加分題，包含 `hw1-bonus.sh` 腳本、`README.md` 說明與 `report.pdf`。  
- EN: HW1 bonus folder. Includes `hw1-bonus.sh`, a `README.md`, and `report.pdf`.

## 期末專題 / Final Project

### `final project/`
- 中文：期末專題，主題是 speculative decoding 的 CUDA fused rejection sampling 加速。內容包含：
  - CUDA extension（C++/CUDA + PyTorch 綁定）與 V1/V2 kernel
  - Python baseline 與 torch.compile 版本對照
  - Benchmark 與效能報告產生腳本
  - Demo（本機與真實 LLM 兩種）與詳細說明文件
  - 技術報告與實作指南

- EN: Final project on CUDA fused rejection sampling for speculative decoding. Includes:
  - CUDA extension (C++/CUDA + PyTorch binding) with V1/V2 kernels
  - Python baseline and torch.compile versions for comparison
  - Benchmark scripts and report generator
  - Demos (local + real LLM) with documentation
  - Technical report and implementation guide

- 專題入口 / Entry point: `final project/README.md`
- 簡報與報告 / Slides & report:
  - `final project/final project presentation slides.pptx`
  - `final project/final project report.pdf`

## 使用方式 / How to use

- 中文：每個作業資料夾均包含 `Makefile` 與 `report.pdf`，可依作業說明進行編譯或執行。  
- EN: Each homework folder includes a `Makefile` and `report.pdf` for build/run instructions.

- 中文：期末專題包含 CUDA extension 與完整文件，請先閱讀 `final project/README.md`。  
- EN: The final project includes a CUDA extension and detailed documentation; start with `final project/README.md`.

## 備註 / Notes

- 中文：資料夾名稱保持原提交格式（含 `homeworrk1 bonus` 的拼字）。  
- EN: Folder names are kept as submitted (including the `homeworrk1 bonus` spelling).
