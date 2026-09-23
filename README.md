# 🤖 ESP32 Micro-LLM Benchmark (`llama2.c`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SoC](https://img.shields.io/badge/SoC-ESP32%20%7C%20ESP32--S3-red.svg)](https://www.espressif.com/)
[![Framework](https://img.shields.io/badge/Model-llama2.c%20(260K)-blue.svg)](https://github.com/karpathy/llama2.c)

Can a $3 microcontroller run a local Generative AI model completely offline? **Yes.**

This repository contains the C/Arduino port and hardware benchmarks for Andrej Karpathy's **260,000-parameter Llama 2 model** (`llama2.c`) running across four different ESP32 development boards. It evaluates the raw memory bandwidth limits of direct SPI Flash streaming versus PSRAM cached execution.

---

## 📊 Benchmark Results

All benchmarks were recorded evaluating a **64-token sequence** using FP32 floating-point precision on a 1.05MB `stories260K.bin` checkpoint.

| Rank | Board Model | Flash Memory | PSRAM | Memory Access Mode | Speed (Tokens/Sec) | Relative Speedup |
| :-: | :--- | :-: | :-: | :--- | :-: | :-: |
| 🥇 | **ESP32-S3 DevKit** | 16 MB | 8 MB | Octal-SPI (OPI) @ 80MHz | **22.00 tok/s** | **9.48x** |
| 🥈 | **Seeed XIAO ESP32-S3** | 8 MB | 8 MB | Octal-SPI (OPI) @ 80MHz | **21.00 tok/s** | **9.05x** |
| 🥉 | **ESP32-CAM** | 4 MB | 4 MB | Quad-SPI (QSPI) @ 40MHz | **11.00 tok/s** | **4.74x** |
| 4 | **Standard ESP32** | 4 MB | None | Direct Flash (`fread`) | **2.32 tok/s** | **1.00x (Baseline)** |

## ⚡ Technical Takeaways

1. **The SPI Flash Bottleneck:** On boards without PSRAM (like the standard ESP32), the model cannot fit into internal SRAM (~320KB). Model parameters are read chunk-by-chunk directly off SPI Flash via filesystem block reads (`fread`). While slow at **2.32 tok/s**, it demonstrates that transformer inference is possible on non-PSRAM hardware.
2. **The PSRAM Speedup:** Pre-loading weights into PSRAM bypasses file system operations completely. Upgrading to **Octal-SPI PSRAM** on the ESP32-S3 achieves **22 tokens/second**—nearly 4x faster than typical human reading speed.

---

## 📁 Repository Structure

```text
├── src/
│   ├── main.cpp            # Main execution loop & context setup
│   ├── run.c               # Refactored C transformer inference engine
│   └── run.h               # Core structures & weight buffers
├── data/
│   ├── stories260K.bin     # 260K Llama model binary (~1.05 MB)
│   └── tok512.bin          # Tokenizer dictionary file
└── platformio.ini          # Pre-configured build environments


