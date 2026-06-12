<h1 align="center">MVOFormer: Flow-Semantic Transformer for Robust Monocular Visual Odometry</h1>

<p align="center">
  <strong>Accepted by IEEE Robotics and Automation Letters (RA-L)</strong>
</p>

<p align="center">
  <a href="https://github.com/Sun-Shun/MVOFormer">
    <img src="https://img.shields.io/badge/status-accepted-brightgreen" alt="Status">
  </a>
</p>

---

## 📖 Abstract

Monocular visual odometry (MVO) is foundational to autonomous navigation and robotic localization. However, existing learning-based MVO approaches often struggle with either a lack of interpretable, complementary features or overly complex multi-stage architectures. These limitations inherently restrict their robustness and cross-domain generalization.

In this work, we propose **MVOFormer**, a novel transformer framework for robust monocular visual odometry. Our architecture features a **Flow-Semantic Dual Branch Encoder** that synergizes dense geometric motion cues with object-centric semantic priors, explicitly distinguishing static structures from dynamic distractors. These representations are then fused by an **Iterative Multimodal Decoder**, enabling coarse-to-fine pose refinement while dynamically suppressing attention on unreliable regions.

Extensive evaluations demonstrate that, without any target-domain fine-tuning, MVOFormer achieves superior zero-shot generalization and robustness, significantly outperforming prior learning-based frame-to-frame methods across diverse benchmarks including TartanAir, KITTI, TUM-RGBD, and ETH3D-SLAM.

---

## 🚀 Code Coming Soon

The full source code, including training and evaluation scripts, will be released. Stay tuned!

---

## 📄 Citation

If this project is useful for your research, please consider citing:

```bibtex
@article{sun2026mvoformer,
  title     = {MVOFormer: Flow-Semantic Transformer for Robust Monocular Visual Odometry},
  author    = {Sun, Shunwang and others},
  journal   = {IEEE Robotics and Automation Letters (RA-L)},
  year      = {2026}
}
```

---

## 📄 License

This project is licensed under the **BSD 3-Clause License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026, Zhejiang University
