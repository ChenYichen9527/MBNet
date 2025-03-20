# Event-based Motion Deblurring with Blur-aware Reconstruction Filter

<p align="center">
  <img src="C:\Users\90597\Desktop\MBNet\第二次审稿后修改\TCSVT_GITHUB\MBNet\imgs/left_down_1.jpg" alt="Transafer" width="400"/>
 </a>
</p>

This repository contains the official implementation of the paper [**"Event-based Motion Deblurring with Blur-aware Reconstruction Filter"**](https://ieeexplore.ieee.org/abstract/document/10926552), accepted at **TCSVT 2025**. The project focuses on deblurring motion-blurred images using event-based cameras and a novel blur-aware reconstruction filter.

---



## 🌟 Abstract
Event-based motion deblurring aims at reconstructing a sharp image from a single blurry image and its corresponding events triggered during the exposure time. Existing methods learn the spatial distribution of blur from blurred images, then treat events as temporal residuals and learn blurred temporal features from them, and finally restore clear images through spatio-temporal interaction of the two features. However, due to the high coupling of detailed features such as the texture and contour of the scene with blur features, it is difficult to directly learn effective blur spatial distribution from the original blurred image. In this paper, we provide a novel perspective, i.e., employing the blur indication provided by events, to instruct the network in spatially differentiated image reconstruction. Due to the consistency between event spatial distribution and image blur, event spatial indication can learn blur spatial features more simply and directly, and serve as a complement to temporal residual guidance to improve deblurring performance. Based on the above insight, we propose an event-based motion deblurring network consisting of a Multi-Scale Event-based Double Integral  (MS-EDI) module designed from temporal residual guidance, and a Blur-Aware Filter Prediction (BAFP) module to conduct filter processing directed by spatial blur indication. 
The network, after incorporating spatial residual guidance, has significantly enhanced its generalization ability, surpassing the best-performing image-based and event-based methods on both synthetic, semi-synthetic, and real-world datasets.
In addition, our method can be extended to blurry image super-resolution and achieves impressive performance.


---

## 🚀 Usage

### Train
	python training.py

### Test
	python testing.py
---
  ## 📊 Visualization results
<p align="center">
  <img src="C:\Users\90597\Desktop\MBNet\第二次审稿后修改\TCSVT_GITHUB\MBNet\imgs/SOTA_gopro.jpg" alt="Transafer" width="600"/>
 </a>
</p>

## 📝 Citation
If you use any of this code, please cite the following publication:

```bibtex
@ARTICLE{10926552,
  author={Chen, Nuo and Zhang, Chushu and An, Wei and Wang, Longgaung and Li, Miao and Ling, Qiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Event-based Motion Deblurring with Blur-aware Reconstruction Filter}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Event detection;Image reconstruction;Cameras;Graphical models;Distribution functions;Brightness;Training;Superresolution;Electronic mail;Data mining;Motion deblurring;Event-based vision;Super-resolution},
  doi={10.1109/TCSVT.2025.3551516}}
```
