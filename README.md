# Awesome Visual RL [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![GitHub stars](https://img.shields.io/github/stars/qiwang067/awesome-visual-rl)](https://github.com/qiwang067/awesome-visual-rl/stargazers) [![GitHub forks](https://img.shields.io/github/forks/qiwang067/awesome-visual-rl)](https://github.com/qiwang067/awesome-visual-rl/network) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fqiwang067%2Fawesome-visual-rl%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

This is a collection of research papers on **Visual Reinforcement Learning (Visual RL)** and other vision-related reinforcement learning.

If you find some ignored papers, **feel free to [*open issues*](https://github.com/qiwang067/awesome-visual-rl/issues/new), or [*email* Qi Wang](mailto:qiwang067@163.com) / [GuoZheng Ma](mailto:guozheng_ma@163.com) / [Yuan Pu](mailto:puyuan1996@qq.com)**. Contributions in any form to make this list more comprehensive are welcome. 📣📣📣

If you find this repository useful, please consider **giving us a star** 🌟. 

<!--**[citing](#citation)** and-->

Feel free to share this list with others! 🥳🥳🥳

<!-- ## Workshop & Challenge

- [`CVPR 2024 Workshop & Challenge | OpenDriveLab`](https://opendrivelab.com/challenge2024/#predictive_world_model) Track #4: Predictive World Model.
  > Serving as an abstract spatio-temporal representation of reality, the world model can predict future states based on the current state. The learning process of world models has the potential to elevate a pre-trained foundation model to the next level. Given vision-only inputs, the neural network outputs point clouds in the future to testify its predictive capability of the world.
  
- [`CVPR 2023 Workshop on Autonomous Driving`](https://cvpr2023.wad.vision/) CHALLENGE 3: ARGOVERSE CHALLENGES, [3D Occupancy Forecasting](https://eval.ai/web/challenges/challenge-page/1977/overview) using the [Argoverse 2 Sensor Dataset](https://www.argoverse.org/av2.html#sensor-link). Predict the spacetime occupancy of the world for the next 3 seconds. -->

## Papers
```
format:
- publisher **[abbreviation of proposed model]** title [paper link] [code link]
```
:large_blue_diamond: Model-Based &nbsp; :large_orange_diamond: Model-Free

## 2024
- :large_blue_diamond: **`ICML 2024 Oral`** [**Dynalang**] Learning to Model the World With Language [[Paper](https://openreview.net/pdf?id=7dP6Yq9Uwv)] [[JAX Code](https://github.com/jlin816/dynalang)]
- :large_blue_diamond: **`ICLR 2024`** [**TD-MPC2**] TD-MPC2: Scalable, Robust World Models for Continuous Control [[Paper](https://arxiv.org/pdf/2310.16828)] [[Torch Code](https://github.com/nicklashansen/tdmpc2)] 
- :large_orange_diamond: **`ICLR 2024`** [**DrM**] DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization [[Paper](https://arxiv.org/pdf/2310.19668)] 
- :large_orange_diamond: **`ICLR 2024 Oral`** [**PTGM**] Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning [[Paper](https://openreview.net/pdf?id=o2IEmeLL9r)] [[Torch Code](https://github.com/PKU-RL/PTGM)]
- :large_blue_diamond: **`ICLR 2024`** [**DreamSmooth**] DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing [[Paper](https://arxiv.org/pdf/2311.01450)]
- :large_blue_diamond: **`ICLR 2024 Oral`** [**R2I**] Mastering Memory Tasks with World Models [[Paper](http://arxiv.org/pdf/2403.04253)] [[JAX Code](https://github.com/OpenDriveLab/ViDAR)]
- :large_orange_diamond: **`ICLR 2024 Spotlight`** [**PULSE**] Universal Humanoid Motion Representations for Physics-Based Control [[Paper](https://openreview.net/pdf?id=OrOd8PxOO2)] [[Torch Code](https://github.com/ZhengyiLuo/PULSE)]
- :large_orange_diamond: **`RLC 2024`** [**SADA**] A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2405.17416)][[Torch Code](https://github.com/aalmuzairee/dmcgb2)]
- :large_blue_diamond: **`arXiv 2024.5`** [**Puppeteer**] Hierarchical World Models as Visual Whole-Body Humanoid Controllers [[Paper](https://arxiv.org/pdf/2405.18418)] [[Torch Code](https://github.com/nicklashansen/puppeteer)]

## 2023

- :large_orange_diamond: **`NeurIPS 2023`** [**HAVE**] Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning [[Paper](https://NeurIPS.cc/virtual/2023/poster/70701)][[Torch Code](https://github.com/Yara-HYR/HAVE)]
- :large_orange_diamond: **`NeurIPS 2023`** [**TACO**] TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2306.13229)][[Torch Code](https://github.com/frankzheng2022/taco)]
- :large_orange_diamond: **`NeurIPS 2023`** [**CMID**] Conditional Mutual Information for Disentangled Representations in Reinforcement Learning [[Paper](https://arxiv.org/pdf/2305.14133)][[Torch Code](https://github.com/uoe-agents/cmid)]
- :large_orange_diamond: **`ICLR 2023`** [**CoIT**] On the Data-Efficiency with Contrastive Image Transformation in Reinforcement Learning [[Paper](https://openreview.net/forum?id=-nm-rHXi5ga)] [[Torch Code](https://github.com/Kamituna/CoIT)]
- :large_orange_diamond: **`ICCV 2023`** [**CG2A**] Improving Generalization in Visual Reinforcement Learning via Conflict-aware Gradient Agreement Augmentation [[Paper](https://arxiv.org/abs/2308.01194)]
- :large_blue_diamond: **`arXiv 2023.5`** [**CoWorld**] Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2305.15260)]
- :large_blue_diamond: **`arXiv 2023.1`** [**DreamerV3**] Mastering Atari with Discrete World Models [[Paper](https://arxiv.org/pdf/2301.04104)][[JAX Code](https://github.com/danijar/dreamerv3)][[Torch Code](https://github.com/NM512/dreamerv3-torch)]

## 2022

- :large_blue_diamond: **`ICML 2022`** [**TD-MPC**] Temporal Difference Learning for Model Predictive Control [[Paper](https://arxiv.org/pdf/2203.04955)][[Torch Code](https://github.com/nicklashansen/tdmpc)]
- :large_orange_diamond: **`ICML 2022`** [**DRIBO**] DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck [[Paper](https://proceedings.mlr.press/v162/fan22b.html)][[Torch Code](https://github.com/BU-DEPEND-Lab/DRIBO)]
- :large_blue_diamond: **`ICML 2022`** [**DreamerPro**] DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations [[Paper](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf)][[TF Code](https://github.com/fdeng18/dreamer-pro)]
- :large_blue_diamond: **`NeurIPS 2022`**  Deep Hierarchical Planning from Pixels.  [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a766f56d2da42cae20b5652970ec04ef-Paper-Conference.pdf)][[TF Code](https://github.com/danijar/director)]
- :large_blue_diamond: **`NeurIPS 2022 Spotlight`** [**Iso-Dream**] Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9316769afaaeeaad42a9e3633b14e801-Paper-Conference.pdf)][[Torch Code](https://github.com/panmt/Iso-Dream)]
- :large_blue_diamond: **`CoRL 2022`** [**DayDreamer**] DayDreamer: World Models for Physical Robot Learning [[Paper](https://proceedings.mlr.press/v205/wu23c/wu23c.pdf)] [[TF Code](https://github.com/danijar/daydreamer)]
- :large_orange_diamond: **`ICLR 2022`** [**DrQ-v2**] Local Feature Swapping for Generalization in Reinforcement Learning [[Paper](https://arxiv.org/pdf/2107.09645)][[Torch Code](https://github.com/facebookresearch/drqv2)]
- :large_orange_diamond: **`ICLR 2022`** [**CLOP**] Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning [[Paper](https://openreview.net/forum?id=Sq0-tgDyHe4)][[Torch Code](https://github.com/DavidBert/CLOP)]


## 2021
- :large_orange_diamond: **`ICLR 2021 Spotlight`** [**DrQ**] Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels  [[Paper](https://arxiv.org/pdf/2004.13649)][[Torch Code](https://github.com/denisyarats/drq)]
- :large_blue_diamond: **`ICLR 2021`** [**DreamerV2**] Mastering Atari with Discrete World Models [[Paper](https://arxiv.org/pdf/2010.02193)][[TF Code](https://github.com/danijar/dreamerv2)][[Torch Code](https://github.com/jsikyoon/dreamer-torch)]

## 2020
- :large_blue_diamond: **`ICML 2020`** [**Plan2Explore**] Planning to Explore via Self-Supervised World Models [[Paper](https://arxiv.org/pdf/2005.05960)][[TF Code](https://github.com/ramanans1/plan2explore)][[Torch Code](https://github.com/yusukeurakami/plan2explore-pytorch)]
- :large_orange_diamond: **`ICML 2020`** [**CURL**] CURL: Contrastive Unsupervised Representations for Reinforcement Learning [[Paper](https://arxiv.org/pdf/2004.04136)] [[Torch Code](https://github.com/MishaLaskin/curl)]
- :large_blue_diamond: **`ICLR 2020`** [**DreamerV1**] Dream to Control: Learning Behaviors by Latent Imagination [[Paper](https://arxiv.org/pdf/1912.01603)][[TF Code](https://github.com/danijar/dreamer)][[Torch Code](https://github.com/juliusfrost/dreamer-pytorch)]

## 2018
- :large_blue_diamond: **`NeurIPS 2018 Oral`** World Models [[Paper](https://arxiv.org/pdf/1803.10122)]

## Other Vision-Related Reinforcement Learning Papers
- :large_blue_diamond: **`ICLR 2024 Oral`** Predictive auxiliary objectives in deep RL mimic learning in the brain [[Paper](https://openreview.net/pdf?id=agPpmEgf8C)]
- :large_orange_diamond: **`ICLR 2024 Oral`** [**METRA**] METRA: Scalable Unsupervised RL with Metric-Aware Abstraction [[Paper](https://openreview.net/pdf?id=c5pwL0Soay)] [[Torch Code](https://seohong.me/projects/metra/)]
- :large_orange_diamond: **`ICLR 2024 Spotlight`** Selective Visual Representations Improve Convergence and Generalization for Embodied AI [[Paper](https://openreview.net/pdf?id=kC5nZDU5zf)] [[Torch Code](https://github.com/allenai/procthor-rl)]
- :large_orange_diamond: **`ICLR 2024 Spotlight`** Towards Principled Representation Learning from Videos for Reinforcement Learning [[Paper](https://openreview.net/pdf?id=3mnWvUZIXt)] [[Torch Code](https://github.com/microsoft/Intrepid)]
## Contributors

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://github.com/qiwang067"><img width="70" height="70" src="https://github.com/qiwang067.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/qiwang067">Qi Wang</a> 
        <p> Shanghai Jiao Tong University </p>
      </td>
      <td>
         <a href="https://github.com/Guozheng-Ma"><img width="70" height="70" src="https://github.com/Guozheng-Ma.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/Guozheng-Ma">GuoZheng Ma</a>
         <p>Tsinghua University</p>
      </td>
      <td>
         <a href="https://github.com/puyuan1996"><img width="70" height="70" src="https://github.com/puyuan1996.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/puyuan1996">Yuan Pu</a><br>
        <a href="https://github.com/opendilab">Shanghai Artificial Intelligence Laboratory (OpenDILab)</a>
      </td>
    </tr>
  </tbody>
</table>

<p align="right">(<a href="#top">Back to top</a>)</p>

