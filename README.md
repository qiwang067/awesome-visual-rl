# Awesome Visual RL [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![GitHub stars](https://img.shields.io/github/stars/qiwang067/awesome-visual-rl)](https://github.com/qiwang067/awesome-visual-rl/stargazers) [![GitHub forks](https://img.shields.io/github/forks/qiwang067/awesome-visual-rl)](https://github.com/qiwang067/awesome-visual-rl/network) <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

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

## 2025
- :large_blue_diamond: **`ICLR 2025 Oral`** [**LS-Imagine**] Open-World Reinforcement Learning over Long Short-Term Imagination [[Paper](https://openreview.net/pdf?id=vzItLaEoDa)] [[Torch Code](https://github.com/qiwang067/LS-Imagine)]
- :large_blue_diamond: **`ICLR 2025`** [**MR.Q**] Towards General-Purpose Model-Free Reinforcement Learning [[Paper](https://openreview.net/pdf/f87ce86b057bcdd9534b1e2b01995b32ae7e84da.pdf)] [[Torch Code](https://github.com/facebookresearch/MRQ)]
- :large_orange_diamond: **`AAAI 2025`** [**ResAct**] Visual Reinforcement Learning with Residual Action [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34097/36252)] [[Torch Code](https://github.com/LiuZhenxian123/ResAct)]
- :large_blue_diamond: **`Nature 2025`** [**DreamerV3**] Mastering Diverse Domains through World Models [[Paper](https://www.nature.com/articles/s41586-025-08744-2.pdf)][[JAX Code](https://github.com/danijar/dreamerv3)][[Torch Code](https://github.com/NM512/dreamerv3-torch)]
## 2024
- :large_orange_diamond: **`ICLR 2024`** Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages [[Paper](https://openreview.net/pdf?id=0aR1s9YxoL)] [[Torch Code](https://github.com/Guozheng-Ma/Adaptive-Replay-Ratio)] 
- :large_blue_diamond: **`ICLR 2024`** [**TD-MPC2**] TD-MPC2: Scalable, Robust World Models for Continuous Control [[Paper](https://arxiv.org/pdf/2310.16828)] [[Torch Code](https://github.com/nicklashansen/tdmpc2)] 
- :large_orange_diamond: **`ICLR 2024 Oral`** [**PTGM**] Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning [[Paper](https://openreview.net/pdf?id=o2IEmeLL9r)] [[Torch Code](https://github.com/PKU-RL/PTGM)]
- :large_orange_diamond: **`ICLR 2024 Spotlight`** [**DrM**] DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization [[Paper](https://arxiv.org/pdf/2310.19668)] [[Torch Code](https://github.com/XuGW-Kevin/DrM)]
- :large_blue_diamond: **`ICLR 2024`** [**DreamSmooth**] DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing [[Paper](https://arxiv.org/pdf/2311.01450)]
- :large_blue_diamond: **`ICLR 2024 Oral`** [**R2I**] Mastering Memory Tasks with World Models [[Paper](http://arxiv.org/pdf/2403.04253)] [[JAX Code](https://github.com/OpenDriveLab/ViDAR)]
- :large_orange_diamond: **`ICLR 2024 Spotlight`** [**PULSE**] Universal Humanoid Motion Representations for Physics-Based Control [[Paper](https://openreview.net/pdf?id=OrOd8PxOO2)] [[Torch Code](https://github.com/ZhengyiLuo/PULSE)]
- :large_blue_diamond: **`ICML 2024 Oral`** [**Dynalang**] Learning to Model the World With Language [[Paper](https://openreview.net/pdf?id=7dP6Yq9Uwv)] [[JAX Code](https://github.com/jlin816/dynalang)]
- :large_blue_diamond: **`ICML 2024`** [**HarmonyDream**] HarmonyDream: Task Harmonization Inside World Models [[Paper](https://arxiv.org/pdf/2310.00344)] [[JAX Code](https://github.com/thuml/HarmonyDream)]
- :large_orange_diamond: **`ICML 2024`** Investigating Pre-Training Objectives for Generalization in Vision-Based Reinforcement Learning [[Paper](https://arxiv.org/pdf/2406.06037)]
- :large_orange_diamond: **`ICML 2024`** [**BeigeMaps**] BeigeMaps: Behavioral Eigenmaps for Reinforcement Learning from Images [[Paper](https://openreview.net/pdf?id=myCgfQZzbc)]
- :large_blue_diamond: **`NeurIPS 2024`** [**CoWorld**] Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning [[Paper](https://openreview.net/pdf?id=ucxQrked0d)] [[Website](https://qiwang067.github.io/coworld)] [[Torch Code](https://github.com/qiwang067/CoWorld)]
- :large_blue_diamond: **`NeurIPS 2024`** [**DIAMOND**] Diffusion for World Modeling: Visual Details Matter in Atari [[Paper](https://arxiv.org/pdf/2405.12399)] [[Torch Code](https://github.com/eloialonso/diamond)]
- :large_blue_diamond: **`NeurIPS 2024`** [**GenRL**] GenRL: Multimodal-foundation world models for generalization in embodied agents [[Paper](https://arxiv.org/pdf/2406.18043)] [[Torch Code](https://github.com/mazpie/genrl)]
- :large_orange_diamond: **`NeurIPS 2024`** [**CP3ER**] CP3ER: Generalizing Consistency Policy to Visual RL with Prioritized Proximal Experience Regularization
 [[Paper](https://arxiv.org/pdf/2410.00051)] [[Torch Code](https://github.com/jzndd/CP3ER)]
- :large_orange_diamond: **`RLC 2024`** [**SADA**] A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2405.17416)][[Torch Code](https://github.com/aalmuzairee/dmcgb2)]
- :large_blue_diamond: **`IEEE IOT`** [**CarDreamer**] CarDreamer: Open-Source Learning Platform for World Model based Autonomous Driving [[Paper](https://arxiv.org/pdf/2405.09111)] [[Code](https://github.com/ucd-dare/CarDreamer)]
- :large_orange_diamond: **`arXiv 2024.10`** [**MENTOR**] MENTOR: Mixture-of-Experts Network with Task-Oriented Perturbation for Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2410.14972)] 
- :large_blue_diamond: **`arXiv 2024.6`** [**DLLM**] World Models with Hints of Large Language Models for Goal Achieving [[Paper](http://arxiv.org/pdf/2406.07381)]
- :large_blue_diamond: **`arXiv 2024.5`** [**Puppeteer**] Hierarchical World Models as Visual Whole-Body Humanoid Controllers [[Paper](https://arxiv.org/pdf/2405.18418)] [[Torch Code](https://github.com/nicklashansen/puppeteer)]

## 2023
- :large_orange_diamond: **`ICLR 2023`** [**CoIT**] On the Data-Efficiency with Contrastive Image Transformation in Reinforcement Learning [[Paper](https://openreview.net/forum?id=-nm-rHXi5ga)] [[Torch Code](https://github.com/Kamituna/CoIT)]
- :large_blue_diamond: **`ICLR 2023`** [**MoDem**] MoDem: Accelerating Visual Model-Based Reinforcement Learning with Demonstrations [[Paper](https://openreview.net/pdf?id=JdTnc9gjVfJ)] [[Torch Code](https://github.com/facebookresearch/modem)]
- :large_orange_diamond: **`ICLR 2023`** [**TED**] Temporal Disentanglement of Representations for Improved Generalisation in Reinforcement Learning [[Paper](https://openreview.net/pdf?id=sPgP6aISLTD)] [[Torch Code](https://github.com/uoe-agents/TED)]
- :large_orange_diamond: **`ICLR 2023 Spotlight`** [**VIP**] VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training [[Paper](https://openreview.net/pdf?id=YJ7o2wetJ2)] [[Torch Code](https://github.com/facebookresearch/vip)]
- :large_blue_diamond: **`ICML 2023 Oral`** Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels [[Paper](https://openreview.net/attachment?id=eSpbTG0TZN&name=pdf)]
- :large_orange_diamond: **`ICML 2023`** On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline [[Paper](https://proceedings.mlr.press/v202/hansen23c/hansen23c.pdf)] [[Torch Code](https://github.com/gemcollector/learning-from-scratch?tab=readme-ov-file)]
- :large_orange_diamond: **`ICCV 2023`** [**CG2A**] Improving Generalization in Visual Reinforcement Learning via Conflict-aware Gradient Agreement Augmentation [[Paper](https://arxiv.org/abs/2308.01194)]
- :large_blue_diamond: **`NeurIPS 2023`** [**STORM**] STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/5647763d4245b23e6a1cb0a8947b38c9-Paper-Conference.pdf)][[Torch Code](https://github.com/weipu-zhang/STORM)]
- :large_orange_diamond: **`NeurIPS 2023`** [**HAVE**] Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning [[Paper](https://NeurIPS.cc/virtual/2023/poster/70701)][[Torch Code](https://github.com/Yara-HYR/HAVE)]
- :large_orange_diamond: **`NeurIPS 2023`** [**PIE-G**] Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning [[Paper](https://NeurIPS.cc/virtual/2023/poster/70701)][[Torch Code](https://github.com/Yara-HYR/HAVE)]
- :large_orange_diamond: **`NeurIPS 2023`** Learning Better with Less: Effective Augmentation for Sample-Efficient Visual Reinforcement Learning [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/bc26087d3f82e62044fc77752e86737e-Paper-Conference.pdf)][[Torch Code](https://github.com/Guozheng-Ma/CycAug)]
- :large_orange_diamond: **`NeurIPS 2023`** [**TACO**] TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2306.13229)][[Torch Code](https://github.com/frankzheng2022/taco)]
- :large_orange_diamond: **`NeurIPS 2023`** [**CMID**] Conditional Mutual Information for Disentangled Representations in Reinforcement Learning [[Paper](https://arxiv.org/pdf/2305.14133)][[Torch Code](https://github.com/uoe-agents/cmid)]

## 2022
- :large_orange_diamond: **`ICLR 2022`** [**DrQ-v2**] Local Feature Swapping for Generalization in Reinforcement Learning [[Paper](https://arxiv.org/pdf/2107.09645)][[Torch Code](https://github.com/facebookresearch/drqv2)]
- :large_orange_diamond: **`ICLR 2022`** [**CLOP**] Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning [[Paper](https://openreview.net/forum?id=Sq0-tgDyHe4)][[Torch Code](https://github.com/DavidBert/CLOP)]
- :large_blue_diamond: **`ICML 2022`** [**TD-MPC**] Temporal Difference Learning for Model Predictive Control [[Paper](https://arxiv.org/pdf/2203.04955)][[Torch Code](https://github.com/nicklashansen/tdmpc)]
- :large_orange_diamond: **`ICML 2022`** [**DRIBO**] DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck [[Paper](https://proceedings.mlr.press/v162/fan22b.html)][[Torch Code](https://github.com/BU-DEPEND-Lab/DRIBO)]
- :large_blue_diamond: **`ICML 2022`** [**DreamerPro**] DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations [[Paper](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf)][[TF Code](https://github.com/fdeng18/dreamer-pro)]
- :large_orange_diamond: **`IJCAI 2022`** [**CCLF**] CCLF: A Contrastive-Curiosity-Driven Learning Framework for Sample-Efficient Reinforcement Learning [[Paper](https://arxiv.org/abs/2205.00943)]
- :large_orange_diamond: **`IJCAI 2022`** [**TLDA**] Don’t Touch What Matters: Task-Aware Lipschitz Data Augmentation for Visual Reinforcement Learning [[Paper](https://arxiv.org/abs/2202.09982)][[Torch Code](https://github.com/gemcollector/TLDA)]
- :large_orange_diamond: **`NeurIPS 2022`** [**PIE-G**] Pre-Trained Image Encoder for Generalizable Visual Reinforcement Learning [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/548a482d4496ce109cddfbeae5defa7d-Paper-Conference.pdf)][[Torch Code](https://github.com/gemcollector/PIE-G)]
- :large_orange_diamond: **`NeurIPS 2022`** Efficient Scheduling of Data Augmentation for Deep Reinforcement Learning [[Paper](https://arxiv.org/abs/2102.08581)][[Torch Code](https://github.com/kbc-6723/es-da)]
- :large_orange_diamond: **`NeurIPS 2022`** Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels? [[Paper](https://arxiv.org/abs/2206.05266)]
- :large_orange_diamond: **`NeurIPS 2022`** [**A2LS**] Reinforcement Learning with Automated Auxiliary Loss Search [[Paper](https://arxiv.org/abs/2210.06041)][[Torch Code](https://seqml.github.io/a2ls/)]
- :large_orange_diamond: **`NeurIPS 2022`** [**MLR**] Mask-based Latent Reconstruction for Reinforcement Learning [[Paper](https://arxiv.org/abs/2201.12096)][[Torch Code](https://github.com/microsoft/Mask-based-Latent-Reconstruction)]
- :large_orange_diamond: **`NeurIPS 2022`** [**SRM**] Spectrum Random Masking for Generalization in Image-based Reinforcement Learning [[Paper](https://openreview.net/forum?id=m16lH6XJsbb)][[Torch Code](https://github.com/Yara-HYR/SRM)]
- :large_blue_diamond: **`NeurIPS 2022`**  Deep Hierarchical Planning from Pixels.  [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a766f56d2da42cae20b5652970ec04ef-Paper-Conference.pdf)][[TF Code](https://github.com/danijar/director)]
- :large_blue_diamond: **`NeurIPS 2022 Spotlight`** [**Iso-Dream**] Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9316769afaaeeaad42a9e3633b14e801-Paper-Conference.pdf)][[Torch Code](https://github.com/panmt/Iso-Dream)]
- :large_orange_diamond: **`TPAMI 2022`** [**M-CURL**] Masked Contrastive Representation Learning for Reinforcement Learning [[Paper](https://ieeexplore.ieee.org/abstract/document/9779589)]
- :large_blue_diamond: **`CoRL 2022`** [**DayDreamer**] DayDreamer: World Models for Physical Robot Learning [[Paper](https://proceedings.mlr.press/v205/wu23c/wu23c.pdf)] [[TF Code](https://github.com/danijar/daydreamer)]
- :large_blue_diamond: **`arXiv 2022.3`** [**DreamingV2**] DreamingV2: Reinforcement Learning with Discrete World Models without Reconstruction [[Paper](https://arxiv.org/pdf/2203.00494)] 



## 2021
- :large_orange_diamond: **`ICLR 2021 Spotlight`** [**DrQ**] Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels  [[Paper](https://arxiv.org/pdf/2004.13649)][[Torch Code](https://github.com/denisyarats/drq)]
- :large_orange_diamond: **`ICLR 2021`** [**MixStyle**] Domain Generalization with MixStyle [[Paper](https://openreview.net/forum?id=6xHJ37MVxxp)][[Torch Code](https://github.com/KaiyangZhou/mixstyle-release)]
- :large_orange_diamond: **`ICLR 2021`** [**SPR**] Data-Efficient Reinforcement Learning with Self-Predictive Representations [[Paper](https://openreview.net/forum?id=uCQfPZwRaUu&fbclid=IwAR3FMvlynXXYEMJaJzPki1x1wC9jjA3aBDC_moWxrI91hLaDvtk7nnnIXT8)][[Torch Code](https://github.com/mila-iqia/spr)]
- :large_blue_diamond: **`ICLR 2021`** [**DreamerV2**] Mastering Atari with Discrete World Models [[Paper](https://arxiv.org/pdf/2010.02193)][[TF Code](https://github.com/danijar/dreamerv2)][[Torch Code](https://github.com/jsikyoon/dreamer-torch)]
- :large_orange_diamond: **`ICML 2021`** [**SECANT**] Self-Expert Cloning for Zero-Shot Generalization of Visual Policies [[Paper](https://proceedings.mlr.press/v139/fan21c.html)] [[Torch Code](https://github.com/LinxiFan/SECANT)]
- :large_orange_diamond: **`NeurIPS 2021`** [**PlayVirtual**] Augmenting Cycle-Consistent Virtual Trajectories for Reinforcement Learning [[Paper](https://proceedings.neurips.cc/paper/2021/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html)][[Torch Code](https://github.com/microsoft/Playvirtual)]
- :large_orange_diamond: **`NeurIPS 2021`** [**EXPAND**] Widening the Pipeline in Human-Guided Reinforcement Learning with Explanation and Context-Aware Data Augmentation [[Paper](https://proceedings.neurips.cc/paper/2021/hash/b6f8dc086b2d60c5856e4ff517060392-Abstract.html)]
- :large_orange_diamond: **`NeurIPS 2021`** [**SVEA**] Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation [[Paper](https://proceedings.neurips.cc/paper/2021/hash/1e0f65eb20acbfb27ee05ddc000b50ec-Abstract.html)] [[Torch Code](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)]
- :large_orange_diamond: **`NeurIPS 2021`** [**UCB-DrAC**] Automatic Data Augmentation for Generalization in Reinforcement Learning [[Paper](https://proceedings.neurips.cc/paper/2021/hash/2b38c2df6a49b97f706ec9148ce48d86-Abstract.html)] [[Torch Code](https://github.com/rraileanu/auto-drac)]
- :large_blue_diamond: **`ICRA 2021`** [**Dreaming**] Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction [[Paper](https://arxiv.org/pdf/2007.14535)]
- :large_blue_diamond: **`CVPR 2021`** [**VAI**] Unsupervised Visual Attention and Invariance for Reinforcement Learning [[Paper](https://arxiv.org/pdf/2104.02921)] [[Torch Code](https://github.com/TonyLianLong/VAI-ReinforcementLearning)]

## 2020
- :large_blue_diamond: **`ICML 2020`** [**Plan2Explore**] Planning to Explore via Self-Supervised World Models [[Paper](https://arxiv.org/pdf/2005.05960)][[TF Code](https://github.com/ramanans1/plan2explore)][[Torch Code](https://github.com/yusukeurakami/plan2explore-pytorch)]
- :large_orange_diamond: **`ICML 2020`** [**CURL**] CURL: Contrastive Unsupervised Representations for Reinforcement Learning [[Paper](https://arxiv.org/pdf/2004.04136)] [[Torch Code](https://github.com/MishaLaskin/curl)]
- :large_blue_diamond: **`ICLR 2020`** [**DreamerV1**] Dream to Control: Learning Behaviors by Latent Imagination [[Paper](https://arxiv.org/pdf/1912.01603)][[TF Code](https://github.com/danijar/dreamer)][[Torch Code](https://github.com/juliusfrost/dreamer-pytorch)]

## 2018
- :large_blue_diamond: **`NeurIPS 2018 Oral`** World Models [[Paper](https://arxiv.org/pdf/1803.10122)]

## Other Vision-Related Reinforcement Learning Papers
## 2024
- :large_blue_diamond: **`ICLR 2024 Oral`** Predictive auxiliary objectives in deep RL mimic learning in the brain [[Paper](https://openreview.net/pdf?id=agPpmEgf8C)]
- :large_orange_diamond: **`ICLR 2024 Oral`** [**METRA**] METRA: Scalable Unsupervised RL with Metric-Aware Abstraction [[Paper](https://openreview.net/pdf?id=c5pwL0Soay)] [[Torch Code](https://seohong.me/projects/metra/)]
- :large_orange_diamond: **`ICLR 2024 Spotlight`** Selective Visual Representations Improve Convergence and Generalization for Embodied AI [[Paper](https://openreview.net/pdf?id=kC5nZDU5zf)] [[Torch Code](https://github.com/allenai/procthor-rl)]
- :large_orange_diamond: **`ICLR 2024 Spotlight`** Towards Principled Representation Learning from Videos for Reinforcement Learning [[Paper](https://openreview.net/pdf?id=3mnWvUZIXt)] [[Torch Code](https://github.com/microsoft/Intrepid)]
  
## Technical Blog
- [Can RL From Pixels be as Efficient as RL From State?](https://bair.berkeley.edu/blog/2020/07/19/curl-rad/)

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation


```bibtex
@inproceedings{wang2024making,
  title={Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning}, 
  author={Qi Wang and Junming Yang and Yunbo Wang and Xin Jin and Wenjun Zeng and Xiaokang Yang},
  booktitle={NeurIPS},
  year={2024}
}
```
```bibtex
@article{ma2025comprehensive,
  title={A comprehensive survey of data augmentation in visual reinforcement learning},
  author={Ma, Guozheng and Wang, Zhen and Yuan, Zhecheng and Wang, Xueqian and Yuan, Bo and Tao, Dacheng},
  journal={IJCV},
  pages={1--38},
  year={2025}
}
```

```bibtex
@inproceedings{li2025open,
    title={Open-World Reinforcement Learning over Long Short-Term Imagination}, 
    author={Jiajian Li and Qi Wang and Yunbo Wang and Xin Jin and Yang Li and Wenjun Zeng and Xiaokang Yang},
    booktitle={ICLR},
    year={2025}
}
```




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
         <p>Nanyang Technological University</p>
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

