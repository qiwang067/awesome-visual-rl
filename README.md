# Awesome Visual RL [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![GitHub stars](https://img.shields.io/github/stars/qiwang067/awesome-visual-rl)](https://github.com/qiwang067/awesome-visual-rl/stargazers) [![GitHub forks](https://img.shields.io/github/forks/qiwang067/awesome-visual-rl)](https://github.com/qiwang067/awesome-visual-rl/network) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fqiwang067%2Fawesome-visual-rl%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

Collect some Visual Reinforcement Learning papers. 

If you find some ignored papers, **feel free to [*open issues*](https://github.com/qiwang067/awesome-visual-rl/issues/new), or [*email* Qi Wang](mailto:qiwang067@163.com) / [GuoZheng Ma](mailto:guozheng_ma@163.com)**. Contributions in any form to make this list more comprehensive are welcome. 📣📣📣

If you find this repository useful, please consider **[citing](#citation)** and **giving us a star** 🌟. 

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
- :large_blue_diamond: **`arXiv 2024.5`** [**Puppeteer**] Hierarchical World Models as Visual Whole-Body Humanoid Controllers [[Paper](https://arxiv.org/pdf/2405.18418)] [[Torch Code](https://github.com/nicklashansen/puppeteer)]
- :large_blue_diamond: **`ICLR 2024`** [**TD-MPC2**] TD-MPC2: Scalable, Robust World Models for Continuous Control [[Paper](https://arxiv.org/pdf/2310.16828)] [[Torch Code](https://github.com/nicklashansen/tdmpc2)] 
- :large_orange_diamond: **`ICLR 2024`** [**DrM**] DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization [[Paper](https://arxiv.org/pdf/2310.19668)] 
- :large_orange_diamond: **`ICLR 2024 Oral`** [**PTGM**] Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning [[Paper](https://openreview.net/pdf?id=o2IEmeLL9r)] [[Torch Code](https://github.com/PKU-RL/PTGM)]
- :large_blue_diamond: **`ICLR 2024`** [**DreamSmooth**] DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing [[Paper](https://arxiv.org/pdf/2311.01450)]
- :large_blue_diamond: **`ICLR 2024 Oral`** [**R2I**] Mastering Memory Tasks with World Models [[Paper](http://arxiv.org/pdf/2403.04253)] [[JAX Code](https://github.com/OpenDriveLab/ViDAR)]
- :large_orange_diamond: **`RLC 2024`** [**SADA**] A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2405.17416)][[Torch Code](https://github.com/aalmuzairee/dmcgb2)]

## 2023

- :large_orange_diamond: **`ICLR 2023`** [**CoIT**] On the Data-Efficiency with Contrastive Image Transformation in Reinforcement Learning [[Paper](https://openreview.net/forum?id=-nm-rHXi5ga)] [[Torch Code](https://github.com/Kamituna/CoIT)]
- :large_orange_diamond: **`ICCV 2023`** [**CG2A**] Improving Generalization in Visual Reinforcement Learning via Conflict-aware Gradient Agreement Augmentation [[Paper](https://arxiv.org/abs/2308.01194)]
- :large_orange_diamond: **`NeurIPS 2023`** [**HAVE**] Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning [[Paper](https://NeurIPS.cc/virtual/2023/poster/70701)][[Torch Code](https://github.com/Yara-HYR/HAVE)]
- :large_orange_diamond: **`NeurIPS 2023`** [**TACO**] TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2306.13229)][[Torch Code](https://github.com/frankzheng2022/taco)]
- :large_blue_diamond: **`arXiv 2023.8`** [**Dynalang**] Learning to Model the World with Language [[Paper](https://arxiv.org/pdf/2308.01399)] [[JAX Code](https://github.com/jlin816/dynalang)]
- :large_blue_diamond: **`arXiv 2023.5`** [**CoWorld**] Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning [[Paper](https://arxiv.org/pdf/2305.15260)]
- :large_blue_diamond: **`arXiv 2023.1`** [**DreamerV3**] Mastering Atari with Discrete World Models [[Paper](https://arxiv.org/pdf/2301.04104)][[JAX Code](https://github.com/danijar/dreamerv3)][[Torch Code](https://github.com/NM512/dreamerv3-torch)]

## 2022
- :large_orange_diamond: **`ICLR 2022`** [**DrQ-v2**] Local Feature Swapping for Generalization in Reinforcement Learning [[Paper](https://arxiv.org/pdf/2107.09645)][[Torch Code](https://github.com/facebookresearch/drqv2)]
- :large_orange_diamond: **`ICLR 2022`** [**CLOP**] Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning [[Paper](https://openreview.net/forum?id=Sq0-tgDyHe4)][[Torch Code](https://github.com/DavidBert/CLOP)]
- :large_blue_diamond: **`ICML 2022`** [**TD-MPC**] Temporal Difference Learning for Model Predictive Control [[Paper](https://arxiv.org/pdf/2203.04955)][[Torch Code](https://github.com/nicklashansen/tdmpc)]
- :large_orange_diamond: **`ICML 2022`** [**DRIBO**] DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck [[Paper](https://proceedings.mlr.press/v162/fan22b.html)][[Torch Code](https://github.com/BU-DEPEND-Lab/DRIBO)]
- :large_blue_diamond: **`ICML 2022`** [**DreamerPro**] DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations
.  [[Paper](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf)][[TF Code](https://github.com/fdeng18/dreamer-pro)]
- :large_blue_diamond: **`CoRL 2022`** [**DayDreamer**] DayDreamer: World Models for Physical Robot Learning [[Paper](https://proceedings.mlr.press/v205/wu23c/wu23c.pdf)] [[TF Code](https://github.com/danijar/daydreamer)]
- :large_blue_diamond: **`NeurIPS 2022`**  Deep Hierarchical Planning from Pixels.  [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/a766f56d2da42cae20b5652970ec04ef-Paper-Conference.pdf)][[TF Code](https://github.com/danijar/director)]
- :large_blue_diamond: **`NeurIPS 2022 Spotlight`** [**Iso-Dream**] Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9316769afaaeeaad42a9e3633b14e801-Paper-Conference.pdf)][[Torch Code](https://github.com/panmt/Iso-Dream)]


## 2021
- :large_orange_diamond: **`ICLR 2021 Spotlight`** [**DrQ**] Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels  [[Paper](https://arxiv.org/pdf/2004.13649)][[Torch Code](https://github.com/denisyarats/drq)]
- :large_blue_diamond: **`ICLR 2021`** [**DreamerV2**] Mastering Atari with Discrete World Models [[Paper](https://arxiv.org/pdf/2010.02193)][[TF Code](https://github.com/danijar/dreamerv2)][[Torch Code](https://github.com/jsikyoon/dreamer-torch)]

## 2020
- :large_blue_diamond: **`ICLR 2020`** [**DreamerV1**] Dream to Control: Learning Behaviors by Latent Imagination [[Paper](https://arxiv.org/pdf/1912.01603)][[TF Code](https://github.com/danijar/dreamer)][[Torch Code](https://github.com/juliusfrost/dreamer-pytorch)]
- :large_blue_diamond: **`ICML 2020`** [**Plan2Explore**] Planning to Explore via Self-Supervised World Models [[Paper](https://arxiv.org/pdf/2005.05960)][[TF Code](https://github.com/ramanans1/plan2explore)][[Torch Code](https://github.com/yusukeurakami/plan2explore-pytorch)]
- :large_orange_diamond: **`ICML 2020`** [**CURL**] CURL: Contrastive Unsupervised Representations for Reinforcement Learning [[Paper](https://arxiv.org/pdf/2004.04136)] [[Torch Code](https://github.com/MishaLaskin/curl)]

## 2018
- :large_blue_diamond: **`NeurIPS 2018 Oral`** World Models [[Paper](https://arxiv.org/pdf/1803.10122)]

<!-- ## Survey

- A survey on multimodal large language models for autonomous driving. **`WACVW 2024`** [[Paper](https://arxiv.org/abs/2311.12320)] [[Code](https://github.com/IrohXu/Awesome-Multimodal-LLM-Autonomous-Driving)]
- World Models for Autonomous Driving: An Initial Survey. **`2024.3, arxiv`** [[Paper](https://arxiv.org/abs/2403.02622)]

## 2024

- [**ViDAR**] Visual Point Cloud Forecasting enables Scalable Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2312.17655)] [[Code](https://github.com/OpenDriveLab/ViDAR)]
- [**GenAD**] Generalized Predictive Model for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2403.09630)] [[Data](https://github.com/OpenDriveLab/DriveAGI)]
- [**Cam4DOCC**] Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.17663)] [[Code](https://github.com/haomo-ai/Cam4DOcc)]
- [**Drive-WM**] Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.17918)] [[Code](https://github.com/BraveGroup/Drive-WM)]
- [**DriveWorld**] DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving. **`CVPR 2024`** [[Code](https://github.com/chaytonmin/DriveWorld)]
- [**Panacea**] Panacea: Panoramic and Controllable Video Generation for Autonomous Driving. **`CVPR 2024`** [[Paper](https://arxiv.org/abs/2311.16813)] [[Code](https://panacea-ad.github.io/)]
- [**MagicDrive**] MagicDrive: Street View Generation with Diverse 3D Geometry Control. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2310.02601)] [[Code](https://github.com/cure-lab/MagicDrive)]
- [**Copilot4D**] Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion. **`ICLR 2024`** [[Paper](https://arxiv.org/abs/2311.01017)]
- [**SafeDreamer**] SafeDreamer: Safe Reinforcement Learning with World Models. **`ICLR 2024`** [[Paper](https://openreview.net/forum?id=tsE5HLYtYg)] [[Code](https://github.com/PKU-Alignment/SafeDreamer)]
- [**RoboDreamer**] RoboDreamer: Learning Compositional World Models for Robot Imagination. **`2024.4, arxiv`** [[Paper](https://arxiv.org/abs/2404.12377)] [[Code](https://robovideo.github.io/)]
- [**LidarDM**] LidarDM: Generative LiDAR Simulation in a Generated World. **`2024.4, arxiv`** [[Paper](https://arxiv.org/abs/2404.02903)] [[Code](https://github.com/vzyrianov/lidardm)]
- [**3D-VLA**] 3D-VLA: A 3D Vision-Language-Action Generative World Model.  **`2024.3, arxiv`** [[Paper](https://arxiv.org/abs/2403.09631)]
- [**DriveDreamer-2**] DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation. **`2024.3, arxiv`** [[Paper](https://arxiv.org/abs/2403.06845)] [[Code](https://drivedreamer2.github.io/)]
- [**Think2Drive**] Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving. **`2024.2, arxiv`** [[Paper](https://arxiv.org/abs/2402.16720)]

## 2023

- [**TrafficBots**] TrafficBots: Towards World Models for Autonomous Driving Simulation and Motion Prediction. **`ICRA 2023`** [[Paper](https://arxiv.org/abs/2303.04116)] [[Code](https://github.com/zhejz/TrafficBots)]
- [**WoVoGen**] WoVoGen: World Volume-aware Diffusion for Controllable Multi-camera Driving Scene Generation. **`2023.12, arxiv`** [[Paper](https://arxiv.org/abs/2312.02934)] [[Code](https://github.com/fudan-zvg/WoVoGen)]
- [**CTT**] Categorical Traffic Transformer: Interpretable and Diverse Behavior Prediction with Tokenized Latent. **`2023.11, arxiv`** [[Paper](https://arxiv.org/abs/2311.18307)]
- [**OccWorld**] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving. **`2023.11, arxiv`** [[Paper](https://arxiv.org/abs/2311.16038)] [[Code](https://github.com/wzzheng/OccWorld)]
- [**MUVO**] MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations. **`2023.11, arxiv`** [[Paper](https://arxiv.org/abs/2311.11762)]
- [**DrivingDiffusion**] DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model. **`2023.10, arxiv`** [[Paper](https://arxiv.org/abs/2310.07771)] [[Code](https://github.com/shalfun/DrivingDiffusion)]
- [**GAIA-1**] GAIA-1: A Generative World Model for Autonomous Driving. **`2023.9, arxiv`** [[Paper](https://arxiv.org/abs/2309.17080)]
- [**ADriver-I**] ADriver-I: A General World Model for Autonomous Driving. **`2023.9, arxiv`** [[Paper](https://arxiv.org/abs/2311.13549)]
- [**DriveDreamer**] DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. **`2023.9, arxiv`** [[Paper](https://arxiv.org/abs/2309.09777)] [[Code](https://github.com/JeffWang987/DriveDreamer)]
- [**UniWorld**] UniWorld: Autonomous Driving Pre-training via World Models. **`2023.8, arxiv`** [[Paper](https://arxiv.org/abs/2308.07234)] [[Code](https://github.com/chaytonmin/UniWorld)]


## 2022

- [**MILE**] Model-Based Imitation Learning for Urban Driving. **`NeurIPS 2022`** [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/827cb489449ea216e4a257c47e407d18-Abstract-Conference.html)] [[Code](https://github.com/wayveai/mile)]
- [**Symphony**] Symphony: Learning Realistic and Diverse Agents for Autonomous Driving Simulation. **`ICRA 2022`** [[Paper](https://arxiv.org/abs/2205.03195)] 
- Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving. **`IROS 2022`** [[Paper](https://arxiv.org/abs/2210.09539)]

## Other World Model Paper

### 2024

- [**Genie**] Genie: Generative Interactive Environments. **`DeepMind`** [[Paper](https://arxiv.org/abs/2402.15391)] [[Blog](https://sites.google.com/view/genie-2024/home)]
- [**Sora**] Video generation models as world simulators. **`OpenAI`** [[Technical report](https://openai.com/research/video-generation-models-as-world-simulators)]
- [**IWM**] Learning and Leveraging World Models in Visual Representation Learning. **`Meta AI`** [[Paper](https://arxiv.org/abs/2403.00504)] 
- [**V-JEPA**] V-JEPA: Video Joint Embedding Predictive Architecture. **`Meta AI`** [[Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)] [[Paper](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)] [[Code](https://github.com/facebookresearch/jepa)]
- [**Newton**] Newton™ – a first-of-its-kind foundation model for understanding the physical world. **`Archetype AI`** [[Blog](https://www.archetypeai.io/blog/introducing-archetype-ai---understand-the-real-world-in-real-time)]
- [**MAMBA**] MAMBA: an Effective World Model Approach for Meta-Reinforcement Learning. **`ICLR 2024`**  [[Paper](https://arxiv.org/abs/2403.09859)] [[Code](https://github.com/zoharri/mamba)]
- [**Compete and Compose**] Compete and Compose: Learning Independent Mechanisms for Modular World Models. **`2024.4, arxiv`**  [[Paper](https://arxiv.org/abs/2404.15109)]
- [**MagicTime**] MagicTime: Time-lapse Video Generation Models as Metamorphic Simulators. **`2024.4, arxiv`**  [[Paper](https://arxiv.org/abs/2404.05014)] [[Code](https://github.com/PKU-YuanGroup/MagicTime)]
- [**Dreaming of Many Worlds**] Dreaming of Many Worlds: Learning Contextual World Models Aids Zero-Shot Generalization. **`2024.3, arxiv`**  [[Paper](https://arxiv.org/abs/2403.10967)] [[Code](https://github.com/sai-prasanna/dreaming_of_many_worlds)]
- [**ManiGaussian**] ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation. **`2024.3, arxiv`**  [[Paper](https://arxiv.org/abs/2403.08321)] [[Code](https://guanxinglu.github.io/ManiGaussian/)]
- [**LWM**] World Model on Million-Length Video And Language With RingAttention. **`2024.2, arxiv`**  [[Paper](https://arxiv.org/abs/2402.08268)] [[Code](https://github.com/LargeWorldModel/LWM)]
- Planning with an Ensemble of World Models. **`OpenReview`** [[Paper](https://openreview.net/forum?id=cvGdPXaydP)]
- [**WorldDreamer**] WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens. **`2024.1, arxiv`** [[Paper](https://arxiv.org/abs/2401.09985)] [[Code](https://github.com/JeffWang987/WorldDreamer)] -->

<p align="right">(<a href="#top">Back to top</a>)</p>

