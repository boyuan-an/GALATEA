<h1 align="center">GALATEA: Grounding Generated Video Plans in Simulation Towards Versatile Dexterous Controllers</h1>

<p align="center">
  <a href="https://tianyueh8erobot.github.io/">Tianyue Wu</a><sup>2,3,1,*,‡</sup>,
  <a href="https://boyuan-an.github.io/">Boyuan An</a><sup>2,1,*</sup>,
  <a href="https://zhao-sq.github.io/">Shuqi Zhao</a><sup>1</sup>,
  <a href="https://github.com/GuoHeyu">Heyu Guo</a><sup>2</sup>,
  <a href="https://wlxing1901.github.io/">Wanli Xing</a><sup>2</sup>,
  <br>
  <a href="https://people.eecs.berkeley.edu/~yima/">Yi Ma</a><sup>3,1</sup>,
  <a href="https://openreview.net/profile?id=%7EKaifeng_Zhang1">Kaifeng Zhang</a><sup>2</sup>,
  <a href="https://warshallrho.github.io/">Ruihai Wu</a><sup>1,†</sup>,
  <a href="https://me.berkeley.edu/people/masayoshi-tomizuka/">Masayoshi Tomizuka</a><sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>UC Berkeley &nbsp;&nbsp; <sup>2</sup>Sharpa &nbsp;&nbsp; <sup>3</sup>The University of Hong Kong
  <br>
  <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Co-advisors &nbsp;&nbsp; <sup>‡</sup>Corresponding author
</p>

<h3 align="center">
  <a href="https://boyuan-an.github.io/GALATEA/resources/GALATEA_paper.pdf">📄 Paper</a> |
  <a href="https://boyuan-an.github.io/GALATEA/">🌐 Project Page</a>
</h3>

<p align="center">
  <img src="assets/galatea_pipeline_overview.jpg" width="95%" alt="Overview of the GALATEA pipeline">
</p>

## TL;DR

Video generation as reference data to train RL dexterous controller when training, and as motion planner executed by the tracker when deploying.

## Abstract

Generated hand–object interaction videos provide a controllable way to propose manipulation motions. **GALATEA** combines generated videos with simulation-based HOI grounding: generated videos provide diverse motion references for learning a multi-object, multi-trajectory HOI tracker, while at deployment time video models produce motion plans that are executed by the learned tracker. With minimal manual intervention, the reconstruction pipeline yields approximately 2,000 usable references from 2,500 generated clips, of which more than 1,500 trajectories are grounded in simulation. In closed-loop real-world experiments, the distilled controller performs functional grasps, non-prehensile manipulation, and post-grasp object-pose tracking.

---

## 📋 TODO

We will publicly release both items below **before November 2026**.

- [ ] Release the reinforcement-learning tracking code and example datasets.
- [ ] Release the HOI reconstruction code and example generated videos, together with their prompts.
