# Holdout Paper Set (Prefilter Run 7)

## What is a holdout set?

A holdout set is a collection of real-world scientific ML code that was **never used during pattern development or refinement**. It provides an unbiased estimate of detection precision on unseen data.

- Patterns were iteratively improved using the **meta loop set** (feedback from verification results guided changes to detection questions, test files, and prompts)
- The holdout set was sampled **after** all iteration was complete, explicitly excluding all papers from the meta loop set
- Holdout results should **not** be used for further pattern tuning — doing so would compromise the unbiased measurement

If precision on the holdout set is close to the meta loop set, patterns generalize well. A large gap indicates overfitting to the iteration data.

## Metadata

- **Prefilter run:** 7
- **Self-contained files:** 45 (from 17 repos)
- **Analysis run:** 60
- **Papers:** 35
- **Seed:** 99
- **Excluded:** prefilter runs 4, 6 (meta loop set papers)
- **Date:** 2026-03-16

## Data Funnel

| Stage | Count | Note |
|-------|-------|------|
| Papers sampled | 35 | seed 99, excluding meta loop set |
| Repos cloned | 37 | some papers have multiple repos, 1 clone failed |
| Repos with qualifying files | 31 | Python files with ML imports |
| Qualifying files | 562 | |
| **Papers with self-contained files** | **17** | 18 papers had only fragments |
| Repos with self-contained files | 17 | 14 repos had only fragments |
| Self-contained files | 45 | 8% of qualifying files |
| Files with findings | 37 | 82% of analyzed files |
| Clean files (no findings) | 8 | 18% of analyzed files |
| Papers with any findings | 15 | 88% of papers with self-contained files |
| **Papers with verified real bugs** | **12** | **71% of papers with self-contained files** |

51% of papers (18/35) had no self-contained files after prefiltering — all their code was fragments/utilities requiring other files to run.

## Results

Overall precision: **37.9%** (39 valid / 103 findings)

### By Category

| Category | Findings | Valid | Precision |
|----------|----------|-------|-----------|
| ai-inference | 24 | 15 | 62.5% |
| scientific-reproducibility | 29 | 16 | 55.2% |
| scientific-numerical | 7 | 3 | 42.9% |
| ai-training | 26 | 3 | 11.5% |
| scientific-performance | 17 | 2 | 11.8% |

### By Severity

| Severity | Findings | Valid | Precision |
|----------|----------|-------|-----------|
| medium | 30 | 16 | 53.3% |
| high | 55 | 20 | 36.4% |
| critical | 18 | 3 | 16.7% |

### By Domain

| Domain | Papers Analyzed | Findings | Valid | Precision |
|--------|----------------|----------|-------|-----------|
| astronomy | 2 | 25 | 11 | 44.0% |
| none (general ML) | 2 | 23 | 7 | 30.4% |
| earth_science | 1 | 15 | 3 | 20.0% |
| medicine | 2 | 11 | 7 | 63.6% |
| engineering | 3 | 11 | 4 | 36.4% |
| biology | 1 | 9 | 5 | 55.6% |
| neuroscience | 1 | 3 | 0 | 0.0% |
| medicine, earth_science | 1 | 3 | 2 | 66.7% |
| chemistry | 1 | 2 | 0 | 0.0% |
| physics | 1 | 1 | 0 | 0.0% |

## Papers

| Domain | Title | URL |
|--------|-------|-----|
| astronomy | AstroCLIP: A Cross-Modal Foundation Model for Galaxies | https://paperswithcode.com/paper/astroclip-cross-modal-pre-training-for |
| astronomy | Mantis Shrimp: Exploring Photometric Band Utilization in Computer Vision Networks for Photometric Redshift Estimation | https://paperswithcode.com/paper/mantis-shrimp-exploring-photometric-band |
| biology | Generative diffusion model with inverse renormalization group flows | https://paperswithcode.com/paper/generative-diffusion-model-with-inverse |
| chemistry | All-atom Diffusion Transformers: Unified generative modelling of molecules and materials | https://paperswithcode.com/paper/all-atom-diffusion-transformers-unified |
| chemistry | Graph Diffusion Transformers for Multi-Conditional Molecular Generation | https://paperswithcode.com/paper/inverse-molecular-design-with-multi |
| earth_science | A case study of spatiotemporal forecasting techniques for weather forecasting | https://paperswithcode.com/paper/a-case-study-of-spatiotemporal-forecasting |
| earth_science | Building Machine Learning Limited Area Models: Kilometer-Scale Weather Forecasting in Realistic Settings | https://paperswithcode.com/paper/building-machine-learning-limited-area-models |
| earth_science | Fuxi-DA: A Generalized Deep Learning Data Assimilation Framework for Assimilating Satellite Observations | https://paperswithcode.com/paper/fuxi-da-a-generalized-deep-learning-data |
| earth_science | Increasing the Robustness of Model Predictions to Missing Sensors in Earth Observation | https://paperswithcode.com/paper/increasing-the-robustness-of-model |
| earth_science | Robust and Conjugate Spatio-Temporal Gaussian Processes | https://paperswithcode.com/paper/robust-and-conjugate-spatio-temporal-gaussian |
| economics | ST-RAP: A Spatio-Temporal Framework for Real Estate Appraisal | https://paperswithcode.com/paper/st-rap-a-spatio-temporal-framework-for-real |
| engineering | A Hybrid Sparse-Dense Monocular SLAM System for Autonomous Driving | https://paperswithcode.com/paper/a-hybrid-sparse-dense-monocular-slam-system |
| engineering | A Semi-Decoupled Approach to Fast and Optimal Hardware-Software Co-Design of Neural Accelerators | https://paperswithcode.com/paper/a-semi-decoupled-approach-to-fast-and-optimal |
| engineering | GP-PCS: One-shot Feature-Preserving Point Cloud Simplification with Gaussian Processes on Riemannian Manifolds | https://paperswithcode.com/paper/one-shot-feature-preserving-point-cloud |
| engineering | Graph Neural Network for Large-Scale Network Localization | https://paperswithcode.com/paper/graph-neural-network-for-large-scale-network |
| engineering | Unsupervised Optimal Power Flow Using Graph Neural Networks | https://paperswithcode.com/paper/unsupervised-optimal-power-flow-using-graph |
| materials | AtomAI: A Deep Learning Framework for Analysis of Image and Spectroscopy Data in (Scanning) Transmission Electron Microscopy and Beyond | https://paperswithcode.com/paper/atomai-a-deep-learning-framework-for-analysis |
| medicine | An Improved Deep Convolutional Neural Network by Using Hybrid Optimization Algorithms to Detect and Classify Brain Tumor Using Augmented MRI Images | https://paperswithcode.com/paper/an-improved-deep-convolutional-neural-network-1 |
| medicine | Large AI Models in Health Informatics: Applications, Challenges, and the Future | https://paperswithcode.com/paper/large-ai-models-in-health-informatics |
| medicine | Precision Anti-Cancer Drug Selection via Neural Ranking | https://paperswithcode.com/paper/precision-anti-cancer-drug-selection-via |
| medicine | Selective Information Passing for MR/CT Image Segmentation | https://paperswithcode.com/paper/selective-information-passing-for-mr-ct-image |
| medicine | VolTex: Food Volume Estimation using Text-Guided Segmentation and Neural Surface Reconstruction | https://paperswithcode.com/paper/voltex-food-volume-estimation-using-text |
| medicine, earth_science | Estimating Epistemic and Aleatoric Uncertainty with a Single Model | https://paperswithcode.com/paper/hyper-diffusion-estimating-epistemic-and |
| medicine, earth_science | Regression Conformal Prediction under Bias | https://paperswithcode.com/paper/regression-conformal-prediction-under-bias |
| neuroscience | GOUHFI: a novel contrast- and resolution-agnostic segmentation tool for Ultra-High Field MRI | https://paperswithcode.com/paper/gouhfi-a-novel-contrast-and-resolution |
| neuroscience | Unsupervised Learning of Spatio-Temporal Patterns in Spiking Neuronal Networks | https://paperswithcode.com/paper/unsupervised-learning-of-spatio-temporal |
| none | Ai2 Scholar QA: Organized Literature Synthesis with Attribution | https://paperswithcode.com/paper/ai2-scholar-qa-organized-literature-synthesis |
| none | CNN Fixations: An unraveling approach to visualize the discriminative image regions | https://paperswithcode.com/paper/cnn-fixations-an-unraveling-approach-to |
| none | Lifelong Learning on Evolving Graphs Under the Constraints of Imbalanced Classes and New Classes | https://paperswithcode.com/paper/lifelong-learning-in-evolving-graphs-with |
| none | UPGPT: Universal Diffusion Model for Person Image Generation, Editing and Pose Transfer | https://paperswithcode.com/paper/upgpt-universal-diffusion-model-for-person |
| none | What to Remember: Self-Adaptive Continual Learning for Audio Deepfake Detection | https://paperswithcode.com/paper/what-to-remember-self-adaptive-continual |
| physics | Automating quantum feature map design via large language models | https://paperswithcode.com/paper/automating-quantum-feature-map-design-via |
| physics | PDE-DKL: PDE-constrained deep kernel learning in high dimensionality | https://paperswithcode.com/paper/pde-dkl-pde-constrained-deep-kernel-learning |
| physics | Quantum adiabatic machine learning with zooming | https://paperswithcode.com/paper/quantum-adiabatic-machine-learning-with |
| physics | Reverse Map Projections as Equivariant Quantum Embeddings | https://paperswithcode.com/paper/reverse-map-projections-as-equivariant |
