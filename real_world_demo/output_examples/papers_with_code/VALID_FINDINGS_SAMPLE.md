# Valid Findings - Quick Verification Sample

**10 verified findings** (2 critical + 8 high) with pattern diversity for fast manual verification.

## 1. ml-001 (critical)

**File:** 
**Repo:** [louzounlab__pygon](https://github.com/louzounlab/pygon)
**Location:** function `_scale_matrices` (line 58)
**Paper:** Planted Dense Subgraphs in Dense Random Graphs Can Be Recovered using Graph-based Machine Learning
**Authors:** Itay Levinas, yoram louzoun

**Issue:** ml-001: Issue detected

**Explanation:** Data leakage: scaler/encoder is fit on full data including test set. Model performance will be inflated. Use sklearn.pipeline.Pipeline so fitting happens inside each fold.



**Code:**
```python
self._labels += [labels]
        self._scale_matrices()

    def _scale_matrices(self):
        scaler = StandardScaler()
        all_matrix = np.vstack(self._feature_matrices)
        scaler.fit(all_matrix)
```

**Verification reasoning:** VALID

The scaler is fit on `all_matrix` which is the full dataset stacked together before any train/test split occurs. When `train()` later calls `pygon.run_pygon` with `check='split'`, the test split has already been scaled using statistics derived from it, causing data leakage. The scaling should happen inside each cross-validation fold or split.

---

## 2. ml-007 (critical)

**File:** 
**Repo:** [openclimatefix__graph_weather](https://github.com/openclimatefix/graph_weather)
**Location:** function `Era5Dataset.__init__` (line 120)
**Paper:** WeatherMesh-3: Fast and accurate operational global weather forecasting
**Authors:** Haoxing Du et al.

**Issue:** ml-007: Issue detected

**Explanation:** Data leakage: fit_transform on test data means the test set uses its own statistics instead of training statistics. Use transform() on test data.



**Code:**
```python
Arguments:
            #TODO
        """
        ds = np.asarray(xarr.to_array())
        ds = torch.from_numpy(ds)
        ds -= ds.min(0, keepdim=True)[0]
        ds /= ds.max(0, keepdim=True)[0]
```

**Verification reasoning:** VALID: This is a real issue that should be fixed

The code performs min-max normalization using each dataset instance's own statistics (`ds.min`, `ds.max`). If `Era5Dataset` is instantiated separately for train and test splits, each will be normalized by its own min/max values rather than the training set's statistics — this is the same data leakage problem as `fit_transform` on test data, just implemented manually. The fix requires computing normalization statistics from the training set and passing them in (e.g., via the `transform` parameter that's already accepted but unused).

---

## 3. perf-001 (high)

**File:** 
**Repo:** [czq142857__NDC](https://github.com/czq142857/NDC)
**Location:** function `get_v_t_count` (line 61)
**Paper:** Neural Dual Contouring
**Authors:** Zhiqin Chen et al.

**Issue:** perf-001: Issue detected

**Explanation:** Python loop over NumPy array: 100-1000x slower than vectorized operation. Use broadcasting, np.where, or fancy indexing.



**Code:**
```python
def get_v_t_count(q, name_list):
    name_num = len(name_list)
    for nid in range(name_num):
        pid = name_list[nid][0]
        idx = name_list[nid][1]
        gt_obj_name = name_list[nid][2]
        pred_obj_name = name_list[nid][3]


                
                  
        v,t = load_obj(pred_obj_name)
        
        slot = np.zeros([180],np.int32)

        for i in range(len(t)):
            for j in range(3):
                v1i = t[i,(j)%3]
                v2i = t[i,(j+1)%3]
                v3i = t[i,(j+2)%3]
                
                v1 = v[v1i]
                v2 = v[v2i]
                v3 = v[v3i]

                v12 = v2-v1
                v13 = v3-v1

                s_a = np.sqrt( np.sum(v12*v12)*np.sum(v13*v13) )
                if s_a<1e-10:
                    s_a = 1e-10
                cos_a = np.sum(v12*v13)/s_a
                if cos_a>1: cos_a=1
                if cos_a<-1: cos_a=-1
                a = (np.arccos(cos_a)/np.pi*180).astype(np.int32)
                if a>=180:
                    a = 180-1
                slot[a] += 1


        print(idx)
        q.put([idx,slot])
```

**Verification reasoning:** VALID: The nested Python loops over triangle array `t` (lines 61-83) iterate element-by-element to compute per-vertex angles, which is exactly the 100-1000x slower pattern. This could be fully vectorized using NumPy fancy indexing to gather all vertex triplets at once, then computing dot products and arccos across the entire triangle array in batch. For meshes with thousands of triangles, this loop is a significant bottleneck.

---

## 4. rep-002 (high)

**File:** 
**Repo:** [lsh0520__3d-molm](https://github.com/lsh0520/3d-molm)
**Location:** function `main` (line 23)
**Paper:** Towards 3D Molecule-Text Interpretation in Language Models
**Authors:** Sihang Li et al.

**Issue:** rep-002: Issue detected

**Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.



**Code:**
```python
def main(args):
    pl.seed_everything(args.seed)
           
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)

    print(' total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2opt.llm_tokenizer
          

    dm = Stage2DM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, model.blip2opt.dictionary, tokenizer, args)

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )

    if args.mode == 'pretrain':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'ft':
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()
```

**Verification reasoning:** VALID: The code calls `pl.seed_everything(args.seed)` but never sets `torch.use_deterministic_algorithms(True)` or `torch.backends.cudnn.benchmark = False`, meaning CUDA operations (scatter, atomics, etc.) remain non-deterministic regardless of the seed. Notably, `torch.set_float32_matmul_precision('medium')` further reduces precision reproducibility. One mitigating caveat: Flash Attention is enabled by default (`args.enable_flash = True`), which is incompatible with `use_deterministic_algorithms(True)` and would require disabling flash attention to achieve full determinism.

---

## 5. perf-004 (high)

**File:** 
**Repo:** [cbuelt__pfno](https://github.com/cbuelt/pfno)
**Location:** method `Spatial_CRPS.calculate_score` (line 70)
**Paper:** Probabilistic neural operators for functional uncertainty quantification
**Authors:** Christopher Bülte et al.

**Issue:** perf-004: Issue detected

**Explanation:** Materializing large intermediate arrays wastes memory. Use np.linalg.norm or in-place operations to avoid temporary allocations.



**Code:**
```python
def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the CRPS for two tensors, without aggregating.

        Args:
            x (torch.Tensor): Model x (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: CRPS
        """
        n_samples = x.size()[-1]

                                               
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

                         
        es_12 = torch.abs(x-y).mean(axis = -1)

        diff = torch.unsqueeze(x, dim=-2) - torch.unsqueeze(x, dim=-1)
        es_22 = torch.abs(diff).sum(axis = (-1,-2))

        score = es_12 - es_22 / (2 * n_samples * (n_samples - 1))
        return score
```

**Verification reasoning:** VALID: The `diff = torch.unsqueeze(x, dim=-2) - torch.unsqueeze(x, dim=-1)` line materializes a full O(n_samples²) pairwise difference tensor across all batch and spatial dimensions, which grows quadratically with ensemble size. For physics simulation data with large spatial fields (this appears to be ERA5 weather data), this intermediate tensor can be enormous and cause OOM or severe memory pressure. The energy score term for CRPS can be computed more memory-efficiently using a sequential or chunked approach rather than materializing the full pairwise matrix.

---

## 6. perf-002 (high)

**File:** 
**Repo:** [MLRG-CEFET-RJ__stconvs2s](https://github.com/MLRG-CEFET-RJ/stconvs2s)
**Location:** function `run_arima` (line 74)
**Paper:** STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting
**Authors:** Rafaela Castro et al.

**Issue:** perf-002: Issue detected

**Explanation:** Array allocation in loop: creating arrays inside loops causes repeated malloc/free operations. Pre-allocate the output array before the loop and fill it using indexing.


**Code:**
```python
def run_arima(df, chirps, step):
    series = None
    rmse_val, mae_val = 0.,0. 
    rmse_mean, mae_mean = -999., -999.
    lat = df['lat'].unique()
    lon = df['lon'].unique()
    try:
        series = df['precip'] if (chirps) else df['air_temp']
        if ((series > 0).any()):
            split = len(series) - (step + 5)
            train = series[:split].values
            test = series[split:].values
            test_sequence = create_test_sequence(test, step)
            for observation, sequence in zip(test,test_sequence):
                start_index = len(train)
                end_index = start_index + (step-1)
                model = SARIMAX(train, order=(5,0,1)) 
                results = model.fit(disp=False) 
                pred_sequence = results.predict(start=start_index, end=end_index, dynamic=False)
                rmse_val += rmse(sequence, pred_sequence) 
                mae_val += mean_absolute_error(sequence, pred_sequence)
                np.append(train,observation) 
            
            rmse_mean = rmse_val/len(test_sequence) 
            mae_mean = mae_val/len(test_sequence)
            print(f'\n=> Model ARIMA lat: {lat}, lon: {lon}')
            print(f'RMSE: {rmse_mean:.8f}')
            print(f'MAE: {mae_mean:.8f}')    
        else:
            print(f'\n** lat: {lat}, lon: {lon} has all zero values')
    except Exception as e:
        print(f'\n## lat: {lat}, lon: {lon} error: {e}')
    
    sys.stdout.flush()
    return (rmse_mean, mae_mean)
```

**Verification reasoning:** VALID: `np.append(train, observation)` at line 74 creates a new array on every loop iteration (np.append always allocates a new array), and the result is discarded — making this both a performance issue (repeated allocation/deallocation each iteration) and a correctness bug (the training set never actually grows as intended). The intended rolling-window ARIMA update is silently broken.

---

## 7. pt-007 (high)

**File:** 
**Repo:** [openclimatefix__graph_weather](https://github.com/openclimatefix/graph_weather)
**Location:** method `LitModel.plot_sample` (line 189)
**Paper:** WeatherMesh-3: Fast and accurate operational global weather forecasting
**Authors:** Haoxing Du et al.

**Issue:** pt-007: Issue detected

**Explanation:** Missing model.eval() leaves dropout active and batchnorm using batch statistics instead of learned running statistics, producing incorrect inference results.


**Code:**
```python
def plot_sample(self, prev_inputs, target_residuals):
        """Plot 2m_temperature and geopotential"""
        prev_inputs = prev_inputs[:1, :, :, :]
        target = target_residuals[:1, :, :, :]
        sampler = Sampler()
        preds = sampler.sample(self.model, prev_inputs)

        fig1, ax = plt.subplots(2)
        ax[0].imshow(preds[0, :, :, 78].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Diffusion sampling prediction")

        ax[1].imshow(target[0, :, :, 78].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Ground truth")

        fig2, ax = plt.subplots(2)
        ax[0].imshow(preds[0, :, :, 12].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Diffusion sampling prediction")

        ax[1].imshow(target[0, :, :, 12].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Ground truth")

        return fig1, fig2
```

**Verification reasoning:** VALID: The `plot_sample` method is called from `on_train_epoch_start`, meaning `self.model` is in training mode when `sampler.sample(self.model, prev_inputs)` runs. Without `self.model.eval()`, dropout stays active and batchnorm uses batch statistics during this diffusion sampling, producing noisy/incorrect visualization samples. The model also isn't restored to training mode afterward (though it stays in training mode since `eval()` was never called, so that part is fine).

---

## 8. num-005 (high)

**File:** 
**Repo:** [ArnaudFerre__C-Norm](https://github.com/ArnaudFerre/C-Norm)
**Location:** function `normalizeEmbedding` (line 49)
**Paper:** C-Norm: a neural approach to few-shot entity normalization
**Authors:** Arnaud Ferré et al.

**Issue:** num-005: Issue detected

**Explanation:** Division by zero in normalization: std is 0 for constant features, producing inf or NaN. Check for zero std and handle it.



**Code:**
```python
def normalizeEmbedding(vst_onlyTokens):

    for token in vst_onlyTokens.keys():
        vst_onlyTokens[token] = vst_onlyTokens[token] / numpy.linalg.norm(vst_onlyTokens[token])

    return vst_onlyTokens
```

**Verification reasoning:** VALID

The code divides each token embedding by its L2 norm (`numpy.linalg.norm`), which returns 0.0 for a zero vector. This produces `NaN` values (0/0) rather than `inf`, but the result is equally problematic — NaN values would silently propagate through all downstream neural network computations (the TensorFlow/Keras model visible in imports). There is no guard against zero-norm vectors anywhere in the function.

---

## 9. pt-015 (high)

**File:** 
**Repo:** [GentleDell__Better-Patch-Stitching](https://github.com/GentleDell/Better-Patch-Stitching)
**Location:** function `main` (line 114)
**Paper:** Better Patch Stitching for Parametric Surface Reconstruction
**Authors:** Zhantao Deng et al.

**Issue:** pt-015: Issue detected

**Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.


**Code:**
```python
if args.resume:
    print("Resuming training")
    trstate = torch.load(helpers.jn(args.output, 'chkpt.tar'))
    model.load_state_dict(trstate['weights'])
    opt.load_state_dict(trstate['optimizer'])
```

**Verification reasoning:** VALID: `torch.load()` is called without `map_location` at line 114. The code uses a `gpu` variable and `DataLoaderDevice` suggesting GPU-aware training, so a checkpoint saved on GPU will fail to load on a CPU-only system. The `gpu` variable is already in scope and should be passed as `map_location`.

---

## 10. rep-002 (high)

**File:** 
**Repo:** [rose-stl-lab__spherical-dyffusion](https://github.com/rose-stl-lab/spherical-dyffusion)
**Location:** function `main` (line 388)
**Paper:** Probabilistic Emulation of a Global Climate Model with Spherical DYffusion
**Authors:** Salva Rühling Cachay et al.

**Issue:** rep-002: Issue detected

**Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.



**Code:**
```python
def main(yaml_config: str):
    dist = Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(train_config.experiment_dir):
        os.makedirs(train_config.experiment_dir)
    with open(os.path.join(train_config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    train_config.configure_logging(log_filename="out.log")
    env_vars = logging_utils.retrieve_env_vars()
    gcs_utils.authenticate()
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    train_config.configure_wandb(env_vars=env_vars, resume=True, notes=beaker_url)
    trainer = Trainer(train_config)
    trainer.train()
    logging.info("DONE ---- rank %d" % dist.rank)
```

**Verification reasoning:** VALID: The code explicitly sets `torch.backends.cudnn.benchmark = True` without setting `torch.use_deterministic_algorithms(True)`, which introduces non-determinism in GPU operations. This is a training script for an earth science model where reproducibility matters. The benchmark flag prioritizes speed over determinism, which can produce different results across runs.

---
