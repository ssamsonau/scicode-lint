# Real-World Scientific ML Code Analysis Report

Analysis of Python code from **AI applications to scientific domains**. Papers sourced from PapersWithCode, filtered to include only scientific domains (biology, chemistry, physics, medicine, earth science, astronomy, materials, etc.) where ML/AI is applied to scientific discovery and domain-specific research.

## Analysis Summary

- **Analysis Date:** 2026-03-16 19:47
- **Report Generated:** 2026-03-16 18:01
- **scicode-lint Version:** 0.2.1
- **Papers with Findings:** 27 / 32 (84.4%)
- **Repos with Findings:** 30 / 35 (85.7%)
- **Files Analyzed:** 120 / 120
- **Files with Findings:** 90 (75.0%)
- **Total Findings:** 219

## Prefilter Summary

Files were classified by LLM as self-contained ML pipelines vs code fragments. Only self-contained files (complete training/inference workflows) were analyzed.

| Classification | Files | % |
|----------------|-------|---|
| Self-contained (analyzed) | 120 | 13.6% |
| Fragment (skipped) | 749 | 84.7% |
| Error | 15 | 1.7% |
| **Total** | **884** | |

### Papers/Repos Filtered

- **Papers:** 32 analyzed / 38 original (6 dropped)
- **Repos:** 35 analyzed / 47 original (12 dropped)

**12 repos dropped** (all files classified as fragments):

- `AstraZeneca__chemicalx`
- `CompRhys__aviary`
- `CompRhys__roost`
- `citrineinformatics-erd-public__piml_glass_forming_ability`
- `coarse-graining__cgnet`
- `idea-iitd__greed`
- `idea-iitd__neurosed`
- `mtian8__gw_spatiotemporal_gnn`
- `sheoyon-jhin__contime`
- `tpospisi__DeepCDE`
- ... and 2 more

*Prefilter model: qwen3-8b-fp8*

## Papers by Severity

Papers with at least one finding of each severity level (a paper may appear in multiple rows):

| Severity | Papers | % of Papers Analyzed |
|----------|--------|----------------------|
| Critical | 12 | 37.5% |
| High | 25 | 78.1% |
| Medium | 18 | 56.2% |

*Total papers analyzed: 32*

## Findings Distribution (per paper)

| Metric | All | Critical | High | Medium | Low |
|--------|-----|----------|------|--------|-----|
| Papers | 27 | 12 | 25 | 18 | 0 |
| Min | 1 | 1 | 1 | 1 | - |
| Max | 41 | 4 | 20 | 20 | - |
| Mean | 8.1 | 2.4 | 5.4 | 3.1 | - |
| Median | 6.0 | 2.5 | 4.0 | 2.0 | - |
| Std Dev | 8.9 | 1.2 | 5.0 | 4.4 | - |

## Verification Summary

**Overall Precision:** 45.2% (99 valid / 219 verified)

| Status | Count | % |
|--------|-------|---|
| Valid (confirmed) | 99 | 45.2% |
| Invalid (false positive) | 111 | 50.7% |
| Uncertain | 9 | 4.1% |

### Verified Findings by Severity

| Severity | Total | Valid | Invalid | Uncertain | Pending | Precision |
|----------|-------|-------|---------|-----------|---------|-----------|
| Critical | 29 | 3 | 24 | 2 | 0 | 10% |
| High | 135 | 69 | 60 | 6 | 0 | 51% |
| Medium | 55 | 27 | 27 | 1 | 0 | 49% |

## Findings by Scientific Domain

| Domain | Files Analyzed | With Findings | Finding Rate | Total Findings |
|--------|---------------|---------------|--------------|----------------|
| chemistry | 45 | 31 | 68.9% | 60 |
| earth_science | 17 | 16 | 94.1% | 46 |
| materials | 5 | 5 | 100.0% | 22 |
| none | 8 | 7 | 87.5% | 19 |
| economics | 6 | 5 | 83.3% | 19 |
| biology | 7 | 7 | 100.0% | 14 |
| engineering | 10 | 6 | 60.0% | 13 |
| medicine | 8 | 7 | 87.5% | 13 |
| astronomy | 5 | 3 | 60.0% | 7 |
| neuroscience | 3 | 2 | 66.7% | 4 |
| physics | 2 | 1 | 50.0% | 2 |
| mathematics | 2 | 0 | 0.0% | 0 |
| social_science | 1 | 0 | 0.0% | 0 |

## Findings by Category

| Category | Count | Unique Files | Unique Repos |
|----------|-------|--------------|--------------|
| scientific-reproducibility | 73 | 56 | 24 |
| scientific-performance | 55 | 40 | 19 |
| ai-training | 40 | 29 | 16 |
| ai-inference | 38 | 35 | 16 |
| scientific-numerical | 13 | 11 | 7 |

## Findings by Severity

| Severity | Count | % of Total |
|----------|-------|------------|
| Critical | 29 | 13.2% |
| High | 135 | 61.6% |
| Medium | 55 | 25.1% |

## Most Common Patterns

| Pattern | Category | Count | Files | Repos | Avg Confidence |
|---------|----------|-------|-------|-------|----------------|
| rep-002 | scientific-reproducibility | 32 | 32 | 14 | 95% |
| perf-004 | scientific-performance | 19 | 19 | 11 | 95% |
| rep-003 | scientific-reproducibility | 17 | 17 | 13 | 95% |
| pt-015 | ai-inference | 14 | 14 | 6 | 95% |
| par-005 | scientific-performance | 13 | 13 | 10 | 93% |
| pt-013 | ai-inference | 11 | 11 | 5 | 95% |
| rep-004 | scientific-reproducibility | 10 | 10 | 7 | 95% |
| perf-001 | scientific-performance | 10 | 10 | 7 | 95% |
| num-005 | scientific-numerical | 8 | 8 | 5 | 95% |
| ml-009 | ai-training | 8 | 8 | 8 | 95% |

## Example Findings

Representative findings from each category (with links to source):

### ai-inference

**pt-015** (high, 95% confidence)

- **Repo:** GentleDell__Better-Patch-Stitching (none)
- **Location:** function `main` (line 114)
- **Paper:** Better Patch Stitching for Parametric Surface Reconstruction
- **Authors:** Zhantao Deng et al.
- **Issue:** pt-015: Issue detected
- **Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.

- **Suggestion:** Review the code and fix according to the explanation.

```python
if args.resume:
    print("Resuming training")
    trstate = torch.load(helpers.jn(args.output, 'chkpt.tar'))
    model.load_state_dict(trstate['weights'])
    opt.load_state_dict(trstate['optimizer'])
```

**pt-015** (high, 95% confidence)

- **Repo:** songtaoliu0823__crebm (chemistry)
- **Location:** module `<module>` (line 67)
- **Paper:** Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models
- **Authors:** Songtao Liu et al.
- **Issue:** pt-015: Issue detected
- **Explanation:** torch.load() without map_location can fail on CPU-only systems. Always specify map_location='cpu' or map_location=device for portable model loading.

- **Suggestion:** Review the code and fix according to the explanation.

```python
np.random.seed(args.seed)
random.seed(args.seed)

              
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ValueMLP(
        n_layers=args.n_layers,
```

**pt-007** (high, 95% confidence)

- **Repo:** rose-stl-lab__spherical-dyffusion (earth_science)
- **Location:** method `validate_one_epoch` (line 318)
- **Paper:** Probabilistic Emulation of a Global Climate Model with Spherical DYffusion
- **Authors:** Salva Rühling Cachay et al.
- **Issue:** pt-007: Issue detected
- **Explanation:** Missing model.eval() leaves dropout active and batchnorm using batch statistics instead of learned running statistics, producing incorrect inference results.

- **Suggestion:** Review the code and fix according to the explanation.

```python
def validate_one_epoch(self):
        aggregator = OneStepAggregator(
            self.train_data.area_weights.to(fme.get_device()),
            self.train_data.sigma_coordinates,
            self.train_data.metadata,
        )

        with torch.no_grad(), self._validation_context():
            for batch in self.valid_data.loader:
                stepped = self.stepper.run_on_batch(
                    batch.data,
                    optimization=NullOptimization(),
                    n_forward_steps=self.config.n_forward_steps,
                    aggregator=NullAggregator(),
                )
                stepped = compute_stepped_derived_quantities(stepped, self.valid_data.sigma_coordinates)
                aggregator.record_batch(
                    loss=stepped.metrics["loss"],
                    target_data=stepped.target_data,
                    gen_data=stepped.gen_data,
                    target_data_norm=stepped.target_data_norm,
                    gen_data_norm=stepped.gen_data_norm,
                )
        return aggregator.get_logs(label="val")
```

### ai-training

**ml-005** (critical, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** function `_init_data_loader` (line 279)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** ml-005: Issue detected
- **Explanation:** Temporal leakage: shuffled cross-validation on time-series data trains on future to predict past. Use TimeSeriesSplit.


- **Suggestion:** Review the code and fix according to the explanation.

```python
self.dataset : TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = ChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
```

**pt-004** (critical, 95% confidence)

- **Repo:** PaddlePaddle__PaddleScience (earth_science)
- **Location:** function `util.train_one_epoch` (line 77)
- **Paper:** GenCast: Diffusion-based ensemble forecasting for medium-range weather
- **Authors:** Ilan Price et al.
- **Issue:** pt-004: Issue detected
- **Explanation:** Missing optimizer.zero_grad(): Gradients accumulate across batches instead of being reset. This causes incorrect updates, exploding gradients, and NaN losses. Call optimizer.zero_grad() before each backward() in your training loop.

- **Suggestion:** Review the code and fix according to the explanation.

```python
total_epochs = config["num_epochs"] + 1
    while epoch < total_epochs:
        util.train_one_epoch(
            epoch,
            model,
            trainloader,
```

**pt-004** (critical, 95% confidence)

- **Repo:** rose-stl-lab__spherical-dyffusion (earth_science)
- **Location:** function `train_one_epoch` (line 240)
- **Paper:** Probabilistic Emulation of a Global Climate Model with Spherical DYffusion
- **Authors:** Salva Rühling Cachay et al.
- **Issue:** pt-004: Issue detected
- **Explanation:** Missing optimizer.zero_grad(): Gradients accumulate across batches instead of being reset. This causes incorrect updates, exploding gradients, and NaN losses. Call optimizer.zero_grad() before each backward() in your training loop.

- **Suggestion:** Review the code and fix according to the explanation.

```python
self.config.clean_wandb()

    def train_one_epoch(self):
        """Train for one epoch and return logs from TrainAggregator."""
        wandb = WandB.get_instance()
        aggregator = TrainAggregator()
        if self.num_batches_seen == 0:
```

### scientific-numerical

**py-001** (medium, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** method `runs` (line 874)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** py-001: Issue detected
- **Explanation:** Mutable default argument: the default object is shared across all calls. Use None as default and create inside the function.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def runs(self, seeds: List[int] = [42,233,666,19971203,19980224]):
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return 
        if self._use_wandb():
            wandb.config.update({"seeds": seeds})

        
        results = []
        for i, seed in enumerate(seeds):
            self.current_run = i
            if self._use_wandb():
                wandb.run.summary["at_run"] = i
            result = self.run(seed=seed)
            results.append(result)

            if self._use_wandb():
                for name, metric_value in result.items():
                    wandb.run.summary["test_" + name] = metric_value

        df = pd.DataFrame(results)
        self.metric_mean_std = df.agg(["mean", "std"]).T
        print(
            self.metric_mean_std.apply(
                lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
            )
        )
        if self._use_wandb():
            for index, row in self.metric_mean_std.iterrows():
                wandb.run.summary[f"{index}_mean"] = row["mean"]
                wandb.run.summary[f"{index}_std"] = row["std"]
                wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"
        wandb.finish()
```

**num-005** (high, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** method `_init_data_loader` (line 255)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** num-005: Issue detected
- **Explanation:** Division by zero in normalization: std is 0 for constant features, producing inf or NaN. Check for zero std and handle it.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def _init_data_loader(self):
        self.dataset : TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = ChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            scale_in_train=False,
            shuffle_train=True,
            freq="h",
            batch_size=self.batch_size,
            train_ratio=0.7,
            val_ratio=0.2,      
            num_worker=self.num_worker,
        )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")
```

**py-001** (medium, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** function `runs` (line 1050)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** py-001: Issue detected
- **Explanation:** Mutable default argument: the default object is shared across all calls. Use None as default and create inside the function.


- **Suggestion:** Review the code and fix according to the explanation.

```python
return self._test()

    def runs(self, seeds: List[int] = [42,43,44]):
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return
```

### scientific-performance

**perf-004** (high, 95% confidence)

- **Repo:** songtaoliu0823__crebm (chemistry)
- **Location:** function `convert_symbols_to_inputs` (line 178)
- **Paper:** Preference Optimization for Molecule Synthesis with Conditional Residual Energy-based Models
- **Authors:** Songtao Liu et al.
- **Issue:** perf-004: Issue detected
- **Explanation:** Materializing large intermediate arrays wastes memory. Use np.linalg.norm or in-place operations to avoid temporary allocations.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def convert_symbols_to_inputs(input_list, output_list, max_length):
    num_samples = len(input_list)
          
    input_ids = np.zeros((num_samples, max_length))
    input_mask = np.zeros((num_samples, max_length))

           
    output_ids = np.zeros((num_samples, max_length))
    output_mask = np.zeros((num_samples, max_length))

               
    token_ids = np.zeros((num_samples, max_length))
    token_mask = np.zeros((num_samples, max_length))

    for cnt in range(num_samples):
        input_ = '^' + input_list[cnt] + '$'
        output_ = '^' + output_list[cnt] + '$'
        
        for i, symbol in enumerate(input_):
            input_ids[cnt, i] = char_to_ix[symbol]
        input_mask[cnt, :len(input_)] = 1

        for i in range(len(output_)-1):
            output_ids[cnt, i] = char_to_ix[output_[i]]
            token_ids[cnt, i] = char_to_ix[output_[i+1]]
            if i != len(output_)-2:
                token_mask[cnt, i] = 1
        output_mask[cnt, :len(output_)-1] = 1
    return (input_ids, input_mask, output_ids, output_mask, token_ids, token_mask)
```

**perf-004** (high, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** function `_process_batch` (line 81)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** perf-004: Issue detected
- **Explanation:** Materializing large intermediate arrays wastes memory. Use np.linalg.norm or in-place operations to avoid temporary allocations.


- **Suggestion:** Review the code and fix according to the explanation.

```python
dec_inp_pred = torch.zeros(
```

**perf-004** (high, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** method `_train` (line 731)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** perf-004: Issue detected
- **Explanation:** Materializing large intermediate arrays wastes memory. Use np.linalg.norm or in-place operations to avoid temporary allocations.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def _sep_train(self):
        with torch.enable_grad(), tqdm(total=self.train_steps,position=0, leave=True) as progress_bar:
            self.model.train()
                                  
                                        
            times = []
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device)
                self.n_model_optim.zero_grad()
                self.f_model_optim.zero_grad()
                bs = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                start = time.time()

                pred, true = self._process_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                
                loss = self.loss_func(pred, true)
                lossn = self.model.nm.loss(true)

                
                loss.backward(retain_graph=True)
                lossn.backward(retain_graph=True)

                                                 
                                                                 
                   
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.f_model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

                self.n_model_optim.step()
                self.f_model_optim.step()
                
                
                end = time.time()
                times.append(end-start)
                
            print("average iter: {}ms", np.mean(times)*1000)
                
            return train_loss
```

### scientific-reproducibility

**rep-002** (high, 95% confidence)

- **Repo:** samgoldman97__ms-pred (chemistry)
- **Location:** function `train_model` (line 80)
- **Paper:** MassFormer: Tandem Mass Spectrum Prediction for Small Molecules using Graph Transformers
- **Authors:** Adamo Young et al.
- **Issue:** rep-002: Issue detected
- **Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="dag_inten_train.log", debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

               
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

                 
                                                  
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]
    split_file = data_dir / "splits" / kwargs["split_name"]

                               
    df = pd.read_csv(labels, sep="\t")
    if kwargs["debug"]:
        df = df[:100]
        kwargs["num_workers"] = 0

    spec_names = df["spec"].values
    if kwargs["debug_overfit"]:
        kwargs["warmup"] = 0
        train_inds, val_inds, test_inds = common.get_splits(
            spec_names, split_file, val_frac=0
        )
        train_inds = train_inds[:100]
    else:
        train_inds, val_inds, test_inds = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    num_workers = kwargs.get("num_workers", 0)
    form_dag_folder = Path(kwargs["formula_folder"])
    all_json_pths = [Path(i) for i in form_dag_folder.glob("*.json")]
    name_to_json = {i.stem.replace("pred_", ""): i for i in all_json_pths}
    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k=kwargs["pe_embed_k"])
    atom_feats = graph_featurizer.atom_feats
    bond_feats = graph_featurizer.bond_feats

                             
    train_dataset = scarf_data.IntenDataset(
        train_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        form_map=name_to_json,
        num_workers=num_workers,
        root_embedder=kwargs["root_embedder"],
        binned_targs=kwargs["binned_targs"],
    )
    val_dataset = scarf_data.IntenDataset(
        val_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        form_map=name_to_json,
        num_workers=num_workers,
        root_embedder=kwargs["root_embedder"],
        binned_targs=kwargs["binned_targs"],
    )

    test_dataset = scarf_data.IntenDataset(
        test_df,
        data_dir=data_dir,
        graph_featurizer=graph_featurizer,
        form_map=name_to_json,
        num_workers=num_workers,
        root_embedder=kwargs["root_embedder"],
        binned_targs=kwargs["binned_targs"],
    )
    ex = train_dataset[0]

    persistent_workers = kwargs["num_workers"] > 0

                        
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
        persistent_workers=persistent_workers,
    )

                  
    model = scarf_model.ScarfIntenNet(
        hidden_size=kwargs["hidden_size"],
        gnn_layers=kwargs["gnn_layers"],
        mlp_layers=kwargs["mlp_layers"],
        set_layers=kwargs["set_layers"],
        form_set_layers=kwargs["form_set_layers"],
        formula_dim=common.NORM_VEC.shape[0],
        mpnn_type=kwargs["mpnn_type"],
        dropout=kwargs["dropout"],
        learning_rate=kwargs["learning_rate"],
        weight_decay=kwargs["weight_decay"],
        atom_feats=atom_feats,
        bond_feats=bond_feats,
        pe_embed_k=kwargs["pe_embed_k"],
        pool_op=kwargs["pool_op"],
        num_atom_feats=graph_featurizer.num_atom_feats,
        num_bond_feats=graph_featurizer.num_bond_feats,
        lr_decay_rate=kwargs["lr_decay_rate"],
        loss_fn=kwargs["loss_fn"],
        warmup=kwargs.get("warmup", 1000),
        info_join=kwargs["info_join"],
        root_embedder=kwargs["root_embedder"],
        embedder=kwargs["embedder"],
        binned_targs=kwargs["binned_targs"],
        embed_adduct=kwargs["embed_adduct"],
    )

                                           
                      
                                     
                                         
                                   
                                           
       

                    
    monitor = "val_loss"
    if kwargs["debug"]:
        kwargs["max_epochs"] = 2

    if kwargs["debug_overfit"]:
        kwargs["min_epochs"] = 1000
        kwargs["max_epochs"] = kwargs["min_epochs"]
        kwargs["no_monitor"] = True
        monitor = "train_loss"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = common.ConsoleLogger()

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_path,
        filename="best",                             
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=20)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [earlystop_callback, checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if kwargs["gpu"] else "cpu",
        devices=1 if kwargs["gpu"] else 0,
        callbacks=callbacks,
        gradient_clip_val=5,
        min_epochs=kwargs["min_epochs"],
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
    )

    if kwargs["debug_overfit"]:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

    checkpoint_callback = trainer.checkpoint_callback
    best_checkpoint = checkpoint_callback.best_model_path
    best_checkpoint_score = checkpoint_callback.best_model_score.item()

                          
    model = scarf_model.ScarfIntenNet.load_from_checkpoint(best_checkpoint)
    logging.info(
        f"Loaded model with from {best_checkpoint} with val loss of {best_checkpoint_score}"
    )

    model.eval()
    trainer.test(dataloaders=test_loader)
```

**rep-002** (high, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** method `reproducible` (line 444)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** rep-002: Issue detected
- **Explanation:** CUDA non-determinism: some GPU ops are non-deterministic by default. Set torch.use_deterministic_algorithms(True) and torch.backends.cudnn.benchmark = False.


- **Suggestion:** Review the code and fix according to the explanation.

```python
def reproducible(self, seed):
                             
                                                
        print("torch.get_default_dtype()", torch.get_default_dtype())
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
                                                  
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinstic = True
```

**rep-010** (medium, 95% confidence)

- **Repo:** YukiBear426__AEFIN (economics)
- **Location:** method `_run_print` (line 159)
- **Paper:** Non-Stationary Time Series Forecasting Based on Fourier Analysis and Cross Attention Mechanism
- **Authors:** Yuqi Xiong, Yang Wen
- **Issue:** rep-010: Issue detected
- **Explanation:** Naive datetime depends on system timezone. Use datetime.now(timezone.utc) for reproducible timestamps.

- **Suggestion:** Review the code and fix according to the explanation.

```python
def _run_print(self, *args, **kwargs):
        time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
        
        print(*args, **kwargs)
        if hasattr(self, "run_setuped") and getattr(self, "run_setuped") is True:
            with open(os.path.join(self.run_save_dir, 'output.log'), 'a+') as f:
                print(time, *args, flush=True, file=f)
```

---

*Analysis conducted: 2026-03-16 | Report generated: 2026-03-16 18:01*
