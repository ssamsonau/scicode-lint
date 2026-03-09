import torch


def create_compiled_models(model_class, *args, **kwargs):
    """Create separate compiled models for training and inference modes."""
    train_model = model_class(*args, **kwargs)
    train_model.train()
    compiled_train = torch.compile(train_model)

    eval_model = model_class(*args, **kwargs)
    eval_model.eval()
    compiled_eval = torch.compile(eval_model)

    return compiled_train, compiled_eval


def train_with_separate_compile(model, train_loader, val_loader):
    """Uses separate compiled models for training and validation."""
    train_compiled = torch.compile(model)
    train_compiled.train()

    # Training
    for batch in train_loader:
        loss = train_compiled(batch).sum()
        loss.backward()

    # Create fresh model for evaluation
    eval_model = type(model)()
    eval_model.load_state_dict(model.state_dict())
    eval_model.eval()
    eval_compiled = torch.compile(eval_model)

    # Evaluation
    with torch.no_grad():
        for batch in val_loader:
            _ = eval_compiled(batch)


def fullgraph_false_mode_switch(model):
    """Using fullgraph=False allows safe mode switching."""
    compiled = torch.compile(model, fullgraph=False)

    compiled.train()
    out_train = compiled(torch.randn(1, 64))

    compiled.eval()
    with torch.no_grad():
        out_eval = compiled(torch.randn(1, 64))

    return out_train, out_eval
