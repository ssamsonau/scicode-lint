import torch


def resume_training(model, checkpoint_path, optimizer, train_loader):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    model.train()

    for epoch in range(checkpoint["epoch"], 100):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
