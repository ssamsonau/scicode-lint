def train_multi_task(model, data_loader, optimizer, criterion_a, criterion_b):
    model.train()
    for batch in data_loader:
        inputs, targets_a, targets_b = batch

        out_a, out_b = model(inputs)
        loss_a = criterion_a(out_a, targets_a)
        loss_b = criterion_b(out_b, targets_b)

        loss_a.backward(retain_graph=True)
        loss_b.backward()
        optimizer.step()
