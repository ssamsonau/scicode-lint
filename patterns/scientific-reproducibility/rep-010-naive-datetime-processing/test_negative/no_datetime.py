def log_experiment_config(config):
    return {"config": config, "version": "1.0"}


def record_metrics(loss, accuracy, epoch):
    return {"epoch": epoch, "loss": loss, "accuracy": accuracy}


def save_checkpoint(model, optimizer, epoch, path):
    import torch

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


class ExperimentLogger:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.logs = []

    def log(self, metrics):
        self.logs.append(
            {
                "experiment_id": self.experiment_id,
                "step": len(self.logs),
                **metrics,
            }
        )

    def get_logs(self):
        return self.logs
