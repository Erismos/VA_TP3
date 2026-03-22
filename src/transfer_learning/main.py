import numpy as np
import torch
import json

from train import train_model

CONFIGS = [
    {"lr": 0.001, "optimizer": "adam", "batch_size": 32, "fine_tune": False},
    {"lr": 0.0001, "optimizer": "adam", "batch_size": 32, "fine_tune": False},
    {"lr": 0.001,  "optimizer": "sgd",  "batch_size": 32, "fine_tune": False},
    {"lr": 0.001,  "optimizer": "adam", "batch_size": 64, "fine_tune": False},
    {"lr": 0.01,   "optimizer": "sgd",  "batch_size": 64, "fine_tune": False},

    {"lr": 0.001, "optimizer": "adam", "batch_size": 32, "fine_tune": True},
    {"lr": 0.0001, "optimizer": "adam", "batch_size": 32, "fine_tune": True},
    {"lr": 0.001,  "optimizer": "sgd",  "batch_size": 32, "fine_tune": True},
    {"lr": 0.001,  "optimizer": "adam", "batch_size": 64, "fine_tune": True},
    {"lr": 0.01,   "optimizer": "sgd",  "batch_size": 64, "fine_tune": True},

]

if __name__ == "__main__":
    all_val_accs = []


    for run, config in enumerate(CONFIGS):
        print(f"\nRUN {run+1} — lr={config['lr']}, optimizer={config['optimizer']}, batch_size={config['batch_size']}")

        model, train_losses, val_losses, train_accs, val_accs = train_model(
            lr=config["lr"],
            optimizer_name=config["optimizer"],
            batch_size=config["batch_size"],
            fine_tune=config["fine_tune"]
        )

        torch.save(model.state_dict(), f"results/transfer_model_run_{run+1}.pth")

        history = {
            "config": config,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
        }
        with open(f"results/history_run_{run+1}.json", "w") as f:
            json.dump(history, f, indent=2)

        all_val_accs.append(val_accs[-1])

    mean_acc = np.mean(all_val_accs)
    std_acc = np.std(all_val_accs)

    print("\nFINAL RESULTS")
    print(f"Mean Validation Accuracy: {mean_acc:.3f}")
    print(f"Std Validation Accuracy:  {std_acc:.3f}")