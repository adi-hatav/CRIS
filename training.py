import os
import time

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import preprocess
from constants import INPUT_NC
from models import UNetWithAttention
from utils import (
    compute_training_loss,
    log_model_parameters,
    loss_flags_for_epoch,
    plot_metrics,
    save_metrics_to_json,
    save_networks,
    save_random_predictions,
    validate,
)

matplotlib.use("Agg")


def train_one_epoch(model, optimizer, criterion, train_loader, device, metrics, opt):
    start_time = time.time()
    model.train()
    flags = loss_flags_for_epoch(opt)
    batch_losses = {k: [] for k in ["total_loss", "l2_loss", "ssim_loss", "sobel_loss", "focal_loss"]}
    totals = {k: 0.0 for k in batch_losses}
    step_counter = 0

    # Microscopy datasets are small, so we loop many passes per epoch.
    # MRI datasets are large — a single pass suffices.
    n_repeats = 100 if getattr(opt, "domain", "MRI") == "microscopy" else 1
    for _ in range(n_repeats):
        for data_masked, data_gt, _ in train_loader:
            step_counter += 1
            data_masked, data_gt = data_masked.to(device), data_gt.to(device)
            optimizer.zero_grad()
            predicted_slice = model(data_masked)
            g_loss, parts = compute_training_loss(predicted_slice, data_gt, criterion, opt, flags)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            totals["total_loss"] += g_loss.item()
            batch_losses["total_loss"].append(g_loss.item())
            for key, value in parts.items():
                totals[key] += value
                batch_losses[key].append(value)

            if step_counter % 10 == 0:
                print(f"Step {step_counter}, " + ", ".join(f"{k}: {v:.4f}" for k, v in parts.items()))

    train_loader_len = max(len(train_loader) * n_repeats, 1)
    avg = {k: totals[k] / train_loader_len for k in totals}
    for key in batch_losses:
        metrics["batch"][key].extend(batch_losses[key])

    print(f"Epoch Time: {time.time() - start_time:.2f} seconds")
    print("Train Loss: {:.4f}, L2 Loss: {:.4f}".format(avg["total_loss"], avg["l2_loss"]))
    return avg


def train(opt):
    model_dir = opt.main_dir
    device = opt.device
    planes = opt.planes
    opt.epoch = 0

    best_psnr = float("-inf")
    epochs_no_improve = 0
    metrics = {
        "train": {k: [] for k in ["total_loss", "l2_loss", "ssim_loss", "sobel_loss", "focal_loss"]},
        "val": {"loss": [], "psnr": []},
        "batch": {k: [] for k in ["total_loss", "l2_loss", "ssim_loss", "sobel_loss", "focal_loss"]},
    }

    if opt.build_slice_cache:
        print("Building slice caches for planes...")
        for plane in planes:
            preprocess.build_slice_cache(opt, plane)

    dataset_path = opt.dataset_path
    train_dataset = preprocess.PreprocessedDataset(
        [os.path.join(dataset_path, plane, "train") for plane in planes],
        opt,
        random_normalization=opt.domain == "MRI",
        val=False,
    )
    val_dataset = (
        preprocess.PreprocessedDataset(
            [os.path.join(dataset_path, plane, "val") for plane in planes],
            opt,
            random_normalization=False,
            val=True,
        )
        if not opt.use_only_train
        else preprocess.PreprocessedDataset(
            [os.path.join(dataset_path, plane, "train") for plane in planes],
            opt,
            random_normalization=False,
            val=True,
        )
    )

    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    model = UNetWithAttention(
        INPUT_NC, opt.output_nc, base_filters=opt.base_filters, window_size=opt.window_size
    ).to(device)
    log_model_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    if opt.pre_trained:
        print(f"Loading pre-trained model from {opt.pre_trained}")
        model.load_state_dict(
            torch.load(os.path.join(opt.pre_trained, "models", "best_model.pth"))
        )

    if len(opt.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    criterion = nn.MSELoss()

    if opt.pre_trained:
        _, best_psnr = validate(model, criterion, val_loader, device, name="Train")
        print(f"Best PSNR from pre-trained model: {best_psnr:.4f}")
        save_networks(model, "best", model_dir)

    for epoch in range(opt.n_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{opt.n_epochs}\n" + "-" * 25)
        train_losses = train_one_epoch(model, optimizer, criterion, train_loader, device, metrics, opt)

        if epochs_no_improve >= 5 and optimizer.param_groups[0]["lr"] > 5e-7:
            scheduler.step()
            if epochs_no_improve >= 12:
                scheduler.step()

        val_loss, val_psnr = (
            validate(model, criterion, val_loader, device, name="Validation")
            if val_loader is not None
            else validate(model, criterion, train_loader, device, name="Train")
        )

        for key in metrics["train"]:
            metrics["train"][key].append(train_losses.get(key, 0.0))
        metrics["val"]["loss"].append(val_loss)
        metrics["val"]["psnr"].append(val_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_no_improve = 0
            save_networks(model, "best", model_dir)
            print(f"----------New best model with PSNR: {best_psnr:.4f}----------")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= opt.patience:
            print(f"Early stopping triggered after {opt.patience} epochs without improvement.")
            break

        if (epoch + 1) % opt.save_epoch_freq == 0:
            save_networks(model, epoch + 1, model_dir)

        preview_loader = val_loader if val_loader is not None else train_loader
        save_random_predictions(epoch + 1, model, preview_loader, model_dir, device, n_samples=1)
        save_metrics_to_json(metrics, model_dir)
        plot_metrics(metrics, model_dir)
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        opt.epoch += 1

    save_metrics_to_json(metrics, model_dir)
    print("Training completed.")
