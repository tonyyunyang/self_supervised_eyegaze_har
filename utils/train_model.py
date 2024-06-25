import gc
import json
import os

import h5py
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter


def train_fully_supervised_pretrain_model(model, loss, optimizer, scheduler, loaders, model_config, config,
                                          leave_out_subject):
    train_set = loaders[0]
    val_set = loaders[1]
    test_set = loaders[2]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join(config['result_path'], config['data_type'])

    path = os.path.join(path, "fully_supervised")

    path = os.path.join(path,
                        f"feat_dim_{config['kdd_model']['feat_dim']}_d_model_{config['kdd_model']['d_model']}_"
                        f"n_heads_{config['kdd_model']['n_heads']}_n_layers_{config['kdd_model']['n_layers']}_"
                        f"d_ff_{config['kdd_model']['dim_feedforward']}_emb_dropout_{config['kdd_model']['emb_dropout']}_"
                        f"enc_dropout_{config['kdd_model']['enc_dropout']}_embedding_{config['kdd_model']['embedding']}_"
                        f"conv_config_{config['kdd_model']['conv_config']}")

    path = os.path.join(path,
                        f"epochs_{config['pretrain_epoch']}_max_update_steps_{config['pretrain_max_update_epochs']}_"
                        f"warmup_steps_{config['pretrain_warmup_epochs']}_batch_size_{config['pretrain_batch_size']}_"
                        f"base_lr_{format(config['pretrain_base_lr'], '.10f').rstrip('0').rstrip('.')}_"
                        f"final_lr_{format(config['pretrain_final_lr'], '.10f').rstrip('0').rstrip('.')}_"
                        f"label_smoothing_{format(config['label_smoothing'], '.10f').rstrip('0').rstrip('.')}")

    path = os.path.join(path, f"{leave_out_subject}_leave_out")

    config["model_path"] = path

    os.makedirs(path, exist_ok=True)

    log_dir = os.path.join(path, "TensorBoard_Log")
    writer = SummaryWriter(log_dir=log_dir)

    print(f'Run cmd: tensorboard --logdir={log_dir} then open http://localhost:6006')

    with open(os.path.join(path, 'kdd_model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_score_list = []

    best_val_acc = 0  # Initialize variable to keep track of the best validation accuracy
    best_val_f1 = 0

    for epoch in range(1, config["pretrain_epoch"] + 1):
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        train_loss, val_loss, val_acc, val_f1 = pass_epoch(model, loss, optimizer, scheduler, train_set, val_set,
                                                           device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        test_acc, test_f1 = force_eval_model(model, test_set, device)
        writer.add_scalar('Force_Test/Accuracy', test_acc, epoch)
        writer.add_scalar('Force_Test/F1_Score', test_f1, epoch)

        # Log training and validation metrics
        writer.add_scalar('Train_Loss/train', train_loss, epoch)
        writer.add_scalar('Val_Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1_Score/val', val_f1, epoch)

        # # Save the best model based on validation accuracy
        # if val_acc >= best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(
        #         model.state_dict(), os.path.join(config["general"]["finetune_model"], "best_model.pth")
        #     )

        # Save the best model based on f1 accuracy
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(), os.path.join(config["model_path"], "best_model.pth")
            )

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["model_path"], "last_model.pth")
    )

    save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config["pretrain_epoch"], config)

    del model
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    return writer


def pass_epoch(model, loss, optimizer, scheduler, train_set, val_set, device):
    # Train the model first
    model = model.train()
    train_loss = 0
    total_train_samples = 0

    for batch in train_set:
        mvts_inputs, labels = batch
        mvts_inputs, labels = mvts_inputs.to(device), labels.to(device)

        outputs = model(mvts_inputs)
        computed_loss = loss(outputs, labels)

        optimizer.zero_grad()
        computed_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            train_loss += computed_loss.item() * mvts_inputs.size(0)
            total_train_samples += mvts_inputs.size(0)

    if scheduler:
        scheduler.step()

    train_loss /= total_train_samples

    model = model.eval()
    val_loss = 0
    total_val_samples = 0
    true_labels, pred_labels = [], []

    for batch in val_set:
        mvts_inputs, labels = batch
        mvts_inputs, labels = mvts_inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(mvts_inputs)
            computed_loss = loss(outputs, labels)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(outputs.argmax(dim=1).cpu().numpy())

            val_loss += computed_loss.item() * mvts_inputs.size(0)
            total_val_samples += mvts_inputs.size(0)

    val_loss /= total_val_samples
    val_acc = accuracy_score(true_labels, pred_labels)
    val_f1 = f1_score(true_labels, pred_labels, average='weighted')

    return train_loss, val_loss, val_acc, val_f1


def force_eval_model(model, test_set, device):
    model.eval()

    # Evaluate the model on the test set
    true_labels = []
    pred_labels = []

    for batch in test_set:
        mvts_inputs, labels = batch
        mvts_inputs, labels = mvts_inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(mvts_inputs)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(outputs.argmax(dim=1).cpu().numpy())

    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average='weighted')

    return test_acc, test_f1


def save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, epochs, config):
    # Set the epochs for the x-axis
    epochs = range(1, epochs + 1)

    # Plot and save the loss metrics
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, train_loss_list, label='Train Loss', linestyle='-')
    plt.plot(epochs, val_loss_list, label='Validation Loss', linestyle='-')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["model_path"], "loss_plot.png"))

    # Clear the current figure's context for the next plot
    plt.clf()

    # Plot and save the accuracy metrics
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, val_accuracy_list, label='Validation Accuracy', linestyle='-')
    plt.plot(epochs, val_f1_score_list, label='Validation F1 Score', linestyle='-')
    plt.title('Validation Accuracy and F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["model_path"], "accuracy_f1_plot.png"))


def train_self_supervised_pretrain_model(model, loss, optimizer, scheduler, loaders, model_config, config,
                                          leave_out_subject):
    train_set = loaders[0]
    val_set = loaders[1]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join(config['result_path'], config['data_type'])

    path = os.path.join(path, "self_supervised")

    path = os.path.join(path,
                        f"feat_dim_{config['kdd_model']['feat_dim']}_d_model_{config['kdd_model']['d_model']}_"
                        f"n_heads_{config['kdd_model']['n_heads']}_n_layers_{config['kdd_model']['n_layers']}_"
                        f"d_ff_{config['kdd_model']['dim_feedforward']}_emb_dropout_{config['kdd_model']['emb_dropout']}_"
                        f"enc_dropout_{config['kdd_model']['enc_dropout']}_embedding_{config['kdd_model']['embedding']}_"
                        f"conv_config_{config['kdd_model']['conv_config']}")

    path = os.path.join(path,
                        f"epochs_{config['pretrain_epoch']}_max_update_steps_{config['pretrain_max_update_epochs']}_"
                        f"warmup_steps_{config['pretrain_warmup_epochs']}_batch_size_{config['pretrain_batch_size']}_"
                        f"base_lr_{format(config['pretrain_base_lr'], '.10f').rstrip('0').rstrip('.')}_"
                        f"final_lr_{format(config['pretrain_final_lr'], '.10f').rstrip('0').rstrip('.')}_"
                        f"label_smoothing_{format(config['label_smoothing'], '.10f').rstrip('0').rstrip('.')}")

    path = os.path.join(path, f"{leave_out_subject}_leave_out")

    config["model_path"] = path

    os.makedirs(path, exist_ok=True)

    log_dir = os.path.join(path, "TensorBoard_Log")
    writer = SummaryWriter(log_dir=log_dir)

    print(f'Run cmd: tensorboard --logdir={log_dir} then open http://localhost:6006')

    with open(os.path.join(path, 'kdd_model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []

    best_val_loss = float('inf')

    for epoch in range(1, config["pretrain_epoch"] + 1):

        # Check if it's the last epoch
        is_last_epoch = epoch == config["pretrain_epoch"]

        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        train_loss, val_loss= pass_imputation_epoch(model, loss, optimizer, scheduler, train_set, val_set, device, is_last_epoch, os.path.join(path, "imputate_result"))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Log training and validation metrics
        writer.add_scalar('Train_Loss/train', train_loss, epoch)
        writer.add_scalar('Val_Loss/val', val_loss, epoch)

        # # Save the best model based on validation accuracy
        # if val_acc >= best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(
        #         model.state_dict(), os.path.join(config["general"]["finetune_model"], "best_model.pth")
        #     )

        # Save the best model based on f1 accuracy
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), os.path.join(config["model_path"], "best_model.pth")
            )

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["model_path"], "last_model.pth")
    )

    save_loss_metrics(train_loss_list, val_loss_list, config["pretrain_epoch"], config)

    del model
    torch.mps.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    return writer


def pass_imputation_epoch(model, loss, optimizer, scheduler, train_set, val_set, device, save_last_epoch_samples=False, save_dir_path=None):
    # Train the model first
    model = model.train()
    train_loss = 0
    total_train_samples = 0

    for batch in train_set:
        original_input, mask, masked_input, indices = batch
        original_input, mask, masked_input = original_input.to(device), mask.to(device), masked_input.to(device)

        outputs = model(masked_input)
        computed_loss = loss(outputs, original_input, mask)

        optimizer.zero_grad()
        computed_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            train_loss += computed_loss.item() * indices.size(0)
            total_train_samples += indices.size(0)

    if scheduler:
        scheduler.step()

    train_loss /= total_train_samples

    model = model.eval()
    val_loss = 0
    total_val_samples = 0
    # Variables to store data for saving
    all_original_inputs, all_masks, all_outputs = [], [], []

    for batch in val_set:
        original_input, mask, masked_input, indices = batch
        original_input, mask, masked_input = original_input.to(device), mask.to(device), masked_input.to(device)

        with torch.no_grad():
            outputs = model(masked_input)
            computed_loss = loss(outputs, original_input, mask)

            val_loss += computed_loss.item() * indices.size(0)
            total_val_samples += indices.size(0)

            # Store outputs for saving
            if save_last_epoch_samples and save_dir_path is not None:
                os.makedirs(save_dir_path, exist_ok=True)
                all_original_inputs.append(original_input.cpu())
                all_masks.append(mask.cpu())
                all_outputs.append(outputs.cpu())

    val_loss /= total_val_samples

    # Saving the validation results as h5 file
    if save_last_epoch_samples and save_dir_path is not None:
        print("Writing the imputation result to file")
        with h5py.File(os.path.join(save_dir_path, 'imputation_result.h5'), 'w') as hf:
            hf.create_dataset('original_input', data=torch.cat(all_original_inputs).numpy())
            hf.create_dataset('mask', data=torch.cat(all_masks).numpy())
            hf.create_dataset('predicted_output', data=torch.cat(all_outputs).numpy())
        print("Finish writing the imputation result to file")

    return train_loss, val_loss


def save_loss_metrics(train_loss_list, val_loss_list, epochs, config):
    # Set the epochs for the x-axis
    epochs = range(1, epochs + 1)

    # Plot and save the loss metrics
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, train_loss_list, label='Train Loss', linestyle='-')
    plt.plot(epochs, val_loss_list, label='Validation Loss', linestyle='-')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["model_path"], "loss_plot.png"))

    # Clear the current figure's context for the next plot
    plt.clf()