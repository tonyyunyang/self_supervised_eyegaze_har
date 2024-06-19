import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter


def train_fully_supervised_pretrain_model(model, loss, optimizer, scheduler, loaders, model_config, config,
                                          leave_out_subject):
    train_set = loaders[0]
    val_set = loaders[1]
    # test_set = loaders[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join(config['result_path'], config['data_type'])

    path = os.path.join(path,
                        f"feat_dim_{config['kdd_model']['feat_dim']}_d_model_{config['kdd_model']['d_model']}_"
                        f"n_heads_{config['kdd_model']['n_heads']}_n_layers_{config['kdd_model']['n_layers']}_"
                        f"d_ff_{config['kdd_model']['dim_feedforward']}_emb_dropout_{config['kdd_model']['emb_dropout']}_"
                        f"enc_dropout_{config['kdd_model']['enc_dropout']}_embedding_{config['kdd_model']['embedding']}_"
                        f"conv_config_{config['kdd_model']['conv_config']}")

    path = os.path.join(path, f"epochs_{config['pretrain_epoch']}_max_update_steps_{config['pretrain_max_update_epochs']}_"
                              f"warmup_steps_{config['pretrain_warmup_epochs']}_batch_size_{config['pretrain_batch_size']}_"
                              f"base_lr_{format(config['pretrain_base_lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"final_lr_{format(config['pretrain_final_lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"label_smoothing_{format(config['label_smoothing'], '.10f').rstrip('0').rstrip('.')}")

    path = os.path.join(path, f"{leave_out_subject}_leave_out")

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

    # best_val_acc = 0  # Initialize variable to keep track of the best validation accuracy
    #
    # for epoch in range(1, config["pretrain_epoch"] + 1):
    #     writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    #
    #     train_loss, val_loss, val_acc, val_f1 = pass_epoch(model, loss, optimizer, scheduler, train_set, val_set, config, device, epoch)
    #     train_loss_list.append(train_loss)
    #     val_loss_list.append(val_loss)
    #     val_accuracy_list.append(val_acc)
    #     val_f1_score_list.append(val_f1)
    #
    #     test_acc, test_f1 = force_eval_model(model, test_set, config, device)
    #     writer.add_scalar('Force_Test/Accuracy', test_acc, epoch)
    #     writer.add_scalar('Force_Test/F1_Score', test_f1, epoch)
    #
    #     # Log training and validation metrics
    #     writer.add_scalar('Train_Loss/train', train_loss, epoch)
    #     writer.add_scalar('Val_Loss/val', val_loss, epoch)
    #     writer.add_scalar('Accuracy/val', val_acc, epoch)
    #     writer.add_scalar('F1_Score/val', val_f1, epoch)
    #
    #     # # Save the best model based on validation accuracy
    #     # if val_acc >= best_val_acc:
    #     #     best_val_acc = val_acc
    #     #     torch.save(
    #     #         model.state_dict(), os.path.join(config["general"]["finetune_model"], "best_model.pth")
    #     #     )
    #
    #     # Save the best model based on f1 accuracy
    #     if val_f1 >= best_val_f1:
    #         best_val_f1 = val_f1
    #         torch.save(
    #             model.state_dict(), os.path.join(config["model_path"], "best_model.pth")
    #         )
    #
    # # Save the model for continuing the training
    # torch.save(
    #     model.state_dict(), os.path.join(config["model_path"], "last_model.pth")
    # )
    #
    # save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)
    #
    # del model
    # torch.cuda.empty_cache()
    # gc.collect()

    return writer