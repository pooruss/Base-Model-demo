import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import bmtrain as bmt
# import wandb
import numpy as np
import time


class Trainer():
    def __init__(self, model, train_config):
        self.model = model
        self.model_name = train_config["model_name"]
        self.neg_nums = train_config["neg_nums"]
        self.do_train = train_config["do_train"]
        self.do_test = train_config["do_test"]
        self.mrm = train_config["mrm"]
        self.use_cuda = train_config["use_cuda"]
        self.gpu_ids = train_config["gpu_ids"]
        self.gpus = len(self.gpu_ids)
        self.node = train_config["node"]
        self.bmtrain = train_config["bmtrain"]
        if self.bmtrain:
            self.device = torch.device('cuda:{}'.format(bmt.rank()))
            print(bmt.rank())
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]) if self.use_cuda else 'cpu')

        self.batch_size = train_config["batch_size"]
        self.vocab_size = train_config["vocab_size"]
        self.epoch = train_config["epoch"]
        self.learning_rate = train_config["learning_rate"]
        self.save_path = train_config["save_path"]

        self.checkpoint_num = train_config["checkpoint_num"]
        self.log_file = open(os.path.join(self.save_path, 'train.log'), 'w', encoding='utf-8')
        self.task_name = train_config["task_name"]
        self.pad_id = train_config["pad_id"]
        self.mask_id = int(train_config["mask_id"])
        self.loss_function = None
        self.pretrained_optimizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.pretrained_lr_scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.max_seq_len = 187
        self.save = False

    def set_pretrained_optimizer(self, optimizer):
        self.pretrained_optimizer = optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_pretrained_lr_scheduler(self, lr_scheduler):
        self.pretrained_lr_scheduler = lr_scheduler

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def load_train_data_loader(self, train_data_loader):
        self.train_data_loader = train_data_loader

    def load_val_data_loader(self, val_data_loader):
        self.val_data_loader = val_data_loader

    def prediction(self):
        pass

    def validation(self):
        self.model.eval()
        mlm_loss_sum, mem_loss_sum, mlm_total_acc, mem_total_acc = 0.0, 0.0, 0.0, 0.0
        print("----------- validation -----------")
        self.log_file.write(str(len(self.val_data_loader)) + '\n')
        for iter, batch_data in enumerate(self.val_data_loader):
            # fetch batch data
            with torch.no_grad():
                try:
                    src_ids, input_mask, mlm_label_ids, mem_label_ids = batch_data
                except RuntimeError:
                    print("One data instance's length should be 5, received {}.".format(len(batch_data)))
                    continue

                if self.use_cuda:
                    src_ids, input_mask, mlm_label_ids, mem_label_ids = \
                        src_ids.to(self.device), input_mask.to(self.device), mlm_label_ids.to(
                            self.device), mem_label_ids.to(self.device)

                    mlm_mask_pos = torch.argwhere(mlm_label_ids.reshape(-1) > self.pad_id)
                    mlm_label_ids = mlm_label_ids.view(-1)[mlm_mask_pos].squeeze()
                    mem_mask_pos = torch.argwhere(mem_label_ids.reshape(-1) > self.pad_id)
                    mem_label_ids = mem_label_ids.view(-1)[mem_mask_pos].squeeze()

                input_x = {
                    'src_ids': src_ids,
                    'input_mask': input_mask
                }

                # forward
                logits = self.model(input_x)["logits"]
                mlm_last_hidden_state = logits.view(-1, self.vocab_size)[mlm_mask_pos.view(-1)]
                mem_last_hidden_state = logits.view(-1, self.vocab_size)[mem_mask_pos.view(-1)]
                mlm_acc = mlm_last_hidden_state.max(dim=1)[1].eq(mlm_label_ids.squeeze()).sum()
                mlm_total_acc += (mlm_acc * 100 / mlm_label_ids.size(0))
                mem_acc = mem_last_hidden_state.max(dim=1)[1].eq(mem_label_ids.squeeze()).sum()
                mem_total_acc += (mem_acc * 100 / mem_label_ids.size(0))
                # calc mlm_loss
                mlm_loss = self.loss_function(
                    input=mlm_last_hidden_state,
                    target=mlm_label_ids.squeeze()
                )
                mem_label_ids = mem_label_ids.view(-1, 1).squeeze()
                mem_loss = self.loss_function(
                    input=mem_last_hidden_state,
                    target=mem_label_ids
                )

                if self.bmtrain:
                    loss = (mlm_loss + mem_loss)
                    mlm_global_loss = bmt.sum_loss(mlm_loss).item()
                    mem_global_loss = bmt.sum_loss(mem_loss).item()
                    mlm_loss_sum += mlm_global_loss
                    mem_loss_sum += mem_global_loss
                else:
                    mlm_loss_sum += mlm_loss.data
                    mem_loss_sum += mem_loss.data

        mlm_total_acc = mlm_total_acc / float(len(self.val_data_loader))
        mem_total_acc = mem_total_acc / float(len(self.val_data_loader))
        mlm_loss_sum = mlm_loss_sum / len(self.val_data_loader)
        mem_loss_sum = mem_loss_sum / len(self.val_data_loader)
        if self.bmtrain:
            bmt.print_rank('mlm_loss:{}, acc:{}%, mem_loss:{}, acc:{}%.'.format(
                str(mlm_loss_sum), str(mlm_total_acc.cpu().detach().numpy()), str(mem_loss_sum),
                str(mem_total_acc.cpu().detach().numpy())))
            # bmt.synchronize()
        else:
            print('mlm_loss:{}, acc:{}%, mem_loss:{}, acc:{}%.'.format(
                str(mlm_loss_sum), str(mlm_total_acc), str(mem_loss_sum), str(mem_total_acc)))
        # wandb.log({"validation loss": loss_sum})
        self.log_file.write(('mlm_loss:{}, acc:{}%, mem_loss:{}, acc:{}%.'.format(
            str(mlm_loss_sum), str(mlm_total_acc.cpu().detach().numpy()), str(mem_loss_sum),
            str(mem_total_acc.cpu().detach().numpy()))) + '\n')

        self.model.train()

    def train(self):
        self.log_file.write("LR:{}".format(str(self.learning_rate)) + '\n')
        self.log_file.write("BATCHSIZE:{}".format(str(self.batch_size)) + '\n')
        self.log_file.write("TASK:{}".format(self.task_name) + '\n')
        self.log_file.write("MRM:{}".format(str(self.mrm)) + '\n')
        # for name, params in self.model.named_parameters():
        #     self.log_file.write(name + '\n')
        self.model.train()

        step_per_epoch = len(self.train_data_loader)
        total_train_step = step_per_epoch * self.epoch
        save_step = math.ceil(total_train_step / self.node / self.gpus / self.checkpoint_num)
        for epoch in range(self.epoch):
            neg_batch = [i for i in range(self.neg_nums * self.batch_size) if i % self.neg_nums == 0]
            neg_batch = torch.tensor(np.array(neg_batch), dtype=torch.long).to(self.device)
            for i in range(self.neg_nums - 1):
                neg_batch_n = neg_batch + i + 1
                neg_batch = torch.cat((neg_batch, neg_batch_n), dim=0)

            start_time = time.time()

            for iter, batch_data in enumerate(self.train_data_loader):
                # fetch batch data
                if self.neg_nums > 0:
                    src_ids, input_mask, mlm_label_ids, mem_label_ids, pos_neg_seq = batch_data
                    src_ids, input_mask, mlm_label_ids, mem_label_ids, pos_neg_seq = \
                        src_ids.to(self.device), input_mask.to(self.device), mlm_label_ids.to(self.device), \
                        mem_label_ids.to(self.device), pos_neg_seq.to(self.device)
                else:
                    src_ids, input_mask, mlm_label_ids, mem_label_ids = batch_data
                    src_ids, input_mask, mlm_label_ids, mem_label_ids = \
                        src_ids.to(self.device), input_mask.to(self.device), mlm_label_ids.to(
                            self.device), mem_label_ids.to(self.device)
                with torch.no_grad():
                    mlm_mask_pos = (mlm_label_ids > self.pad_id)
                    mem_mask_pos = (mem_label_ids > self.pad_id)
                mlm_label_ids = mlm_label_ids[mlm_mask_pos].squeeze()
                mem_label_ids = mem_label_ids[mem_mask_pos].squeeze()

                input_x = {
                    'src_ids': src_ids,
                    'pos_neg_seq': pos_neg_seq,
                    'input_mask': input_mask,
                    'mem_mask_pos': mem_mask_pos.to(self.device),
                    'neg_nums': self.neg_nums,
                    'neg_batch': neg_batch,
                    'batch_size': src_ids.size(0)
                }
                # forward
                output_map = self.model(input_x)
                logits = output_map["logits"]

                # calc mlm_loss
                mlm_last_hidden_state = logits[mlm_mask_pos]
                mem_last_hidden_state = logits[mem_mask_pos]
                mlm_loss = self.loss_function(
                    input=mlm_last_hidden_state.half(),
                    target=mlm_label_ids.long()
                )
                mem_loss = self.loss_function(
                    input=mem_last_hidden_state.half(),
                    target=mem_label_ids.long()
                )

                neg_pos_logits = output_map["neg_pos_logits"]
                neg_len = int(len(neg_pos_logits) / 2)
                labels = torch.ones(neg_len).to(self.device)
                margin_lossfct = nn.MarginRankingLoss(margin=4.0)
                margin_loss = margin_lossfct(
                    neg_pos_logits[neg_len:].squeeze(),
                    neg_pos_logits[:neg_len].squeeze(),
                    labels
                )

                mlm_acc = mlm_last_hidden_state.max(dim=1)[1].eq(mlm_label_ids).sum()
                mlm_acc = (mlm_acc * 100 / mlm_label_ids.size(0))
                mem_acc = mem_last_hidden_state.max(dim=1)[1].eq(mem_label_ids).sum()
                mem_acc = (mem_acc * 100 / mem_label_ids.size(0))

                loss = mlm_loss + mem_loss + margin_loss
                # backward
                if self.bmtrain:
                    loss = self.optimizer.loss_scale(loss)
                    loss.mean().backward()
                    lr = self.lr_scheduler.get_lr()
                    bmt.optim_step(self.optimizer)
                    self.optimizer.zero_grad()
                else:
                    loss.mean().backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr = self.lr_scheduler.get_last_lr()[0]

                current_step = epoch * step_per_epoch + iter
                if iter % 100 == 0:
                    if not self.bmtrain:
                        print('Epoch:{}, Step:{}/{}, mlm_loss:{}, mlm_acc:{}%, mem_loss:{}, mem_acc:{}, neg_loss:{}, '
                              'lr:{}.'.format(
                            str(epoch), str(current_step),
                            str(total_train_step),
                            str(mlm_loss.cpu().detach().numpy()),
                            str(mlm_acc.cpu().detach().numpy()),
                            str(mem_loss.cpu().detach().numpy()),
                            str(mem_acc.cpu().detach().numpy()),
                            str(margin_loss.cpu().detach().numpy()),
                            str(lr)
                        )
                        )
                    else:
                        bmt.print_rank(
                            'Epoch:{}, Step:{}/{}, mlm_loss:{}, mlm_acc:{}%, mem_loss:{}, mem_acc:{}, neg_loss:{}, '
                            'lr:{}.'.format(
                                str(epoch), str(current_step),
                                str(total_train_step),
                                str(mlm_loss.cpu().detach().numpy()),
                                str(mlm_acc.cpu().detach().numpy()),
                                str(mem_loss.cpu().detach().numpy()),
                                str(mem_acc.cpu().detach().numpy()),
                                str(margin_loss.cpu().detach().numpy()),
                                str(lr)
                            )
                        )

                    self.log_file.write(
                        'Epoch:{}, Step:{}/{}, mlm_loss:{}, mlm_acc:{}%, mem_loss:{}, mem_acc:{}, neg_loss:{}, '
                        'lr:{}.'.format(
                            str(epoch), str(current_step),
                            str(total_train_step),
                            str(mlm_loss.cpu().detach().numpy()),
                            str(mlm_acc.cpu().detach().numpy()),
                            str(mem_loss.cpu().detach().numpy()),
                            str(mem_acc.cpu().detach().numpy()),
                            str(margin_loss.cpu().detach().numpy()),
                            str(lr)
                        ) + '\n')
                    self.log_file.flush()

                if iter % save_step == 0 and iter != 0:
                    if isinstance(self.model, torch.nn.DataParallel):
                        torch.save(self.model.module.state_dict(),
                                   self.save_path + '{}_lr{}_bs{}_epoch{}.pt'.format(
                                       self.model_name, self.learning_rate,
                                       self.batch_size, str(epoch)))
                    elif self.bmtrain:
                        bmt.save(self.model, self.save_path + '{}_bmt_lr{}_bs{}_epoch{}.pt'.format(
                            self.model_name, self.learning_rate,
                            self.batch_size, str(epoch)))
                    else:
                        torch.save(self.model.state_dict(), self.save_path + '{}_lr{}_bs{}_epoch{}.pt'.format(
                            self.model_name, self.learning_rate,
                            self.batch_size, str(epoch)))
                    self.validation()

            self.lr_scheduler.step() if not self.bmtrain else bmt.optim_step(self.lr_scheduler)

            epoch_time = time.time() - start_time
            if self.bmtrain:
                bmt.print_rank("time:" + str(epoch_time))
            else:
                print("time:" + str(epoch_time))

        return
