import torch
import torch.nn.functional as F
import math
import os
import bmtrain as bmt
# import wandb
import numpy as np
import time


def mask_tokens(vocab_size, tokenizer, inputs, mlm_prob):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = torch.from_numpy(np.array(inputs)).clone()
    inputs_tensor = torch.from_numpy(np.array(inputs)).clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    """
    prob_data = torch.full(labels.shape, args.mlm_probability) 
        是生成一个labels一样大小的矩阵,里面的值默认是0.15.
    torch.bernoulli(prob_data),从伯努利分布中抽取二元随机数(0或者1),
        prob_data是上面产生的是一个所有值为0.15(在0和1之间的数),
        输出张量的第i个元素值,将以输入张量的第i个概率值等于1.
        (在这里 即输出张量的每个元素有 0.15的概率为1, 0.85的概率为0. 15%原始数据 被mask住)
    """
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_prob)).bool()
    """
    mask_indices通过bool()函数转成True,False
    下面对于85%原始数据 没有被mask的位置进行赋值为-1
    """
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    """
    对于mask的数据,其中80%是赋值为MASK.
    这里先对所有数据以0.8概率获取伯努利分布值, 
    然后 和maksed_indices 进行与操作,得到Mask 的80%的概率 indice, 对这些位置赋值为MASK 
    """
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # print(inputs_tensor)
    inputs_tensor[indices_replaced] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
    # print(inputs_tensor)
    # 10% of the time, we replace masked input tokens with random word
    """
    对于mask_indices剩下的20% 在进行提取,取其中一半进行random 赋值,剩下一般保留原来值. 
    """
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs_tensor[indices_random] = random_words[indices_random]
    return inputs_tensor, labels


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer():
    def __init__(self, model, train_config):
        self.model = model
        self.model_name = train_config["model_name"]

        self.do_train = train_config["do_train"]
        self.do_val = train_config["do_val"]
        self.do_test = train_config["do_test"]

        self.use_cuda = train_config["use_cuda"]
        self.gpu_ids = train_config["gpu_ids"]
        self.gpus = len(self.gpu_ids)
        self.node = train_config["node"]
        self.bmtrain = train_config["bmtrain"]
        if self.bmtrain:
            self.device = torch.device('cuda:{}'.format(bmt.rank()))
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]) if self.use_cuda else 'cpu')

        self.batch_size = train_config["batch_size"]
        self.vocab_size = train_config["vocab_size"]
        self.epoch = train_config["epoch"]
        self.learning_rate = train_config["learning_rate"]
        self.skip_steps = train_config["skip_steps"]
        self.save_path = train_config["save_path"]
        self.soft_label = train_config["soft_label"]
        self.use_ema = train_config["use_ema"]
        self.ema_decay = train_config["ema_decay"]

        self.checkpoint_num = train_config["checkpoint_num"]
        self.log_file = open(os.path.join(self.save_path, 'train.log'), 'w', encoding='utf-8')

        self.pad_id = train_config["pad_id"]
        self.mask_id = int(train_config["mask_id"])
        self.e_mask_id = int(train_config["e_mask_id"])
        self.loss_function = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        if self.use_ema:
            self.ema_model = EMA(self.model, self.ema_decay)
        self.best_loss = 10000.0
        self.last_loss = 10000.0
        self.save = False

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def load_train_data_loader(self, train_data_loader):
        self.train_data_loader = train_data_loader

    def load_val_data_loader(self, val_data_loader):
        self.val_data_loader = val_data_loader

    def prediction(self):
        pass

    def validation(self, epoch):
        if self.use_ema:
            self.ema_model.apply_shadow()
        self.model.eval()
        acc, loss_sum, total_acc = 0.0, 0.0, 0.0
        step_per_epoch = len(self.val_data_loader)
        print("----------- validation -----------")
        self.log_file.write(str(len(self.val_data_loader)) + '\n')
        for iter, batch_data in enumerate(self.val_data_loader):
            # fetch batch data
            try:
                src_ids, input_mask, seg_ids, pos_ids, mlm_label_ids, \
                mem_label_ids, mem_mask_pos = batch_data
            except RuntimeError:
                print("One data instance's length should be 5, received {}.".format(len(batch_data)))
                continue
            if self.use_cuda:
                src_ids, input_mask, seg_ids, pos_ids, mlm_label_ids, mem_label_ids, mem_mask_pos = \
                    src_ids.to(self.device), input_mask.to(self.device), seg_ids.to(self.device), \
                    pos_ids.to(self.device), mlm_label_ids.to(self.device), mem_label_ids.to(self.device), \
                    mem_mask_pos.to(self.device)

            mlm_mask_pos = torch.argwhere(src_ids.reshape(-1) == self.mask_id)
            mlm_label_ids = mlm_label_ids.view(-1)[mlm_mask_pos]

            input_x = {
                'src_ids': src_ids,
                'input_mask': input_mask,
                'segment_ids': seg_ids,
                'position_ids': pos_ids,
                'mlm_mask_pos': mlm_mask_pos,
                'mem_mask_pos': mem_mask_pos
            }
            torch.cuda.synchronize()
            self.optimizer.zero_grad()

            # forward
            out = self.model(input_x)
            mlm_last_hidden_state = out["mlm_last_hidden_state"]
            mem_last_hidden_state = out["mem_last_hidden_state"]
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
                global_loss = bmt.sum_loss(loss).item()
                loss = self.optimizer.loss_scale(loss)
                loss_sum += global_loss
            else:
                loss_sum += (mlm_loss.data + mem_loss.data)
            acc = mem_last_hidden_state.max(dim=1)[1].eq(mem_label_ids.squeeze()).sum()
            total_acc += (acc * 100 / mem_label_ids.size(0))

        # acc = acc * 100 / float(len(self.val_data_loader.dataset))
        total_acc = total_acc / float(len(self.val_data_loader))
        loss_sum = loss_sum / len(self.val_data_loader)
        if not self.bmtrain:
            loss_sum = loss_sum.cpu().detach().numpy()
        if self.bmtrain:
            bmt.print_rank('loss:{}, acc:{}.'.format(str(loss_sum), str(acc)))
            bmt.synchronize()
        else:
            print('val mem_loss:{}, val acc:{}.'.format(str(mem_loss.cpu().detach().numpy()), str(acc)))
        # wandb.log({"validation loss": loss_sum})
        self.log_file.write(('loss:{}, acc:{}%.'.format(str(loss_sum), str(acc))) + '\n')

        # save best checkpoint
        if self.last_loss <= loss_sum:
            self.save = True
        self.last_loss = loss_sum
        if self.best_loss > loss_sum and self.save:
            self.best_loss = loss_sum
            if self.use_ema:
                self.ema_model.apply_shadow()
            if isinstance(self.model, torch.nn.DataParallel):
                torch.save(self.model.module.state_dict(), self.save_path + '{}_lr{}_bs{}_ema{}_epoch{}.pt'.format(
                    self.model_name, self.learning_rate,
                    self.batch_size, str(self.use_ema), str(epoch)))
            elif self.bmtrain:
                bmt.save(self.model, self.save_path + '{}_bmt_lr{}_bs{}_ema{}_epoch{}.pt'.format(
                    self.model_name, self.learning_rate,
                    self.batch_size, str(self.use_ema), str(epoch)))
            else:
                torch.save(self.model.state_dict(), self.save_path + '{}_lr{}_bs{}_ema{}_epoch{}.pt'.format(
                    self.model_name, self.learning_rate,
                    self.batch_size, str(self.use_ema), str(epoch)))
            if self.use_ema:
                self.ema_model.restore()

        if self.use_ema:
            self.ema_model.restore()

    def train(self):
        self.model.train()
        self.model.float()
        if self.use_ema:
            self.ema_model.register()
        acc, loss_sum = 0.0, 0.0
        step_per_epoch = len(self.train_data_loader)
        total_train_step = step_per_epoch * self.epoch
        save_step = math.ceil(total_train_step / self.node / self.gpus / self.checkpoint_num)
        for epoch in range(self.epoch):
            start_time = time.time()
            # label_smoothing = 0.0 if epoch < 5 else 0.0
            for iter, batch_data in enumerate(self.train_data_loader):
                # fetch batch data
                try:
                    src_ids, input_mask, seg_ids, pos_ids, mlm_label_ids, \
                    mem_label_ids, mem_mask_pos = batch_data
                except RuntimeError:
                    print("One data instance's length should be 5, received {}.".format(len(batch_data)))
                    continue
                if self.use_cuda:
                    src_ids, input_mask, seg_ids, pos_ids, mlm_label_ids, mem_label_ids, mem_mask_pos = \
                        src_ids.to(self.device), input_mask.to(self.device), seg_ids.to(self.device), \
                        pos_ids.to(self.device), mlm_label_ids.to(self.device), mem_label_ids.to(self.device), \
                        mem_mask_pos.to(self.device)
                with torch.no_grad():
                    mlm_mask_pos = torch.argwhere(src_ids.reshape(-1) == self.mask_id)
                    mlm_label_ids = mlm_label_ids.view(-1)[mlm_mask_pos].squeeze()
                    mem_label_ids = mem_label_ids.view(-1, 1).squeeze()
                input_x = {
                    'src_ids': src_ids,
                    'input_mask': input_mask,
                    'segment_ids': seg_ids,
                    'position_ids': pos_ids,
                    'mlm_mask_pos': mlm_mask_pos,
                    'mem_mask_pos': mem_mask_pos
                }
                torch.cuda.synchronize()
                self.optimizer.zero_grad()

                # forward
                out = self.model(input_x)
                mlm_last_hidden_state = out["mlm_last_hidden_state"]
                mem_last_hidden_state = out["mem_last_hidden_state"]
                # calc mlm_loss
                mlm_loss = self.loss_function(
                    input=mlm_last_hidden_state.half(),
                    target=mlm_label_ids.long()
                )

                mem_loss = self.loss_function(
                    input=mem_last_hidden_state.half(),
                    target=mem_label_ids.long()
                )
                # print(mem_last_hidden_state.shape)
                # print(mem_label_ids.shape)
                # print(mem_label_ids)
                # mem_loss = self.loss_function(
                #     input=mem_last_hidden_state,
                #     target=mem_label_ids.squeeze()
                # )

                # backward
                if self.bmtrain:
                    loss = mlm_loss
                    global_loss = bmt.sum_loss(loss).item()
                    loss = self.optimizer.loss_scale(loss)
                    loss.mean().backward()
                    loss = mem_loss
                    global_loss = bmt.sum_loss(loss).item()
                    loss = self.optimizer.loss_scale(loss)
                    loss.backward()
                    bmt.optim_step(self.optimizer)
                    lr = self.lr_scheduler.get_lr()
                    torch.cuda.synchronize()
                else:
                    loss = mlm_loss + mem_loss
                    loss.mean().backward()
                    self.optimizer.step()
                    lr = self.lr_scheduler.get_last_lr()[0]
                if self.use_ema:
                    self.ema_model.update()

                # log
                # print(logits.max(dim=1)[1])
                acc = mem_last_hidden_state.max(dim=1)[1].eq(mem_label_ids.squeeze()).sum()
                acc = acc * 100 / mem_label_ids.size(0)
                current_step = epoch * step_per_epoch + iter

                if iter % (step_per_epoch / 16) == 0 or iter + 1 == step_per_epoch:
                    if not self.bmtrain:
                        print('Epoch:{}, Step:{}/{}, mlm_loss:{}, mem_loss:{}, cls_acc:{}%, lr:{}.'.format(
                            str(epoch), str(current_step),
                            str(total_train_step),
                            str(mlm_loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                            str(mem_loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                            str(acc.cpu().detach().numpy()), str(lr)
                        )
                        )
                    else:
                        bmt.print_rank('Epoch:{}, Step:{}/{}, mlm_loss:{}, mem_loss:{}, cls_acc:{}%, lr:{}.'.format(
                            str(epoch), str(current_step),
                            str(total_train_step),
                            str(mlm_loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                            str(mem_loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                            str(acc.cpu().detach().numpy()), str(lr)
                        )
                        )

                        # bmt.synchronize()
                    # wandb.log({
                    #     "training loss": loss.cpu().detach().numpy(),
                    # })

                    self.log_file.write('Epoch:{}, Step:{}/{}, loss:{}, acc:{}%, lr:{}.'.format(
                        str(epoch), str(current_step),
                        str(total_train_step),
                        str(loss.cpu().detach().numpy() if not self.bmtrain else global_loss),
                        str(acc.cpu().detach().numpy()), str(lr)) + '\n'
                                        )
                    self.log_file.flush()
            # wandb.log({
            #     "learning rate": np.array(float(lr))
            # })
            # torch.cuda.synchronize()
            self.validation(epoch)
            epoch_time = time.time() - start_time
            if self.bmtrain:
                bmt.print_rank("time:" + str(epoch_time))
            else:
                print("time:" + str(epoch_time))
            self.model.train()
            self.lr_scheduler.step() if not self.bmtrain else bmt.optim_step(self.lr_scheduler)

        return acc, loss_sum
