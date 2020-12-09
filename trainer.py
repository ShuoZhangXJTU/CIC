import time
import json
import os
import torch
from loading.data_loading import my_loader, text2ipt
from Main_Model import main_model
from evaluation.loss import FocalLoss
from evaluation.metrics import evaluation_indicators
from evaluation.result_analysis import rst_analysis
from tensorboardX import SummaryWriter
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from tune_space import get_space, get_resources
from configuration import *
from utils import print_progress


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, config):
        self.config = vars(config)
        with open('best_config.conf') as cf:
            best_config = json.load(cf)[self.config["model"]]
        for key_, val_ in best_config.items():
            self.config[key_] = val_

        self.model_name = self.config["model_name"]

        # -- set tensorboardX
        log_path = './log/{}'.format(self.model_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.writer = SummaryWriter(log_path)

        # -- load data
        self.train_loader, self.val_loader, self.test_loader, self.field, self.ttl_batches, self.POS_field = my_loader(
            self.config)
        self.ntokens, self.nPOS = len(self.field.vocab), len(self.POS_field.vocab)

        # -- model
        self.best_f1 = -1
        self.best_model = None
        self.model = main_model(self.config, self.ntokens, self.nPOS)
        self.load_pre_trained_emb(self.field, self.POS_field)

        # -- loss, optimizer, schedular
        self.criterion = FocalLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 32)

    def load_pre_trained_emb(self, field, POS_field):
        IDX_LST = [field.vocab.stoi[field.unk_token], field.vocab.stoi[field.pad_token], field.vocab.stoi['<blk>']]
        EMSIZE = self.config["emsize"] if self.config["dataset"] == 'cn' else 200

        if self.config["model"] != 'bert':
            self.model.embedding.weight.data.copy_(field.vocab.vectors)
            self.model.embedding.weight.requires_grad = False

            for IDX in IDX_LST:
                self.model.embedding.weight.data[IDX] = torch.zeros(EMSIZE)

            if self.config["use_POS"]:
                self.model.POS_embedding.weight.data.copy_(POS_field.vocab.vectors)
                self.model.POS_embedding.weight.requires_grad = False

                for IDX in IDX_LST:
                    self.model.POS_embedding.weight.data[IDX] = torch.zeros(self.config["POSemsize"])

    def train(self):
        ttl_step = 0
        model = self.model.to(device)
        print('device: {}'.format(device))
        print('-------- CURRENT CONFIG ---------')
        for key_, val_ in self.config.items():
            print("{}: {}".format(key_, val_))
        print('=================================')
        for epoch in range(1, self.config["epochs"] + 1):
            step, total_loss, full_pred, full_target = 0, 0, None, None
            epoch_start_time, log_start_time = time.time(), time.time()
            model.train()

            for input_ in self.train_loader:
                with torch.autograd.set_detect_anomaly(True):
                    ttl_step += 1
                    step += 1
                    target = input_['targets']
                    output1, output2, pred = model(input_)

                    loss = self.criterion(output1, output2, pred, target)

                    if self.config["use_l2"]:
                        l2_reg = None
                        for w in model.decoder.parameters():
                            if not l2_reg:
                                l2_reg = w.norm(2)
                            else:
                                l2_reg = l2_reg + w.norm(2)
                        loss += l2_reg * self.config["reg_lambda"]
                    total_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.6)
                    self.optimizer.step()
                    self.scheduler.step()

                    full_pred = pred.clone().detach() if full_pred is None \
                        else torch.cat((full_pred, pred.clone().detach()), dim=1)
                    full_target = target if full_target is None else torch.cat((full_target, target), dim=0)

                    if ttl_step % self.config["log_interval"] == 0:
                        cur_loss = total_loss / self.config["log_interval"]
                        evaluation_indicators('train', self.writer, full_pred, full_target, cur_loss, ttl_step)

                        elapsed = time.time() - log_start_time
                        print('| epoch {:3d} | batch {:5d}/{:d} | '
                              'lr {:2.6f} | s/batch {:5.2f} | '
                              'loss {:5.6f}'.format(epoch, ttl_step, self.ttl_batches, self.scheduler.get_last_lr()[0],
                                                    elapsed / self.config["log_interval"], cur_loss))

                        total_loss, full_pred, full_target, log_start_time = 0, None, None, time.time()

            # evaluate model
            val_loss, val_F1, format_log = self.evaluate_test('val', epoch, model, self.val_loader)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.6f} | valid f1 {:5.6f} |'.format(epoch,
                 (time.time() - epoch_start_time), val_loss, val_F1))
            print(format_log)
            print('-' * 89)

            if val_F1 > self.best_f1:
                self.best_f1, self.best_model = val_F1, model
                format_log = format_log.replace('%', '')
                self.writer.add_text('Best F1', format_log, epoch)
                folder_path = self.config["path_model_para"] + '{}/'.format(self.config["dataset"])
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(self.best_model.state_dict(),
                           os.path.join(folder_path, '{}.pkl'.format(self.config["model_name"])))

        # test model
        self.evaluate_test(self.model, self.test_loader, 'test')

    def evaluate_test(self, mode, epoch, model, data_source):
        if mode == 'test':
            model = model.to(device)
            folder_path = self.config["path_model_para"] + '{}/'.format(self.config["dataset"])
            path_para = os.path.join(folder_path, '{}.pkl'.format(self.config["model_name"]))
            model.load_state_dict(torch.load(path_para, map_location=torch.device('cuda')))
            print('parameters loaded')

        print('start {}'.format(mode))
        model.eval()
        total_loss = 0
        full_pred, full_target, full_text1, full_offset1, full_text2, full_offset2 = None, None, None, None, None, None

        with torch.no_grad():
            step = 0
            for input_ in data_source:
                step += 1
                self.optimizer.zero_grad()

                target = input_['targets']
                output1, output2, pred = model(input_)

                loss = self.criterion(output1, output2, pred, target)
                total_loss += loss.item()

                full_text1 = input_['raw_text1'] if full_text1 is None else full_text1 + input_['raw_text1']
                full_text2 = input_['raw_text2'] if full_text2 is None else full_text2 + input_['raw_text2']
                full_offset1 = input_['offset1'] if full_offset1 is None else full_offset1 + input_['offset1']
                full_offset2 = input_['offset2'] if full_offset2 is None else full_offset2 + input_['offset2']
                full_pred = pred if full_pred is None else torch.cat((full_pred, pred), dim=1)
                full_target = target if full_target is None else torch.cat((full_target, target), dim=0)

        acc, precision, recall, F1, MCC, AUC = evaluation_indicators(mode, self.writer, full_pred, full_target,
                                                                     total_loss/(len(data_source)-1), epoch)

        metric_info = '[Total Average]  loss {:5.4f} | acc {:5.4f} |' \
                      ' ttl_precision {:5.4f} | ttl_recall {:5.4f} | ttl_F1 {:5.4f}|' \
                      ' ttl_MCC {:5.4f} | ttl_AUC {:5.4f}'.format(total_loss/(len(data_source)-1),
                                                                  acc, precision, recall, F1, MCC, AUC)
        if mode == 'val':
            format_log = 'AUC | acc | p | r | F1 | MCC | COPY_CODE \n' \
                         ' -|-|-|-|-|-|- \n ' \
                         '{0:.4} | {1:.4} | {2:.4} | {3:.4} | {4:.4} | {5:.4} |' \
                         ' & ${0:.2%}$ & ${1:.2%}$ & ${2:.2%}$' \
                         ' & ${3:.2%}$ & ${4:.2%}$ & ${5:.2%}$'.format(AUC, acc, precision, recall, F1, MCC)
            return total_loss, F1, format_log
        if mode == 'test':
            rst_analysis(F1, full_pred, full_target, full_text1, full_text2, full_offset1, full_offset2,
                         self.model_name, metric_info)
            print(metric_info)

    def predict(self, model, RAW_INPUT):
        model = model.to(device)
        folder_path = self.config["path_model_para"] + '{}/'.format(self.config["dataset"])
        path_para = os.path.join(folder_path, '{}.pkl'.format(self.config["model_name"]))
        print(path_para)
        model.load_state_dict(torch.load(path_para, map_location=torch.device('cuda')))
        model.eval()
        print('Model Ready')
        input_ = text2ipt(RAW_INPUT, self.field, self.POS_field)
        output1, output2, pred = model(input_)

    def tune(self):
        server = get_resources()
        ray.init(num_gpus=server.gpu,
                 num_cpus=server.cpu,
                 webui_host='0.0.0.0')

        basic_config = self.config
        search_space, current_best_params = get_space(basic_config['model'])
        for key_, val_ in search_space.items():
            basic_config[key_] = val_

        algo = HyperOptSearch(basic_config,
                              metric="mean_accuracy",
                              mode="max",
                              points_to_evaluate=current_best_params)

        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="mean_accuracy",
            max_t=400,
            grace_period=30)

        analysis = tune.run(train_for_turn,
                            scheduler=scheduler,
                            resources_per_trial={"gpu": server.gpu_per_trial, "cpu": server.cpu_per_trial},
                            num_samples=server.num_samples,
                            search_alg=algo)

        print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))


def train_for_turn(config):
    train_loader, val_loader, test_loader, field, ttl_batches, POS_field = my_loader(config)
    ntokens, nPOS = len(field.vocab), len(POS_field.vocab)
    model = main_model(config, ntokens, nPOS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32)
    criterion = FocalLoss()

    for epoch in range(512):
        model.train()
        total_loss = 0
        for input_ in train_loader:
            target = input_['targets']
            output1, output2, pred = model(input_)
            loss = criterion(output1, output2, pred, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.6)
            optimizer.step()
            scheduler.step()

        model.eval()
        full_pred, full_target = None, None
        with torch.no_grad():
            for input_ in val_loader:
                target = input_['targets']
                output1, output2, pred = model(input_)
                full_pred = pred if full_pred is None else torch.cat((full_pred, pred), dim=1)
                full_target = target if full_target is None else torch.cat((full_target, target), dim=0)
        acc, precision, recall, F1, MCC, AUC = evaluation_indicators('tune', None, full_pred, full_target, None, None)

        tune.report(mean_accuracy=F1)
