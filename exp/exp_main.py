from data_provider.data_factory import data_provider, preload_all_data
from exp.exp_basic import Exp_Basic
from models import MDMLP_EIA
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import math
import copy

warnings.filterwarnings('ignore')

# Only MDMLP_EIA: forward uses model(batch_x)
list_A = {'MDMLP_EIA'}

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.best_model_state = None
        
        try:
            print("Attempting to preload all datasets...")
            self.train_data, self.train_loader, self.val_data, self.val_loader, self.test_data, self.test_loader = preload_all_data(args)
            self.data_preloaded = True
            print("All datasets successfully preloaded")
        except Exception as e:
            print(f"Warning: Could not preload datasets: {e}")
            print("Will load datasets on demand instead")
            self.data_preloaded = False

    def _build_model(self):
        model_dict = {'MDMLP_EIA': MDMLP_EIA}
        if self.args.model not in model_dict:
            raise ValueError(f"Unsupported model: {self.args.model}. Only MDMLP_EIA is available.")
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if hasattr(self, 'data_preloaded') and self.data_preloaded:
            if flag == 'train':
                return self.train_data, self.train_loader
            elif flag == 'val':
                return self.val_data, self.val_loader
            elif flag == 'test':
                return self.test_data, self.test_loader
            else:
                return data_provider(self.args, flag)
        else:
            return data_provider(self.args, flag)

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate) #, weight_decay = 3e-2,)
        return model_optim

    # # MSE criterion
    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion

    # MSE and MAE criterion
    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def vali(self, vali_data, vali_loader, criterion, is_test = True):
        total_loss = []
        self.model.eval()
        
        if not is_test and not hasattr(self, 'vali_ratio'):
            self.vali_ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
            self.vali_ratio = torch.tensor(self.vali_ratio).unsqueeze(-1).to(self.device).contiguous()
            
        forward_times = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                forward_start = time.time()
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in list_A
                                 ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in list_A):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                forward_time = time.time() - forward_start
                forward_times.append(forward_time)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # if train, use ratio to scale the prediction
                if not is_test:
                    pred = outputs * self.vali_ratio
                    true = batch_y * self.vali_ratio
                else:
                    pred = outputs
                    true = batch_y

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        
        avg_forward_time = np.mean(forward_times) if forward_times else 0
        if is_test:
            print(f"Validation inference time (forward pass only): {avg_forward_time:.6f}s per batch")
            
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            train_data, train_loader = self._get_data(flag='train')
            self.train_data = train_data
            self.train_loader = train_loader
        else:
            train_data, train_loader = self.train_data, self.train_loader
            
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            vali_data, vali_loader = self._get_data(flag='val')
            self.val_data = vali_data
            self.val_loader = vali_loader
        else:
            vali_data, vali_loader = self.val_data, self.val_loader
            
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            test_data, test_loader = self._get_data(flag='test')
            self.test_data = test_data
            self.test_loader = test_loader
        else:
            test_data, test_loader = self.test_data, self.test_loader

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # criterion = self._select_criterion() # For MSE criterion
        mse_criterion, mae_criterion = self._select_criterion()
        
        self.train_ratio = np.array([-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(self.args.pred_len)])
        self.train_ratio = torch.tensor(self.train_ratio).unsqueeze(-1).to(self.device).contiguous()

        all_epoch_forward_times = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_forward_times = [] 
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                forward_start = time.time()
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in list_A
                             ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in list_A):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                
                forward_time = time.time() - forward_start
                epoch_forward_times.append(forward_time)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs * self.train_ratio
                batch_y = batch_y * self.train_ratio

                loss = mae_criterion(outputs, batch_y)
                # loss = mae_criterion(outputs, batch_y) + mse_criterion(outputs, batch_y)

                loss_auxi = torch.abs(torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).mean()
                loss = 0.8 * loss_auxi + (1-0.8) * loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            avg_epoch_forward_time = np.mean(epoch_forward_times) if epoch_forward_times else 0
            all_epoch_forward_times.append(avg_epoch_forward_time)
            print(f"Epoch {epoch + 1} average forward pass time: {avg_epoch_forward_time:.6f}s per batch")
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion) # For MSE criterion
            # test_loss = self.vali(test_data, test_loader, criterion) # For MSE criterion
            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            best_state = early_stopping(vali_loss, self.model, path)
            if best_state is not None:
                self.best_model_state = best_state

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
           
        avg_train_forward_time = np.mean(all_epoch_forward_times) if all_epoch_forward_times else 0
        print(f"Average training forward pass time across all epochs: {avg_train_forward_time:.6f}s per batch")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model from memory for final evaluation")
        
        best_model_path = os.path.join(path, 'best_model.pth')
        torch.save(self.best_model_state, best_model_path)
        print(f"Best model saved at: {best_model_path}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                print("Loading best model from memory")
            else:
                model_path = os.path.join(self.args.checkpoints, setting, 'best_model.pth')
                if not os.path.exists(model_path):
                    model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
                
                print(f"Loading model from {model_path}")
                self.model.load_state_dict(torch.load(model_path))

        preds = []
        trues = []
        folder_path = os.path.join('./test_results/', setting, '')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_forward_times = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                forward_start = time.time()
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in list_A
                                 ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in list_A):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                forward_time = time.time() - forward_start
                test_forward_times.append(forward_time)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
            
        avg_test_forward_time = np.mean(test_forward_times) if test_forward_times else 0
        print(f"Test inference time (forward pass only): {avg_test_forward_time:.6f}s per batch")
        print(f"Total test forward pass time: {sum(test_forward_times):.6f}s")
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(self.args.result_file, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write(f'Average inference time: {avg_test_forward_time:.6f}s per batch')
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                print("Loading best model from memory for prediction")
            else:
                path = os.path.join(self.args.checkpoints, setting)
                model_path = os.path.join(path, 'best_model.pth')
                if not os.path.exists(model_path):
                    model_path = os.path.join(path, 'checkpoint.pth')
                
                print(f"Loading model from {model_path}")
                self.model.load_state_dict(torch.load(model_path))

        preds = []
        predict_forward_times = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                forward_start = time.time()
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in list_A
                                 ):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in list_A):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                forward_time = time.time() - forward_start
                predict_forward_times.append(forward_time)
                
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        avg_predict_forward_time = np.mean(predict_forward_times) if predict_forward_times else 0
        print(f"Prediction inference time (forward pass only): {avg_predict_forward_time:.6f}s per batch")
        
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = os.path.join('./results/', setting, '')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        
        with open(os.path.join(folder_path, 'inference_time.txt'), 'w') as f:
            f.write(f'Average prediction inference time: {avg_predict_forward_time:.6f}s per batch\n')
            f.write(f'Total prediction time: {sum(predict_forward_times):.6f}s\n')

        return