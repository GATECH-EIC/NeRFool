import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer, reduction='sum', num_source_views=None):
        self._optim, self._reduction = optimizer, reduction
        self.num_source_views = num_source_views
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()


    def pc_backward(self, objectives, major_loss=None):
        grads, has_grads = self._pack_grad(objectives)
        grads = self._project_conflicting(major_loss, grads, has_grads)
        self._set_grad(grads)
        
        return


    def _project_conflicting(self, major_loss, grads, has_grads):
        for i, grad_dict in enumerate(grads):
            has_grad = list(has_grads[i].values())
            if sum(has_grad) != len(has_grad):  # do not receive multitask losses
                assert sum(has_grad) <= 1
                continue
            
            task_list = list(grad_dict.keys())

            shape = grad_dict[task_list[0]].shape
            
            for task in task_list:
                grad_dict[task] = grad_dict[task].view(-1)
                
            if major_loss:
                g_j = grad_dict[major_loss]

                for task in task_list:
                    if task == major_loss:
                        continue
                    g_i = grad_dict[task]
                    g_i_g_j = torch.dot(g_i, g_j)
                    if g_i_g_j < 0:
                        g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2 + 1e-6)

                    grad_dict[task] = g_i
                    
                grads[i] = list(grad_dict.values())

            else:
                grad_dict = list(grad_dict.values())
                pc_grad = copy.deepcopy(grad_dict)
                for g_i in pc_grad:
                    random.shuffle(grad_dict)
                    for g_j in grad_dict:
                        g_i_g_j = torch.dot(g_i, g_j)
                        if g_i_g_j < 0:
                            g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2 + 1e-6)

                grads[i] = pc_grad
                
            if self._reduction == 'sum':
                grads[i] = torch.stack(grads[i]).sum(dim=0)
            else:
                exit('invalid reduction method')

            grads[i] = grads[i].view(shape)
            
        return grads

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''
        grads_new = []
        for i in range(len(grads)//self.num_source_views):
            grads_new.append(torch.stack(grads[i * self.num_source_views:(i+1)*self.num_source_views], dim=0).unsqueeze(0))
        grads = grads_new
        
        idx = 0
        self._optim.zero_grad()
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx].clone()
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads = []
        has_grads = []
        for i, (name, obj) in enumerate(objectives.items()):
            self._optim.zero_grad()
            
            if i == len(objectives.keys()) - 1:
                obj.backward(retain_graph=False)
            else:
                obj.backward(retain_graph=True)
                
            self._retrieve_grad(name, grads, has_grads)
            
        return grads, has_grads


    def _retrieve_grad(self, task_name, grads, has_grads):
        new_init = len(grads) == 0
        param_cnt = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    for i in range(self.num_source_views):
                        if new_init:
                            grads.append({task_name: None})
                            has_grads.append({task_name: False})
                        else:
                            grads[param_cnt][task_name] = None
                            has_grads[param_cnt][task_name] = False
                        param_cnt += 1
                else:
                    for i in range(self.num_source_views):
                        if new_init:
                            grads.append({task_name: p.grad.squeeze(0)[i].clone()})
                            has_grads.append({task_name: True})
                        else:
                            grads[param_cnt][task_name] = p.grad.squeeze(0)[i].clone()
                            has_grads[param_cnt][task_name] = True
                        param_cnt += 1
        return
