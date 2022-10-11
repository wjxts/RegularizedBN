import numpy as np
import torch
from scipy import io
from .statistics_utils import save_forward_backward_weight_norm
#need to define self.prefix, self.id before initializing statistics 

def init_residual_proportion(self, args):
    self.record_residual_proportion = args.record_residual_proportion
    if self.record_residual_proportion:
        self.record_residual_att_file =  'statistics/{}/residual_att_{}'.format(self.prefix,self.id)
        self.record_residual_ffn_file =  'statistics/{}/residual_ffn_{}'.format(self.prefix,self.id)
        self.x_proportion = {}
        self.residual_proportion = {}
        self.total_proportion = {}
        files = [self.record_residual_att_file, self.record_residual_ffn_file]
        for file in files:
            self.x_proportion[file] = []
            self.residual_proportion[file] = []
            self.total_proportion[file] = []

def backward_hook_weight(self, savefile):
    def backward_hook_template(grad):
        if self.training and self.step%self.save_interval==0:
            self.w_grad_norm[savefile].append(grad.norm())
        if(self.step%self.save_interval==0 and savefile.find("w1_grad")>=0):
            save_forward_backward_weight_norm(self, self.save_weight_file)
    return backward_hook_template

def init_forward_backward_weight_norm(self,args):
    self.record_weight_norm = args.record_weight_norm
    if self.record_weight_norm:
        self.record_w1_grad_file =  'statistics/{}/w1_grad_norm_{}'.format(self.prefix,self.id)
        self.record_w2_grad_file =  'statistics/{}/w2_grad_norm_{}'.format(self.prefix,self.id)
        self.save_weight_file = 'statistics/{}/weight_grad_norm_{}'.format(self.prefix,self.id)
        self.w1_norm = []
        self.w2_norm = []
        self.w_grad_norm = {}
        self.w_singular_value = {}
        self.w_grad_norm[self.record_w1_grad_file] = []
        self.w_grad_norm[self.record_w2_grad_file] = []
        self.w_singular_value[self.record_w1_grad_file] = []
        self.w_singular_value[self.record_w2_grad_file] = []
        self.fc1.weight.register_hook(backward_hook_weight(self, self.record_w1_grad_file))
        self.fc2.weight.register_hook(backward_hook_weight(self, self.record_w2_grad_file))

def init_attn_weight(self, args):
    self.record_attn_weight = args.record_attn_weight
    if self.record_attn_weight:
        self.attn_weight_list = []
        self.attn_weight_savefile = 'statistics/{}/attn_weight_{}'.format(self.prefix, self.id)

def init_probe_norm(self, args, process):
    assert process in ['forward', 'backward']
    self.d_savefile[process] = {}
    self.d_norm[process]= {}
    self.d_norm_distribution[process] = {}
    self.d_dominated_word[process] = {}
    self.d_dominated_index[process] = {}
    self.d_dominated_top2_value[process] = {}
    self.d_dominated_r[process] = {}
    self.d_zero_rate[process] = {}
    #use position to mark everything
    for position in self.probe_positions[process]:
        savefile = 'statistics/{}/{}_norm_{}_{}'.format(self.prefix, process, position, self.id)
        self.d_savefile[process][position] = savefile
        self.d_norm[process][position] = []
        self.d_norm_distribution[process][position] = []
        self.d_dominated_word[process][position]= []
        self.d_dominated_index[process][position] = []
        self.d_dominated_top2_value[process][position] = []
        self.d_dominated_r[process][position] = []
        self.d_zero_rate[process][position] = []

def init_probe_condition(self, args,  process):
    assert process in ['forward', 'backward']
    self.d_condition_savefile[process] = {}
    self.d_condition_number[process] = {}
    self.d_condition_singular_value[process] = {}
    #use position to mark everything
    for position in self.probe_positions[process]:
        savefile = 'statistics/{}/{}_condition_{}_{}'.format(self.prefix, process, position, self.id)
        self.d_condition_savefile[process][position] = savefile
        self.d_condition_number[process][position] = []
        self.d_condition_singular_value[process][position] = []

def init_probe_statistics(self, args, process):
    assert process in ['forward', 'backward']
    def add_prefix(cur_list, prefix):
        cur_list = ["{}_{}".format(prefix, item) for item in cur_list]
        return cur_list
    #self.candidate_positions[process] = add_prefix(self.all_positions, process)
    #self.candidate_parts[process] = add_prefix(self.all_parts, process)
    #self.candidate_norm_items[process] = add_prefix(self.all_norm_items, process)
    #self.candidate_condition_items[process] = add_prefix(self.all_condition_items, process)
    
    #"norm" includes: input_norm_distribution, dominated word(include input_norm), zero_rate
    #
    probe_position = args.forward_probe_position if process=='forward' else args.backward_probe_position
    record_parts = args.forward_record_parts if process=='forward' else args.backward_record_parts
    record_norm_items = args.forward_record_norm_items if process=='forward' else args.backward_record_norm_items
    record_condition_items = args.forward_record_condition_items if process=='forward' else args.backward_record_condition_items

    if probe_position=='all':
        self.probe_positions[process] = self.all_positions
    else:
        self.probe_positions[process] = (probe_position).split(',') if probe_position!='none' else []

    if record_parts=='all':
        self.record_parts[process] = self.all_parts
    else:
        self.record_parts[process] = (record_parts).split(',') if record_parts!='none' else []
        
    
    if record_norm_items=='all':
        self.norm_items[process] = self.all_norm_items
    else:
        self.norm_items[process] = (record_norm_items).split(',') if record_norm_items!='none' else []
        
    if record_condition_items=='all':
        self.condition_items[process] = self.all_condition_items
    else:
        self.condition_items[process] = (record_condition_items).split(',') if record_condition_items!='none' else []

    self.record_process[process] = 0 if probe_position=='none' or record_parts=='none' else 1
    if not self.record_process[process]:
        return 
    init_probe_norm(self, args, process)
    init_probe_condition(self, args, process)

def init_all_statistics(self,args):
    #forward part
    init_residual_proportion(self, args)
    init_forward_backward_weight_norm(self,args)
    init_attn_weight(self, args)
    
    init_probe_statistics(self, args, "forward")

    #backward statistics are exactly the same as forward statistics
    #backward part
    init_probe_statistics(self, args, "backward")

def init_base_dictionary(self,args):
    #condition
    self.d_condition_savefile = {}
    self.d_condition_number = {}
    self.d_condition_singular_value = {}
    #others
    self.d_savefile = {}
    self.d_norm = {}
    self.d_norm_distribution = {}
    self.d_dominated_word = {}
    self.d_dominated_index = {}
    self.d_dominated_top2_value = {}
    self.d_dominated_r = {}
    self.d_zero_rate = {}
    #
    self.probe_positions = {}
    self.record_parts = {}
    self.norm_items = {}
    self.condition_items = {}
    #
    self.all_positions = ['att_input','att_output','att_norm_input','ffn_input','ffn_output','ffn_norm_input','before_relu','after_relu']
    self.all_parts = ['norm','condition']
    self.all_norm_items = ['norm_distribution','dominated_word','zero_rate']
    self.all_condition_items = ['c_max', 'c_20', 'c_50', 'c_80', 'r_total', 'r',  'intra_average_sim', 'inter_average_sim', 'total_average_sim']
    self.candidate_positions = {}
    self.candidate_parts = {}
    self.candidate_norm_items = {}
    self.candidate_condition_items = {}
    #
    self.record_process = {}

def init_config(self, args):
    #initialize configuration for saving statistics
    def extract_id(s):
        cand = s.split('/')
        for c in cand:
            if(c.find('transformer')>=0):
                return c
        print("error path!")
        exit()
        return 'error'
    self.prefix = extract_id(args.save_dir)
    self.step = 0
    self.save_interval = 1100 #standard: 1100
    self.record_interval = 50 #standard: 20
    assert self.save_interval%self.record_interval==0, "wrong interval!"
    self.max_len = 600
    #self.d_statistics = {} #for saving forward & backward statistics
    init_base_dictionary(self,args)
    init_all_statistics(self,args)