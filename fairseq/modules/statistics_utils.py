import numpy as np
import torch
from scipy import io

#save: save statistics in a file
#record: only record 
def save_residual_proportion(self, x, residual, module):
    # T,B,C
    assert module in ['att','ffn'], "wrong module in residual proportion!"
    if not self.training or not self.record_residual_proportion:
        return
    file = self.record_residual_att_file if module=='att' else self.record_residual_ffn_file
    if self.step%self.save_interval==0:
        self.x_proportion[file].append(x[:,0,:].norm())
        self.residual_proportion[file].append(residual[:,0,:].norm())
        self.total_proportion[file].append((x+residual)[:,0,:].norm())
        if(self.step%self.save_interval==0):
            savefile = '{}_{}.mat'.format(file,self.step//self.save_interval)
            d = {}
            d['x'] = np.array(self.x_proportion[file])
            d['residual'] = np.array(self.residual_proportion[file])
            d['total'] = np.array(self.total_proportion[file])
            self.x_proportion[file] = []
            self.residual_proportion[file] = []
            self.total_proportion[file] = []
            io.savemat(savefile,d)

def get_singular_values(cov, eps=1e-5):
    #input: covariance matrix C*C
    #output: singular values in increasing order: C
    C,_ = cov.shape
    cov += eps*torch.eye(C).cuda() #for numerical stability
    s, _ = torch.symeig(cov)
    return s

def record_forward_weight_norm(self):
    if not self.training or not self.record_weight_norm:
        return 
    if self.step%self.record_interval==0:
        w1, w2 = self.fc1.weight.data, self.fc2.weight.data
        self.w1_norm.append(w1.norm())
        self.w2_norm.append(w2.norm())
        cov1 = torch.matmul(w1.transpose(0,1),w1)
        cov2 = torch.matmul(w2,w2.transpose(0,1))
        s1 = get_singular_values(cov1)
        s2 = get_singular_values(cov2)
        self.w_singular_value[self.record_w1_grad_file].append(s1.reshape([1,-1]))
        self.w_singular_value[self.record_w2_grad_file].append(s2.reshape([1,-1]))

def save_forward_backward_weight_norm(self,savefile):

    w1_norm = torch.tensor(self.w1_norm).cpu().numpy()
    w2_norm = torch.tensor(self.w2_norm).cpu().numpy()
    w1_grad_norm = torch.tensor(self.w_grad_norm[self.record_w1_grad_file]).cpu().numpy()
    w2_grad_norm = torch.tensor(self.w_grad_norm[self.record_w2_grad_file]).cpu().numpy()
    w1_singular_value = torch.cat(self.w_singular_value[self.record_w1_grad_file],dim=0).cpu().numpy()
    w2_singular_value = torch.cat(self.w_singular_value[self.record_w2_grad_file],dim=0).cpu().numpy()
    d = {}
    d['w1_norm'] = w1_norm
    d['w2_norm'] = w2_norm
    d['w1_grad_norm'] = w1_grad_norm
    d['w2_grad_norm'] = w2_grad_norm
    d['w1_singular_value'] = w1_singular_value
    d['w2_singular_value'] = w2_singular_value
    file = "{}_{}.mat".format(savefile, self.step//self.save_interval)
    io.savemat(file,d)
    self.w1_norm = []
    self.w2_norm = []
    self.w_grad_norm[self.record_w1_grad_file] = []
    self.w_grad_norm[self.record_w2_grad_file] = []
    self.w_singular_value[self.record_w1_grad_file] = []
    self.w_singular_value[self.record_w2_grad_file] = []

def save_attn_weights(self, attn_weights):
    #B*T*T(tgt_len, src_len)
    if not self.training or not self.record_attn_weight:
        return
    if self.step%self.save_interval==0:
        attn_weights = attn_weights[0]
        x_len = self.mask[0].sum()
        x_len = x_len.int()
        attn_weights = attn_weights[:x_len,:x_len]  #要注意是left-pad还是right-pad bug!
        attn_weights = attn_weights.data.cpu().numpy()
        self.attn_weight_list.append(attn_weights)
        if(self.step%self.save_interval==0):
            file = '{}_{}.mat'.format(self.attn_weight_savefile,self.step//self.save_interval)
            d = {}
            d['attn_weight'] = np.array(self.attn_weight_list)
            self.attn_weight_list = []
            io.savemat(file,d)

def dominated_word(self, x, position, process):
    #x: T,B,C
    copyx = x
    copyx = copyx.transpose(0,1) #B,T,C
    norms = copyx.norm(dim=-1) #B,T
    self.d_norm[process][position].append(norms.norm().cpu().numpy().item())
    #dominated word
    v, _ = norms.max(dim=-1,keepdim=True) #B,1
    
    norms = norms/v
    mean = norms.mean(dim=-1) #B
    index = torch.argmax(norms,dim=-1,keepdim=True) #B,1  #求argmax还不够，还要满足第一与第二的比值足够大 利用topk函数
    word = self.src_tokens.gather(dim=-1,index=index)
    value2, _ = torch.topk(norms,k=2) #B,2
    self.d_dominated_word[process][position].append(word.cpu().numpy().reshape([-1,]))
    self.d_dominated_index[process][position].append(index.cpu().numpy().reshape([-1,]))
    self.d_dominated_r[process][position].append(mean.cpu().numpy().reshape([-1,]))
    self.d_dominated_top2_value[process][position].append(value2.cpu().numpy())

def zero_rate(self, x, position, process):
    T,B,C = x.shape
    copyx = x
    num_tokens = self.mask.sum()
    num_pads = B*T*C-num_tokens*C
    copyx = copyx*(self.mask.transpose(0,1))
    num_zeros = (copyx==0).sum()
    num_zero_neurous = (((copyx==0).sum(dim=0))==T).sum()
    num_zero_words = (((copyx==0).sum(dim=-1))==C).sum()
    num_zero_words = num_zero_words-(B*T-num_tokens)
    num_pads = num_pads.type_as(num_zeros)
    num_zeros = num_zeros-num_pads

    num_zeros = num_zeros.cpu().numpy().item()
    num_tokens = num_tokens.cpu().numpy().item()
    num_zero_neurous = num_zero_neurous.cpu().numpy().item()
    num_zero_words = num_zero_words.cpu().numpy().item()

    r = num_zeros/num_tokens/C
    r_neuron = num_zero_neurous/B/C
    r_word = num_zero_words/num_tokens
    self.d_zero_rate[process][position].append([r, r_neuron, r_word])

def norm_distribution(self, x, position, process):
    #print(copyx.shape)
    copyx = x.transpose(0,1) #B,T,C
    items = min(copyx.shape[0],2)
    for i in range(items):
        temx = copyx[i] #T,C
        len_x = self.mask[i].sum()
        len_x = len_x.int()
        temx = temx[:len_x] #len_x, C
        bag = torch.zeros(self.max_len)
        bag[:len_x] = torch.norm(temx,dim=1)
        bag[-1] = len_x.float()
        self.d_norm_distribution[process][position].append(bag.reshape([1,-1]))  #一维数组

def save_norm_statistics(self, x, position, process):
    #T,B,C
    if len(self.record_parts[process])==0:
        return

    if 'norm_distribution' in self.norm_items[process]:
        norm_distribution(self, x, position, process)
    if 'dominated_word' in self.norm_items[process]:
        dominated_word(self, x, position, process)
    if 'zero_rate' in self.norm_items[process]:
        zero_rate(self, x, position, process)

    if(self.step%self.save_interval==0):
        d = {}
        if 'norm_distribution' in self.norm_items[process]:
            save_norm_distribution = torch.cat(self.d_norm_distribution[process][position],dim=0).cpu().numpy()
            d['norm_distribution'] = save_norm_distribution     
            self.d_norm_distribution[process][position] = []
        if 'dominated_word' in self.norm_items[process]:
            save_dominated_word = np.concatenate(self.d_dominated_word[process][position])
            save_dominated_index = np.concatenate(self.d_dominated_index[process][position])
            save_dominated_r = np.concatenate(self.d_dominated_r[process][position])
            save_dominated_top2_value = np.concatenate(self.d_dominated_top2_value[process][position],axis=0)
            save_norm = np.array(self.d_norm[process][position])
            d['dominated_word'] = save_dominated_word
            d['dominated_index'] = save_dominated_index
            d['dominated_r'] = save_dominated_r
            d['dominated_top2_value'] = save_dominated_top2_value
            d['norm'] = save_norm
            self.d_dominated_word[process][position] = []
            self.d_dominated_index[process][position] = []
            self.d_dominated_r[process][position] = []
            self.d_dominated_top2_value[process][position] = []
            self.d_norm[process][position] = []
        if 'zero_rate' in self.norm_items[process]:
            r_list = [i[0] for i in self.d_zero_rate[process][position]]
            r_neuron_list = [i[1] for i in self.d_zero_rate[process][position]]
            r_word_list = [i[2] for i in self.d_zero_rate[process][position]]
            save_zero_rate = np.array(r_list)
            save_zero_neuron_rate = np.array(r_neuron_list)
            save_zero_word_rate = np.array(r_word_list)
            d['zero_rate'] = save_zero_rate
            d['zero_neuron_rate'] = save_zero_neuron_rate
            d['zero_word_rate'] = save_zero_word_rate
            self.d_zero_rate[process][position] = []
        file = "{}_{}.mat".format(self.d_savefile[process][position],self.step//self.save_interval)
        io.savemat(file,d)

def save_condition_statistics(self, x, position, process):
    #T,B,C
    if len(self.condition_items[process])==0:
        return
    eps = 1e-5
    T, B, C = x.shape
    x_len = self.mask.sum(dim=(1,2)) #B
    
    r = torch.zeros(1).cuda()
    c_max = torch.zeros(1).cuda()
    c_20 = torch.zeros(1).cuda()
    c_50 = torch.zeros(1).cuda()
    c_80 = torch.zeros(1).cuda()
    r_total = torch.zeros(1).cuda()
    intra_average_sim = torch.zeros(1).cuda()
    inter_average_sim = torch.zeros(1).cuda()
    total_average_sim = torch.zeros(1).cuda()
    s = torch.zeros(1).cuda()

    if  'r' in self.condition_items[process]:
        rx = x.transpose(0,1) #B,T,C
        #cov_r = torch.bmm(rx.transpose(1,2),rx) #B,C,C
        iters = min(rx.shape[0],1)
        for i in range(iters):
            feature = rx[i] #T,C
            cov_r = torch.matmul(feature.transpose(0,1),feature)/x_len[i]
            cov_r += torch.eye(C).cuda() #for numerical stability
            try:
                s, _ = torch.symeig(cov_r)  #返回为增序特征值  特征值分解很费时间
            except:
                print("eigenvalue decomposition error when computing stable rank! ")
                print(cov_r)
                print(feature[:,1].sum())
                print(cov_r.shape)
                print(x_len[i])
                #print(s)
                exit()
                s = torch.ones(cov_r.size(1)).cuda()

            temr = torch.sum(s)/s[-1] #square of singular values
            r += temr
        r = r/iters
        #r = r.cpu().numpy().item()

    #start_time = time.time()
    if 'c_max' in self.condition_items[process] or 'r_total' in self.condition_items[process]:
        yx = x.transpose(0,1) #B,T,C
        y = yx.reshape(-1,yx.shape[-1]) #(B*T)*C
        #y = y[index] #去掉pad word
        cov = torch.matmul(y.transpose(0,1),y)/y.shape[0] #C*C C=512
        try:
            s, _ = torch.symeig(cov)  #返回为增序特征值  特征值分解很费时间
        except:
            print("eigenvalue decomposition error!")
            s = torch.ones(cov.size(1)).cuda()

        r_total = (torch.sum(s)/s[-1]) #square of singular values
        c_max = s[-1]
        c_20 = (c_max/s[len(s)*4//5])
        c_50 = (c_max/s[len(s)//2])
        c_80 = (c_max/s[len(s)*1//5])
        
    #print("cov time: {}".format(time.time() - start_time))
    #start_time = time.time()
    if 'intra_average_sim' in self.condition_items[process]:
        sx = x.transpose(0,1) #B,T,C
        sim_items = B//4
        sx = sx[:sim_items]
        y = sx.reshape(-1,sx.shape[-1]) #(sim_items*T)*C
        y = y/(y.norm(dim=-1,keepdim=True)+eps)
        sim = torch.matmul(y, y.transpose(0,1)) #sim_items*T,sim_items*T
        
        #print("sim 1 time: {}".format(time.time() - start_time))
        #start_time = time.time()
        intra_sim = 0
        items_len = x_len[:sim_items].reshape([1,-1])
        for i in range(sim_items):
            temsim = torch.sum(sim[T*i:T*(i+1),T*i:T*(i+1)])
            intra_sim += temsim
        
        dim = torch.sum(items_len)

        inter_sim = torch.sum(sim)-intra_sim
        intra_sim -= dim

        total_average_sim = inter_sim+intra_sim

        intra_items = torch.sum(items_len**2)-torch.sum(items_len)
        total_items = dim*dim-dim
        inter_items = total_items-intra_items
        
        intra_average_sim = intra_sim/intra_items
        inter_average_sim = inter_sim/inter_items
        total_average_sim = total_average_sim/total_items
    
        #print("sim 2 time: {}".format(time.time() - start_time))
        #start_time = time.time()

    collect_items = [c_max, c_20, c_50, c_80, r_total, r,  intra_average_sim, inter_average_sim, total_average_sim]
    self.d_condition_number[process][position].append(collect_items)
    self.d_condition_singular_value[process][position].append(s.reshape([1,-1]))
    if self.step%self.save_interval==0:
        d = {}
        for i, name in enumerate(self.condition_items[process]):
            d[name] = np.array([b[i].cpu().numpy().item() for b in self.d_condition_number[process][position]])
        singular_value = torch.cat(self.d_condition_singular_value[process][position],dim=0).cpu().numpy()
        d['condition_singular_value'] = singular_value[::10] #取1/10即可
        self.d_condition_number[process][position] = []
        self.d_condition_singular_value[process][position] = []
        file = "{}_{}.mat".format(self.d_condition_savefile[process][position],self.step//self.save_interval)
        io.savemat(file,d)

def probe_hook(self, position, process):
    def hook_template(x):
        copyx = x.data.clone()
        copyx = copyx*self.mask.transpose(0,1)
        #x can be forward features or backward gradient
        if 'norm' in self.record_parts[process]:
            save_norm_statistics(self, copyx, position, process)
        if 'condition' in self.record_parts[process]:
            save_condition_statistics(self, copyx, position, process)
    return hook_template

def insert_forward_probe(self, x, position):
    return 
    if self.record_process['forward'] and position in self.probe_positions["forward"]:
        cur_probe_hook = probe_hook(self, position, process="forward")
        cur_probe_hook(x)
        
def insert_backward_probe(self, x, position):
    return 
    if self.record_process['backward'] and position in self.probe_positions["backward"]:
        cur_probe_hook = probe_hook(self, position, process="backward")
        x.register_hook(cur_probe_hook)

def insert_probe_unify(self, x, position, process):
    assert process in ['forward', 'backward'], "wrong process(insert_probe_unify)"
    if self.record_process[process] and position in self.probe_positions[process]:
        cur_probe_hook = probe_hook(self, position, process=process)
        if process=='forward':
            cur_probe_hook(x)
        else:
            x.register_hook(cur_probe_hook)

def insert_probe(self, x, position):
    #includ forward & backward probe
    if not self.training or self.step%self.record_interval!=0:
        return
    insert_probe_unify(self, x, position, process='forward')
    insert_probe_unify(self, x, position, process='backward')