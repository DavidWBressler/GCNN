from fastai import *
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


#create adaptive-softmax language-model dataset
class LMDataset_GCNN(Dataset):
    def __init__(self, tokens):
        self.tokens=tokens
    def __getitem__(self,index):
        #token_list=torch.FloatTensor(self.tokens[index]).cuda()
        token_list=torch.LongTensor(self.tokens[index])
        label=torch.FloatTensor([1])
        #label=torch.ones(len(token_list)-1).float()
        return token_list,label
    def __len__(self):
        return len(self.tokens)
    
    
class SortSampler_GCNN(Sampler): #inspired by fast.ai sortsampler... pass in something like key=lambda x: len(val_clas[x])
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))#return iterator in reverse order, sorted by input key (e.g. length)
    
#this sortishsampler does the following:
    # 1) get a list of randomized indices of length of entire dataset
    # 2) break that into a list of sublists, each sublist of size bs*50
    # 3) create a new list that is sorted within each of those chunks
    # 4) break that sorted list into chunks of size bs
    # 5) create new list by randomizing order of all those bs-chunks (with bs-chunk w/ largest key first)
#this will give batches sorted by key (e.g. length) within the batch
class SortishSampler_GCNN(Sampler): #inspired by fast.ai sortishsampler... pass in something like key=lambda x: len(val_clas[x])
    def __init__(self, data_length, key,bs): self.data_length,self.key,self.bs = data_length,key,bs
    def __len__(self): return self.data_length
    def __iter__(self):
        idxs = np.random.permutation(self.data_length)#random permutation of length of entire dataset
        sz = self.bs*50 #chunk size is bs*50
        #range(0, len(idxs), sz) : go through length of entire dataset, with stepsize=chunk_size
        #idxs[i:i+sz] :within that chunk's range, get all the indices of the random permutation above
        #this creates a list of lists... basically just splitting up idxs into a bunch of chunks
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        #for s in ck_idx: go through each sublist in the big list
        #sorted(s, key=self.key, reverse=True): sort the sublist in reverse order according to the key (e.g. length)
        #np.concatenate: concatenate all the sorted chunk sublists together
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs #now set size to bs
        #similar as before, this creates a list of lists, splitting up sort_idx into chunks of size bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        # go through each bs-chunk, get the key of the first entry (which should be the largest of the chunk)...
        # then do argmax to find the chunk with the largest key
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  
        #switch spots bw the first chunk and the chunk w/ the max key
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]
        #now randomize the order of all the bs-chunks (except the first), then concatenate together
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))# concatenate the first (largest key val) chunk to the rest
        return iter(sort_idx)

#inspired by fast.ai's pad_collate
def pad_collate_GCNN(samples, pad_idx=1):
    #go through all the sentences (found in s[0]), find length of longest
    max_len = max([len(s[0]) for s in samples]) 
    #create a tensor of size [max_len,n_samples], and set all values to pad_idx
    res = torch.zeros(max_len, len(samples)).long() + pad_idx 
    #for each line in res, set so corresponding sentence is aligned to the left edge
        #(right-padded: keep padding on the right edge)
    for i,s in enumerate(samples): res[:len(s[0]),i] = LongTensor(s[0]) #right-padded
    #return res as the padded tensor, and another tensor composed of the labels (found in s[1])
    labelsList = [s[1] for s in samples]
    labels = torch.stack(labelsList, dim=0)
    labels = torch.FloatTensor(labels)
    labels = labels.squeeze().cuda()
    return res.cuda(), labels



    
    
