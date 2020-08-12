import json
import os
import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix, save_npz, load_npz ,vstack
import io
import distutils.dir_util
from collections import Counter
import scipy.sparse as spr
import pickle
import warnings
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import os
print(torch.__version__)
#os.environ["CUDA_VISIBLE_DEVICES"]='2'
#CUDA_VISIBLE_DEVICE=3
#torch.cuda.device(3)
#torch.cuda.get_device_name(3)
def write_json(data, fname):
    def _conv(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding='utf8') as f:
        json_obj = json.load(f)

    return json_obj


def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))

# from arena_util import load_json
class CustomEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        rec_playlists = load_json(rec_fname)
        
        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname):
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print("Music nDCG: {}".format(music_ndcg))
            print("Tag nDCG: {}".format(tag_ndcg))
            print("Score: {}".format(score))
        except Exception as e:
            print(e)



song_meta = pd.read_json("./kakaoarena/song_meta.json")
train = pd.read_json("./arena_data/orig/train.json")
test = pd.read_json("./arena_data/questions/val.json")

train['istrain'] = 1 # train 데이터에는 1을 줌
test['istrain'] = 0 # test 데이터에는 0을 줌

n_train = len(train) # train의 개수
n_test = len(test) # test의 개수

# train + test
plylst = pd.concat([train, test], ignore_index=True) # train + test를 합친 plylst

# playlist id
plylst["nid"] = range(n_train + n_test) # 0부터 train + test 합친 개수만큼 번호를 줌

# id <-> nid -> 고유한 인덱스 번호를 주기 위해
plylst_id_nid = dict(zip(plylst["id"],plylst["nid"])) 
plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))



plylst_tag = plylst['tags']
tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
tag_dict = {x: tag_counter[x] for x in tag_counter}


tag_id_tid = dict()
tag_tid_id = dict()
for i, t in enumerate(tag_dict):
  tag_id_tid[t] = i
  tag_tid_id[i] = t

n_tags = len(tag_dict)

plylst_song = plylst['songs']
song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
song_dict = {x: song_counter[x] for x in song_counter}

song_id_sid = dict()
song_sid_id = dict()
for i, t in enumerate(song_dict):
  song_id_sid[t] = i
  song_sid_id[i] = t

n_songs = len(song_dict)

plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])

plylst_use = plylst[['istrain','nid','songs_id','tags_id']]
plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
plylst_use = plylst_use.set_index('nid')


plylst_train = plylst_use.iloc[:n_train,:]
plylst_test = plylst_use.iloc[n_train:,:]


test = plylst_test
#print(test)
row = np.repeat(range(n_train), plylst_train['num_songs'])
col = [song for songs in plylst_train['songs_id'] for song in songs]
dat = np.repeat(1, plylst_train['num_songs'].sum())
train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

row = np.repeat(range(n_train), plylst_train['num_tags'])
col = [tag for tags in plylst_train['tags_id'] for tag in tags]
dat = np.repeat(1, plylst_train['num_tags'].sum())
train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_tags))

train_songs_A_csr = train_songs_A.tocsr()
train_tags_A_csr = train_tags_A.tocsr()
WT = csr_matrix.count_nonzero(train_songs_A)/(train_songs_A.shape[0] * train_songs_A.shape[1])
Wt = csr_matrix.count_nonzero(train_tags_A)/(train_tags_A.shape[0] * train_tags_A.shape[1])
print(Wt)
j=1
print(j)

x_train_song = train_songs_A_csr
batch_size=1
epochs=1
original_dim= n_songs # number of songs


########AutoEncoder _ songs 

class AE(nn.Module):
    def __init__(self,original_dim):
        super(AE,self).__init__()
        self.fc1=nn.Linear(original_dim,2048)
        #self.fc2=nn.Linear(2048,512)
        #self.fc3=nn.Linear(512,2048)
        self.fc4=nn.Linear(2048,original_dim)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        out=F.relu(self.fc4(x))
        return torch.sigmoid(out)

model=AE(original_dim).cuda()
model=nn.DataParallel(model)

'''
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)   

model.apply(init_weights)
'''

def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))



#criterion=nn.BCELoss()
weight = torch.tensor([WT,1])


optimizer=optim.Adam(model.parameters(),lr=0.01)  

j=1
print(j)


history = []
def train(model,x_train_song,epochs):
    
  for epoch in range(epochs):
     start = time.time()
     print("song epoch : ", epoch + 1)
     n_batches_for_epoch = x_train_song.shape[0]//batch_size + 1
     
     for i in range(n_batches_for_epoch):
        index_batch = range(x_train_song.shape[0])[batch_size*i:batch_size*(i+1)]       
        X_batch = x_train_song[index_batch,:].toarray().astype("float32")
        X_batch=torch.from_numpy(X_batch).cuda()
        X_batch.requires_grad_(True) 
        
        
        optimizer.zero_grad()
        output=model(X_batch).cuda()
        #print(output)
        print(output.shape)
        loss_func=weighted_binary_cross_entropy(output,X_batch,weight).cuda()
        del output
        #loss_func=criterion(X_batch,output.detach())
        #print(loss_func)
        loss_func.backward()  
        optimizer.step()
        
            
     history.append(loss_func)    
        #history.append(model.train_on_batch(X_batch, X_batch))
        

     end = time.time()
     print("song epoch time : ", end - start)


train(model,x_train_song,epochs)

#Graph
loss = history
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.title('Training loss') 
plt.legend()
plt.show()
plt.savefig("song.png")


####################AutoEncoder _ tags


x_train_tag = train_tags_A_csr
batch_size=1
epochs=1
original_dim= n_tags # number of tags


class AE_tag(nn.Module):
    def __init__(self,input):
        super(AE_tag,self).__init__()
        self.fc1=nn.Linear(input,1024)
        self.fc2=nn.Linear(1024,256)
        self.fc3=nn.Linear(256,1024)
        self.fc4=nn.Linear(1024,input)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        return torch.sigmoid(x)
model2=AE_tag(original_dim).cuda()
model2=nn.DataParallel(model2)
model2.apply(init_weights)
print(model2)

weight = torch.tensor([Wt,1])
optimizer=optim.Adam(model2.parameters(),lr=0.01)   

j=1
print(j)


history = []
def train(model2,x_train_tag,epochs):
    
  for epoch in range(epochs):
     start = time.time()
     print("tag epoch : ", epoch + 1)
     n_batches_for_epoch = x_train_tag.shape[0]//batch_size + 1
     
     for i in range(n_batches_for_epoch):
        index_batch = range(x_train_tag.shape[0])[batch_size*i:batch_size*(i+1)]       
        X_batch = x_train_tag[index_batch,:].toarray().astype("float32")
        X_batch=torch.from_numpy(X_batch).cuda()
        X_batch.requires_grad_(True)
        
        #
        optimizer.zero_grad()
        output=model2(X_batch).cuda()
        #print(output)
        #print(output.shape)
        loss_func=weighted_binary_cross_entropy(output,X_batch,weight).cuda()
        del output
        #loss_func=criterion(X_batch,output.detach())
        print(loss_func)
        loss_func.backward()  
        optimizer.step()
        
            
     history.append(loss_func)    
        #history.append(model.train_on_batch(X_batch, X_batch))
        

     end = time.time()
     print("song epoch time : ", end - start)


train(model2,x_train_tag,epochs)


loss = history
epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.title('Training loss') 
plt.legend()

plt.show()
plt.savefig("tag.png")

def rec(pids):
    tt = 1

    res = []

    for pid in pids:
        
    
        p = np.zeros((n_songs,1))
        p[test.loc[pid,'songs_id']] = 1
        p = p.reshape(1,-1)
        songs_already = test.loc[pid, "songs_id"]
        tags_already = test.loc[pid, "tags_id"]

        # cand_song : 노래 예측 -> 오토인코더의 결과로 대체
        cand_song = model(p) # p를 넣으면 결과를 뱉어줌.
        cand_song_idx = cand_song.reshape(-1).argsort()[-150:][::-1] # 빠질 곡들이 있을 수도 있으므로 넉넉하게 150개 + 역순
        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100] # 겹치지 않는 곡들 100개
        rec_song_idx = [song_sid_id[i] for i in cand_song_idx]

        # cand_tag : 태그 예측
        t = np.zeros((n_tags,1))
        t[test.loc[pid,'tags_id']] = 1
        t = t.reshape(1,-1)
        cand_tag = model2(t)
        cand_tag_idx = cand_tag.reshape(-1).argsort()[-15:][::-1]
        cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
        rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]

        res.append({
                    "id": plylst_nid_id[pid],
                    "songs": rec_song_idx,
                    "tags": rec_tag_idx
                })
    
        if tt % 1000 == 0:
            print(tt)

        tt += 1
    return res
start = time.time()
answers = rec(test.index)
end = time.time()
print("저장 시간 : ", end - start)
write_json(answers, "results/results.json")
print(pd.read_json("./arena_data/results/results.json"))
evaluator = CustomEvaluator()
evaluator.evaluate("./arena_data/answers/val.json", "./arena_data/results/results.json")
