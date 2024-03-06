import  warnings
warnings.filterwarnings("ignore")
import  torch

x=torch.tensor([[2,1],[5,6],[3,7],[12,0]])
y=torch.tensor([0,1,0,1],dtype=torch.float)

edge_index=torch.tensor([[0,1,2,0,3],
                         [1,0,1,3,2]],dtype=torch.long)

from torch_geometric.data import Data

data=Data(x=x,y=y,edge_index=edge_index)


from sklearn.preprocessing import LabelEncoder
import pandas as pd

df=pd.read_csv('yoochoose-clicks.dat',header=None)
df.columns=['session_id','timestamp','item_id','category']

buy_df=pd.read_csv('yoochoose-buys.dat',header=None)
buy_df.columns=['session_id','timestamp','item_id','price','quantity']

item_encoder=LabelEncoder()
df['item_id']=item_encoder.fit_transform(df.item_id)
df.head()

import numpy as np

sampled_session_id=np.random.choice(df.session_id.unique(),10000,replace=False)
df=df.loc[df.session_id.isin(sampled_session_id)]
df.nunique()

df['label']=df.session_id.isin(buy_df.session_id)
df.head()
#
# from torch_geometric.data import InMemoryDataset
# from tqdm import tqdm
# df_test=df[:100]
# grouped=df_test.groupby('session_id')
# for session_id, group in tqdm(grouped):
#     print('session_id:',session_id)
#     sess_item_id=LabelEncoder().fit_transform(group.item_id)
#     print('sess_item_id:',sess_item_id)
#     group=group.reset_index(drop=True)
#     group['sess_item_id']=sess_item_id
#     print('group:',group)
#     node_features=group.loc[group.session_id==session_id,['sess_item_id','item_id']].sort_values('sess_item_id').item_id.drop_duplicates().values
#     node_features=torch.LongTensor(node_features).unsqueeze(1)
#     print('node_features:',node_features)
#     target_nodes=group.sess_item_id.values[1:]
#     source_nodes=group.sess_item_id.values[:-1]
#     print('target_nodes:',target_nodes)
#     print('source_nodes:',source_nodes)
#     edge_index=torch.tensor([source_nodes,target_nodes],dtype=torch.long)
#     x=node_features
#     y=torch.FloatTensor([group.label.values[0]])
#     data=Data(x=x,edge_index=edge_index,y=y)
#     print('data:',data)

from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property  # python装饰器， 只读属性，方法可以像属性一样访问
    def raw_file_names(self):  # ①检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # ②如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):  # ③检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，有则直接加载
        # ④没有就会走process,得到'yoochoose_click_binary_1M_sess.dataset'文件
        return ['yoochoose_click_binary_1M_sess.dataset']

    def download(self):  # ①检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # ②如有文件不存在，则调用download()方法执行原始文件下载
        pass

    def process(self):  # ④没有就会走process,得到'yoochoose_click_binary_1M_sess.dataset'文件

        data_list = []  # 保存最终生成图的结果

        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])
            # 创建图
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)  # 转换成可以保存到本地的格式
        torch.save((data, slices), self.processed_paths[0])  # 保存操作，名字跟yoochoose_click_binary_1M_sess.dataset一致

dataset=YooChooseBinaryDataset(root='data/')

embed_dim=128
from torch_geometric.nn import TopKPooling,SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1=SAGEConv(embed_dim,128)
        self.pool1=TopKPooling(128,ratio=0.8)
        self.conv2=SAGEConv(128,128)
        self.pool2=TopKPooling(128,ratio=0.8)
        self.conv3=SAGEConv(128,128)
        self.pool3=TopKPooling(128,ratio=0.8)
        self.item_embedding=torch.nn.Embedding(num_embeddings=df.item_id.max()+10,embedding_dim=embed_dim)
        self.lin1=torch.nn.Linear(128,128)
        self.lin2=torch.nn.Linear(128,64)
        self.lin3=torch.nn.Linear(64,1)
        self.bn1=torch.nn.BatchNorm1d(128)
        self.bn2=torch.nn.BatchNorm1d(64)
        self.act1=torch.nn.ReLU()
        self.act2=torch.nn.ReLU()

    def forward(self,data):
        x,edge_index,batch=data.x,data.edge_index,data.batch
        x=self.item_embedding(x)
        x=x.squeeze(1)

        x=F.relu(self.conv1(x,edge_index))

        x,edge_index, _,batch, _, _=self.pool1(x,edge_index,None,batch)

        x1=gap(x,batch)

        x=F.relu(self.conv2(x,edge_index))

        x,edge_index, _,batch, _, _=self.pool2(x,edge_index,None,batch)

        x2=gap(x,batch)

        x=F.relu(self.conv3(x,edge_index))

        x,edge_index, _,batch, _, _=self.pool3(x,edge_index,None,batch)

        x3=gap(x,batch)

        x=x1+x2+x3

        x=self.lin1(x)
        x=self.act1(x)
        x=self.lin2(x)
        x=self.act2(x)
        x=F.dropout(x,p=0.5,training=self.training)

        x=torch.sigmoid(self.lin3(x)).squeeze(1)

        return x


from torch_geometric.loader import  DataLoader

model=Net()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
crit=torch.nn.BCELoss()
train_loader=DataLoader(dataset,batch_size=64)

def train():
    model.train()

    loss_all=0

    for data in train_loader:
        data=data

        optimizer.zero_grad()
        output=model(data)
        label=data.y
        loss=crit(output,label)
        loss.backward()
        loss_all+=data.num_graphs*loss.item()
        optimizer.step()

    return loss_all/len(dataset)

for epoch in range(10):
    print('epoch:',epoch)
    loss=train()
    print(loss)

from sklearn.metrics import roc_auc_score

def evalute(loader,model):
    model.eval()

    prediction=[]
    labels=[]

    with torch.no_grad():

        for data in loader:
            data=data
            pred=model(data)

            label=data.y
            prediction.append(pred)
            labels.append(label)

    prediction=np.hstack(prediction)
    labels=np.hstack(labels)

    return roc_auc_score(labels,prediction)

for epoch in range(1):
    roc_auc_score=evalute(dataset,model)
    print('roc_auc_score',roc_auc_score)
