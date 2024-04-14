 import numpy as np
import scipy.sparse as sp
import pickle
import datetime
from tqdm import tqdm

print(datetime.datetime.now())

np.random.seed(30)  #make generate the same metapath when running this code

with open("data.pkl", 'rb') as fs:
    data = pickle.load(fs)

(trainMat, _, trustMat, categoryMat, categoryDict) = data

trainMat=trainMat.tocsr()

userNum, itemNum = trainMat.shape


#UIU
UIU_mat = sp.dok_matrix((userNum, userNum))
for i in tqdm(range(userNum)):
    data = trainMat[i].toarray()
    _, iid = np.where(data != 0)
    uidList, _ = np.where(np.sum(trainMat[:, iid]!=0, axis=1) != 0)
    uidList = uidList.tolist()
    tmp = [i] * len(uidList)
    UIU_mat[tmp, uidList] = 1

UIU_mat = UIU_mat.tocsr()
UIU_mat = UIU_mat + UIU_mat.T + sp.eye(userNum).tocsr()  
UIU_mat = (UIU_mat != 0)


#UITIU
UITIU_mat = sp.dok_matrix((userNum, userNum))
for i in tqdm(range(userNum)):
    data = trainMat[i].toarray()
    _, iid = np.where(data != 0)
    typeidList = categoryMat[iid].toarray().reshape(-1)
    typeidSet = set(typeidList.tolist())
    for typeid in typeidSet:
        iid2 = categoryDict[typeid]
        uidList, _ = np.where(np.sum(trainMat[:, iid2]!=0, axis=1) != 0)
        uidList2 = np.random.choice(uidList, size=int(uidList.size*0.1), replace=False)
        uidList2 = uidList2.tolist()
        tmp = [i]*len(uidList2)
        UITIU_mat[tmp, uidList2] = 1
UITIU_mat = UITIU_mat.tocsr()
UITIU_mat = UITIU_mat + UITIU_mat.T + sp.eye(userNum).tocsr()  
UITIU_mat = (UITIU_mat != 0)





#IUI
IUI_mat = sp.dok_matrix((itemNum, itemNum))
trainMat_T = trainMat.T
for i in tqdm(range(itemNum)):
    data = trainMat_T[i].toarray()
    _, uid = np.where(data != 0)
    iidList, _ = np.where(np.sum(trainMat_T[:, uid] != 0, axis=1) != 0)
    iidList2 = np.random.choice(iidList, size=int(iidList.size*0.1), replace=False)
    iidList2 = iidList2.tolist()
    tmp = [i]*len(iidList2)
    IUI_mat[tmp, iidList2] = 1
IUI_mat = IUI_mat.tocsr()
IUI_mat = IUI_mat + IUI_mat.T + sp.eye(itemNum).tocsr() 
IUI_mat = (IUI_mat != 0)


print(datetime.datetime.now())

metaPath = {}
metaPath['UIU'] = UIU_mat
metaPath['UITIU'] = UITIU_mat
metaPath['IUI'] = IUI_mat

with open('metaPath.pkl', 'wb') as fs:
    pickle.dump(metaPath, fs)
print("Done")


