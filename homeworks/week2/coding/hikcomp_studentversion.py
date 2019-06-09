import os,sys,numpy as np

import torch

import time

def forloopdists(feats1,feats2):

  #todo: done
  dist = np.empty((feats1.shape[0], feats2.shape[0]))

  for i in range(feats1.shape[0]):
    for j in range(feats2.shape[0]):
      dist[i,j] = np.dot(feats1[i,:] - feats2[j,:],
                         (feats1[i,:] - feats2[j,:].transpose()))

  return dist

def numpydists(feats1,feats2):
  #todo: done

  dist = -2 * np.matmul(feats1, feats2.transpose())
  dist += np.sum(feats1**2, axis=1)[:,np.newaxis]
  dist += np.sum(feats2**2, axis=1)[np.newaxis, :]

  return dist
  
def pytorchdists(feats1,feats2,device):
  
  #todo: done
  ft1 = torch.from_numpy(feats1).to(device)
  ft2 = torch.from_numpy(feats2).to(device)

  dist = -2 * torch.mm(ft1,ft2.t())
  dist += torch.sum(torch.pow(ft1, 2.0), dim=1).unsqueeze(1) + torch.sum(torch.pow(ft2,2.0),dim=1).unsqueeze(0)

  return dist.cpu().numpy()


def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############

  numdata1=250
  numdata2=300
  dims=30

  # genarate some random histogram data
  feats1=np.random.normal(size=(numdata1,dims))**2
  feats2=np.random.normal(size=(numdata2,dims))**2

  feats1=feats1/np.sum(feats1,axis=1)[:,np.newaxis]
  feats2=feats2/np.sum(feats2,axis=1)[:,np.newaxis]

  
  since = time.time()
  dists0=forloopdists(feats1,feats2)
  time_elapsed=float(time.time()) - float(since)
  print('for loop, Comp complete in {:.3f}s'.format( time_elapsed ))


  device=torch.device('cpu')
  since = time.time()

  dists1=pytorchdists(feats1,feats2,device)


  time_elapsed=float(time.time()) - float(since)

  print('pytorch, Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists1.shape)

  #print('df0',np.max(np.abs(dists1-dists0)))


  since = time.time()

  dists2=numpydists(feats1,feats2)


  time_elapsed=float(time.time()) - float(since)

  print('numpy, Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df',np.max(np.abs(dists1-dists2)))


if __name__=='__main__':
  run()
