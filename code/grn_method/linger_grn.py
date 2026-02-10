# %%
from LingerGRN.preprocess import *
import pandas as pd
from scipy.sparse import csc_matrix
import scanpy as sc
from LingerGRN.pseudo_bulk import *
import json
import sys
import torch

# %%


# %%
method='LINGER'

from nvitop import Device
devices = Device.all()
memory_free = [device.memory_free() for device in devices]
gpu_index = memory_free.index(max(memory_free))
torch.cuda.set_device(gpu_index)


# %%
## load paramters and configs.
# input_path = "/home/shaliu_fu/multireg/benchmark/bench_dataset/scalability/c1k/"
# output_path = "/home/shaliu_fu/multireg/benchmark/output/scalability/c1k/LINGER_output/"
input_path = sys.argv[1]
output_path = sys.argv[2]

config_path = "{}/RawData.json".format(input_path)
if os.path.exists(config_path):
    data_json = json.load(open(config_path))
else:
    print(f"{config_path} not found! Exit!")
    sys.exit(1)

# %%

rna= sc.read_h5ad(f"{input_path}/{data_json['rna_h5ad_filename']}")
atac = sc.read_h5ad(f"{input_path}/{data_json['atac_h5ad_filename']}")
cell_meta = pd.read_csv(f"{input_path}/{data_json['metadata']}",header=0)
cell_meta['barcode_use'] = cell_meta[data_json['barcode_key']]
cell_meta['label'] = [i.replace("/","_") for i in cell_meta[data_json['celltype_key']]]
cell_meta.reset_index(drop=True,inplace=True)
rna.obs = cell_meta

gtf_file=data_json['gtf_file']
max_cpu=30
gtf_col="gene_name"

if not os.path.exists(output_path):
    os.system("mkdir -p {}".format(output_path))
torch.set_num_threads(max_cpu)


# %%
peaks = [i.split("-") for i in atac.var.index.values]
peaks2 = [f"{i[0]}:{i[1]}-{i[2]}" for i in peaks]
atac.var.index = peaks2


from scipy.sparse import csc_matrix
# Convert the NumPy array to a sparse csc_matrix
matrix = csc_matrix(np.vstack([rna.X.T, atac.X.T]))
features=pd.DataFrame(rna.var.index.tolist()+atac.var.index.tolist(),columns=[1])
K=rna.X.shape[1]
N=K+atac.X.shape[1]
types = ['Gene Expression' if i <= K else 'Peaks' for i in range(0, N)]
features[2]=types
# barcodes=pd.DataFrame(rna.obs.index.values,columns=[0])
barcodes=pd.DataFrame(cell_meta['barcode_use'].values,columns=[0])
from LingerGRN.preprocess import *
adata_RNA,adata_ATAC=get_adata(matrix,features,barcodes,cell_meta)# adata_RNA and adata_ATAC are scRNA and scATAC

# %%


# %%
from LingerGRN.pseudo_bulk import *
samplelist=list(set(adata_ATAC.obs['sample'].values)) # sample is generated from cell barcode 
tempsample=samplelist[0]
TG_pseudobulk=pd.DataFrame([])
RE_pseudobulk=pd.DataFrame([])
singlepseudobulk = (adata_RNA.obs['sample'].unique().shape[0]*adata_RNA.obs['sample'].unique().shape[0]>100)
for tempsample in samplelist:
    adata_RNAtemp=adata_RNA[adata_RNA.obs['sample']==tempsample]
    adata_ATACtemp=adata_ATAC[adata_ATAC.obs['sample']==tempsample]
    TG_pseudobulk_temp,RE_pseudobulk_temp=pseudo_bulk(adata_RNAtemp,adata_ATACtemp,singlepseudobulk)                
    TG_pseudobulk=pd.concat([TG_pseudobulk, TG_pseudobulk_temp], axis=1)
    RE_pseudobulk=pd.concat([RE_pseudobulk, RE_pseudobulk_temp], axis=1)
    RE_pseudobulk[RE_pseudobulk > 100] = 100

import os
if not os.path.exists(f'{output_path}/data/'):
    os.mkdir(f'{output_path}/data/')
adata_ATAC.write(f'{output_path}/data/adata_ATAC.h5ad')
adata_RNA.write(f'{output_path}/data/adata_RNA.h5ad')
TG_pseudobulk=TG_pseudobulk.fillna(0)
RE_pseudobulk=RE_pseudobulk.fillna(0)

# pd.DataFrame(adata_ATAC.var['gene_ids']).to_csv(f'{outdir}/data/Peaks.txt',header=None,index=None)
pd.DataFrame(RE_pseudobulk.index.values).to_csv(f'{output_path}/data/Peaks.txt',header=None,index=None)
TG_pseudobulk.to_csv(f'{output_path}/data/TG_pseudobulk.tsv')
RE_pseudobulk.to_csv(f'{output_path}/data/RE_pseudobulk.tsv')

# %%


# %%


# %%
from LingerGRN.preprocess import *
# Datadir='/path/to/LINGER/'# This directory should be the same as Datadir defined in the above 'Download the general gene regulatory network' section
GRNdir='/home/shaliu_fu/software/bench_tools/LINGER/model/data_bulk/'
if data_json['species'] == 'human':
    genome='hg38'
elif data_json['species'] == 'mouse':
    genome='mm10'
else:
    print(f"Unknown species:{data_json['species']}")
    sys.exit(1)
# outdir='absolute path' #output dir
preprocess(TG_pseudobulk,RE_pseudobulk,GRNdir,genome,method,output_path)

# %%


# %%
import LingerGRN.LINGER_tr as LINGER_tr
activef='ReLU' # active function chose from 'ReLU','sigmoid','tanh'
LINGER_tr.training(GRNdir,method,output_path,activef,'Human')

# %%
import LingerGRN.LL_net as LL_net
# for method in ['baseline','LINGER']:
# for method in ['LINGER']:
LL_net.TF_RE_binding(GRNdir,adata_RNA,adata_ATAC,genome,method,output_path)



# %%


# %%
LL_net.cis_reg(GRNdir,adata_RNA,adata_ATAC,genome,method,output_path)
LL_net.trans_reg(GRNdir,method,output_path,genome)

# %%
for celltype in set(adata_RNA.obs['label'].values):
    LL_net.cell_type_specific_TF_RE_binding(GRNdir,adata_RNA,adata_ATAC,genome,celltype,output_path,method)
    # LL_net.cell_type_specific_cis_reg(GRNdir,adata_RNA,adata_ATAC,genome,celltype,outdir,method)
    LL_net.cell_type_specific_cis_reg(GRNdir,adata_RNA,adata_ATAC,genome,celltype,output_path,method)
    # LL_net.cell_type_specific_trans_reg(GRNdir,adata_RNA,celltype,outdir)

    LL_net.cell_type_specific_trans_reg(GRNdir,adata_RNA,celltype,output_path)

# %%


pid= os.getpid()        
gpu_memory = pd.Series(dtype='str')

devices = Device.all()
for device in devices:
    processes = device.processes()    
    if pid in processes.keys():
        p=processes[pid]
        gpu_memory['device ' + str(device.index)] = p.gpu_memory_human()

gpu_memory.to_csv(os.path.join(output_path, data_json['output_prefix'] + '-GLUE-gpu_memory.csv'), header=["gpu_memory"])
