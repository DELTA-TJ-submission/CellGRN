# %%

import os
import warnings
warnings.filterwarnings("ignore")

from pycistarget.utils import *
from pycistarget.motif_enrichment_cistarget import *
from pycisTopic.cistopic_class import *
import pandas as pd
import pyranges as pr
import scanpy as sc
import pickle
from pycisTopic.lda_models import run_cgs_model
import json
import sys
import pandas as pd

import scipy as sp

# %%
# input_path = "/home/shaliu_fu/multireg/benchmark/bench_dataset/snare_cellline/"
# output_path = "/home/shaliu_fu/multireg/benchmark/output/snare_cellline/scenic2_output"
# soft_config_file = "../algor_config.json"
input_path = sys.argv[1]
output_path = sys.argv[2]
soft_config_file = sys.argv[3]

config_path = "{}/RawData.json".format(input_path)
if os.path.exists(config_path):
    data_json = json.load(open(config_path))
else:
    print(f"{config_path} not found! Exit!")
    sys.exit(1)

# %%
soft_config = json.load(open(soft_config_file))
soft_sel = soft_config['scenic2']
species = data_json['species']
rankings_db = soft_sel['db'][species]['rankings_db']
scores_db =  soft_sel['db'][species]['scores_db']
motif_annotation = soft_sel['db'][species]['motif_annotation']
blacklist_file= soft_sel['db'][species]['blacklist_file']
mallet_path=soft_sel['mallet_path']



# %%

rna = sc.read_h5ad(f"{input_path}/{data_json['rna_h5ad_filename']}")

atac = sc.read_h5ad(f"{input_path}/{data_json['atac_h5ad_filename']}")
metadata = pd.read_csv(f"{input_path}/{data_json['metadata']}",header=0)
metadata.index=metadata[data_json['barcode_key']]

rna.obs = metadata

# %%
rna.raw = rna
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
rna = rna[:, rna.var.highly_variable]
sc.pp.scale(rna, max_value=10)

# %%
sc.tl.pca(rna)
sc.pp.neighbors(rna)
sc.tl.umap(rna)
rna.write(f"{output_path}/rna.h5ad")


# %%
if sp.sparse.issparse(atac.X):
    atac_data = pd.DataFrame(atac.X.A)
else:
    atac_data = pd.DataFrame(atac.X)
atac_data.index = atac.obs.index.values
atac_data.columns = [i.replace("-",":",1) for i in atac.var.index.values]

cistopic_obj = create_cistopic_object(fragment_matrix=atac_data.T, # col: cells, row: peaks
                                      project=data_json['output_prefix'])

# %%
cistopic_obj.add_cell_data(metadata)

# %%
from pycisTopic.lda_models import run_cgs_models_mallet
os.environ['MALLET_MEMORY'] = '200G'
if not os.path.exists(f"{output_path}/tmp/"):
    os.system(f"mkdir -p {output_path}/tmp/")

models=run_cgs_models_mallet(cistopic_obj,
    n_topics=[2, 5, 10, 15, 20, 25, 30, 35, 40,45,50],
    n_cpu=20,
    n_iter=500,
    random_state=555,
    alpha=50,
    alpha_by_topic=True,
    eta=0.1,
    eta_by_topic=False,
    tmp_path=f"{output_path}/tmp/",
    save_path=f"{output_path}/tmp/",
    mallet_path=mallet_path,
)

# %%
pickle.dump(models,
            open(f"{output_path}/tmp/snare_cellline_models_500_iter_LDA.pkl", 'wb'))

# %%
from pycisTopic.lda_models import evaluate_models
model = evaluate_models(
    models,
    return_model = True
)
cistopic_obj.add_LDA_model(model)


# %%
from pycisTopic.topic_binarization import binarize_topics

# 导出peaks
region_bin_topics_otsu = binarize_topics(
    cistopic_obj, method='otsu',
    plot=False, num_columns=5
)

region_bin_topics_top_3k = binarize_topics(
    cistopic_obj, method='ntop', ntop = 3_000,
    plot=False, num_columns=5
)

# %%
# 然后是算差异。
from pycisTopic.diff_features import (
    impute_accessibility,
    normalize_scores,
    find_highly_variable_features,
    find_diff_features
)
import numpy as np

imputed_acc_obj = impute_accessibility(
    cistopic_obj,
    selected_cells=None,
    selected_regions=None,
    scale_factor=10**6
)
normalized_imputed_acc_obj = normalize_scores(imputed_acc_obj, scale_factor=10**4)
variable_regions = find_highly_variable_features(
    normalized_imputed_acc_obj,
    min_disp = 0.05,
    min_mean = 0.0125,
    max_mean = 3,
    max_disp = np.inf,
    n_bins=20,
    n_top_features=None,
    plot=False
)



# %%
from pycisTopic.utils import region_names_to_coordinates


# %%
markers_dict= find_diff_features(
    cistopic_obj,
    imputed_acc_obj,
    variable=data_json['celltype_key'],
    var_features=variable_regions,
    contrasts=None,
    adjpval_thr=0.05,
    log2fc_thr=np.log2(1.5),
    n_cpu=5,
    _temp_dir=None,
    split_pattern = '-'
)

# %%
out_dir = f"{output_path}/regions/Topics_otsu/"
if not os.path.exists(out_dir):
    os.system("mkdir -p {}".format(out_dir))
for topic in region_bin_topics_otsu:
    region_names_to_coordinates(
        region_bin_topics_otsu[topic].index
    ).sort_values(
        ["Chromosome", "Start", "End"]
    ).to_csv(
        os.path.join(out_dir, f"Topics_otsu_{topic}.bed"),
        sep = "\t",
        header = False, index = False
    )

# %%
out_dir = f"{output_path}/regions/DARs_cell_type/"
if not os.path.exists(out_dir):
    os.system("mkdir -p {}".format(out_dir))

for cell_type in markers_dict:
    if len(markers_dict[cell_type].index) >1:
        region_names_to_coordinates(
            markers_dict[cell_type].index
        ).sort_values(
            ["Chromosome", "Start", "End"]
        ).to_csv(
            os.path.join(out_dir, "DARs_cell_type_{0}.bed".format(cell_type.replace("/","_"))), # avoid / in cell types!!
            sep = "\t",
            header = False, index = False
        )

# %%
out_dir = f"{output_path}/regions/Topics_top_3k/"
if not os.path.exists(out_dir):
    os.system("mkdir -p {}".format(out_dir))
    
for topic in region_bin_topics_top_3k:
    region_names_to_coordinates(
        region_bin_topics_top_3k[topic].index
    ).sort_values(
        ["Chromosome", "Start", "End"]
    ).to_csv(
        os.path.join(out_dir,  f"Topics_top_3k_{topic}.bed"),
        sep = "\t",
        header = False, index = False
    )

# %%

pickle.dump(cistopic_obj, open( f'{output_path}/cistopic_obj.pkl', 'wb'))

# %%
config_file = "{}/scenic2_config.yaml".format(soft_config['scenic2']['module_folder'])


# %%


# scenic_config = yaml.safe_load(open(config_file))

# scenic_config['input_data']['cisTopic_obj_fname'] = f'{output_path}/cistopic_obj.pkl'
# scenic_config['input_data']['GEX_anndata_fname'] = f'{output_path}/rna.h5ad'
# scenic_config['input_data']['region_set_folder'] = f'{output_path}/regions/'
# scenic_config['input_data']['ctx_db_fname']  = soft_config['scenic2']['db'][data_json['species']]['rankings_db']
# scenic_config['input_data']['dem_db_fname']  = soft_config['scenic2']['db'][data_json['species']]['scores_db']
# scenic_config['input_data']['path_to_motif_annotations'] = soft_config['scenic2']['db'][data_json['species']]['motif_annotation']

# if data_json['species'] == 'human':
#     scenic_config['params_motif_enrichment']['species'] = "homo_sapiens"
# elif data_json['species'] == 'mouse':
#     scenic_config['params_motif_enrichment']['species'] = "mus_musculus"
# scenic_config['params_general']['temp_dir'] = f'{output_path}/tmp/'
# scenic_config['params_general']['n_cpu'] = 20

# scenic_config["params_data_preparation"]["bc_transform_func"] = r"\"lambda x: f'{x}___" +data_json['output_prefix']+ r"'\""


# class DoubleQuotedDumper(yaml.Dumper):
#     def represent_str(self, data):
#         return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    
# with open(f"{output_path}/Snakemake/config/config.yaml","w") as fo:
#     yaml.dump(scenic_config, fo, Dumper=DoubleQuotedDumper, sort_keys=False, default_flow_style=False, indent=1)

# %%
## 不能用yaml,直接读入字符串修改。

scenic_config = [i.rstrip() for i in open(config_file)]


# %%
scenic_config[1] = scenic_config[1][:22] + f'"{output_path}/cistopic_obj.pkl"'
scenic_config[2] = scenic_config[2][:21] + f'"{output_path}/rna.h5ad"'
scenic_config[3] = scenic_config[3][:21] + f'"{output_path}/regions/"'
scenic_config[4] = scenic_config[4][:16] + '"{}"'.format(soft_config['scenic2']['db'][data_json['species']]['rankings_db'])
scenic_config[5] = scenic_config[5][:16] + '"{}"'.format(soft_config['scenic2']['db'][data_json['species']]['scores_db'])
scenic_config[6] = scenic_config[6][:29] + '"{}"'.format(soft_config['scenic2']['db'][data_json['species']]['motif_annotation'])
scenic_config[41] = scenic_config[41][:12]+f'"{output_path}/tmp/"'
scenic_config[47] = scenic_config[47][:21]+"\""+r"\"lambda x: f'{x}___" +data_json['output_prefix']+ r"'\""+"\""
if data_json['species'] == 'mouse':
    scenic_config[55] = scenic_config[55][:11] + '"{}"'.format("mmusculus")
    scenic_config[63] = scenic_config[63][:11] + '"{}"'.format("mus_musculus")

# %%
with open(f"{output_path}/Snakemake/config/config.yaml","w") as fo:
    for line in scenic_config:
        fo.write(f"{line}\n")

# %%


# %%



