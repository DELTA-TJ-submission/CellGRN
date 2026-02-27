import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import os
import anndata as ad
from scipy import sparse


def normalzie_rna(gene_expr, temperature=0.1,rna_thres=0.5):

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    gene_name = gene_expr.columns.values
    cell_name = gene_expr.index.values
    gene_expr2 = gene_expr.to_numpy()

    rna_scaler = MinMaxScaler(feature_range=(0,1))
    gene_scaled_array = rna_scaler.fit_transform(gene_expr2.T).T   # shape: (n_cell, n_gene)

    gene_scaled = pd.DataFrame(gene_scaled_array, index=cell_name, columns=gene_name)

    # tf_th_fixed = 0.5
    sig_input_fixed = (gene_scaled_array - rna_thres) / temperature
    gene_scaled2_array = 1.0 / (1.0 + np.exp(-sig_input_fixed))
    gene_scaled2 = pd.DataFrame(gene_scaled2_array, index=cell_name, columns=gene_name)
    return gene_scaled, gene_scaled2


def normalzie_rna_sparse(gene_expr_sparse, temperature=0.1, rna_thres=0.5):

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaler = MaxAbsScaler()
    gene_scaled_sparse = scaler.fit_transform(gene_expr_sparse)

    if sparse.issparse(gene_scaled_sparse):
        gene_scaled_dense = gene_scaled_sparse.toarray().astype(np.float32)
    else:
        gene_scaled_dense = gene_scaled_sparse.astype(np.float32)

    sig_input_fixed = (gene_scaled_dense - rna_thres) / temperature
    sig_input_fixed = np.clip(sig_input_fixed, -50, 50) 
    gene_scaled2_array = 1.0 / (1.0 + np.exp(-sig_input_fixed))

    return gene_scaled_sparse, gene_scaled2_array


def parse_edges(cand_df, tf_list, gene_list, peak_list):

    tf_set = set(tf_list)
    gene_set = set(gene_list)
    peak_set = set(peak_list)

    tf_index = {name: i for i, name in enumerate(tf_list)}
    gene_index = {name: i for i, name in enumerate(gene_list)}
    peak_index = {name: i for i, name in enumerate(peak_list)}

    edges_idx = {
        'tf_gene': ([], []),    # (tf_idx_list, gene_idx_list)
        'tf_peak': ([], []),    # (tf_idx_list, peak_idx_list)
        'peak_gene': ([], [])   # (peak_idx_list, gene_idx_list)
    }
    edges_name = {
        'tf_gene': [],
        'tf_peak': [],
        'peak_gene': []
    }

    for _, row in cand_df.iterrows():
        edge_type = row['group_subtype']
        f1, f2 = row['feature1'], row['feature2']

        if edge_type == 'TF-Gene':
            if f1 in tf_set and f2 in gene_set:
                edges_idx['tf_gene'][0].append(tf_index[f1])
                edges_idx['tf_gene'][1].append(gene_index[f2])
                edges_name['tf_gene'].append(f"TF-Gene_{f1}_{f2}")

        elif edge_type == 'TF-Peak':
            if f1 in tf_set and f2 in peak_set:
                edges_idx['tf_peak'][0].append(tf_index[f1])
                edges_idx['tf_peak'][1].append(peak_index[f2])
                edges_name['tf_peak'].append(f"TF-Peak_{f1}_{f2}")

        elif edge_type == 'Peak-Gene':
            if f1 in peak_set and f2 in gene_set:
                edges_idx['peak_gene'][0].append(peak_index[f1])
                edges_idx['peak_gene'][1].append(gene_index[f2])
                edges_name['peak_gene'].append(f"Peak-Gene_{f1}_{f2}")

    return edges_idx, edges_name


def compute_cell_grn_scores(cell_id, tf_df, rna_df, atac_df, edges):

    tf_values = tf_df.loc[cell_id]
    gene_values = rna_df.loc[cell_id]
    peak_values = atac_df.loc[cell_id]
    
    scores = {}

    for tf, gene in edges['tf_gene']:
        key = f"TF-Gene_{tf}_{gene}"
        scores[key] = tf_values[tf] * gene_values[gene]
    
    for tf, peak in edges['tf_peak']:
        key = f"TF-Peak_{tf}_{peak}"
        scores[key] = tf_values[tf] * peak_values[peak]
    
    for peak, gene in edges['peak_gene']:
        key = f"Peak-Gene_{peak}_{gene}"
        scores[key] = peak_values[peak] * gene_values[gene]
    return scores

def compute_all_cells_grn(tf_df, rna_df, atac_df, 
                          edges_idx, edges_name,
                          tf_list, gene_list, peak_list):

    cell_ids = tf_df.index.tolist()

    tf_mat = tf_df[tf_list].to_numpy()      # (n_cell, n_tf)
    rna_mat = rna_df[gene_list].to_numpy()  # (n_cell, n_gene)
    
    if atac_df is None:
        raise ValueError("Must input atac_df to calculate peak-based scores.")
    
    atac_mat = atac_df[peak_list].to_numpy() # (n_cell, n_peak)

    all_edge_scores = []
    all_edge_names = []

    tf_idx, peak_idx = edges_idx['tf_peak']
    if len(tf_idx) > 0:
        tf_sub = tf_mat[:, tf_idx]
        peak_sub = atac_mat[:, peak_idx]
        tf_peak_scores = tf_sub * peak_sub
        all_edge_scores.append(tf_peak_scores)
        all_edge_names.extend(edges_name['tf_peak'])

    peak_idx, gene_idx = edges_idx['peak_gene']
    if len(peak_idx) > 0:
        peak_sub = atac_mat[:, peak_idx]
        gene_sub = rna_mat[:, gene_idx]
        peak_gene_scores = peak_sub * gene_sub
        all_edge_scores.append(peak_gene_scores)
        all_edge_names.extend(edges_name['peak_gene'])

    
    target_tf_idx, target_gene_idx = edges_idx['tf_gene']
    
    if len(target_tf_idx) > 0:

        tp_tf, tp_peak = edges_idx['tf_peak']
        df_tp = pd.DataFrame({'tf': tp_tf, 'peak': tp_peak})

        pg_peak, pg_gene = edges_idx['peak_gene']
        df_pg = pd.DataFrame({'peak': pg_peak, 'gene': pg_gene})
        
        df_target = pd.DataFrame({'tf': target_tf_idx, 'gene': target_gene_idx})
        df_target['edge_id'] = range(len(df_target))  
        
        df_paths = pd.merge(df_tp, df_pg, on='peak')
        
        df_valid_paths = pd.merge(df_paths, df_target, on=['tf', 'gene'])
 
        if len(df_valid_paths) > 0:

            path_peak_indices = df_valid_paths['peak'].values
            path_edge_indices = df_valid_paths['edge_id'].values
            
            n_total_peaks = atac_mat.shape[1]
            n_target_edges = len(df_target)
            
            peak_agg_matrix = sparse.coo_matrix(
                (np.ones(len(df_valid_paths)), (path_peak_indices, path_edge_indices)),
                shape=(n_total_peaks, n_target_edges)
            ).tocsr() 
            
            sum_peak_sq = (atac_mat ** 2) @ peak_agg_matrix
            
            tf_vals = tf_mat[:, target_tf_idx]
            gene_vals = rna_mat[:, target_gene_idx]
            
            tf_gene_scores = tf_vals * gene_vals * sum_peak_sq
            
        else:

            tf_gene_scores = np.zeros((len(cell_ids), len(target_tf_idx)))

        all_edge_scores.append(tf_gene_scores)
        all_edge_names.extend(edges_name['tf_gene'])

    if len(all_edge_scores) == 0:
        return pd.DataFrame(index=cell_ids)

    scores_mat = np.concatenate(all_edge_scores, axis=1)
    scores_df = pd.DataFrame(scores_mat, index=cell_ids, columns=all_edge_names)
    
    return scores_df

class SparseGRNCalculator:
    """
    CellGRN-sparse module
    """
    def __init__(self, edges_idx, edges_name, tf_list, gene_list, peak_list):
        self.edges_idx = edges_idx
        self.edges_name = edges_name
        self.tf_list = tf_list
        self.gene_list = gene_list
        self.peak_list = peak_list
        
        self._prepare_tf_gene_mapping()
        
        self.all_edge_names_flat = (
            self.edges_name['tf_peak'] + 
            self.edges_name['peak_gene'] + 
            self.edges_name['tf_gene']
        )
        self.n_edges = len(self.all_edge_names_flat)
        
        self.global_sum = np.zeros(self.n_edges, dtype=np.float64)
        self.total_cells = 0
        self.ct_accumulators = {}

    def _prepare_tf_gene_mapping(self):
        target_tf_idx, target_gene_idx = self.edges_idx['tf_gene']
        self.has_tf_gene = len(target_tf_idx) > 0
        self.peak_agg_matrix = None
        self.target_tf_idx = target_tf_idx
        self.target_gene_idx = target_gene_idx

        if self.has_tf_gene:
            tp_tf, tp_peak = self.edges_idx['tf_peak']
            df_tp = pd.DataFrame({'tf': tp_tf, 'peak': tp_peak})
            
            pg_peak, pg_gene = self.edges_idx['peak_gene']
            df_pg = pd.DataFrame({'peak': pg_peak, 'gene': pg_gene})
            
            df_target = pd.DataFrame({'tf': target_tf_idx, 'gene': target_gene_idx})
            df_target['edge_id'] = range(len(df_target))
            
            df_paths = pd.merge(df_tp, df_pg, on='peak')
            df_valid_paths = pd.merge(df_paths, df_target, on=['tf', 'gene'])
            
            if len(df_valid_paths) > 0:
                path_peak_indices = df_valid_paths['peak'].values
                path_edge_indices = df_valid_paths['edge_id'].values
                
                n_total_peaks = len(self.peak_list)
                n_target_edges = len(df_target)
                
                self.peak_agg_matrix = sparse.coo_matrix(
                    (np.ones(len(df_valid_paths), dtype=np.float32), 
                     (path_peak_indices, path_edge_indices)),
                    shape=(n_total_peaks, n_target_edges)
                ).tocsr()

    def process_batch(self, tf_mat_batch, rna_mat_batch, atac_mat_batch, cell_types_batch):
        batch_size = tf_mat_batch.shape[0]
        self.total_cells += batch_size
        
        if not sparse.issparse(atac_mat_batch):
            atac_mat_batch = sparse.csr_matrix(atac_mat_batch)

        batch_scores_list = []

        # 1. TF-Peak
        tf_idx, peak_idx = self.edges_idx['tf_peak']
        if len(tf_idx) > 0:
            tf_sub = tf_mat_batch[:, tf_idx] 
            peak_sub = atac_mat_batch[:, peak_idx].toarray() 
            batch_scores_list.append(tf_sub * peak_sub)

        # 2. Peak-Gene
        peak_idx, gene_idx = self.edges_idx['peak_gene']
        if len(peak_idx) > 0:
            peak_sub = atac_mat_batch[:, peak_idx].toarray()
            gene_sub = rna_mat_batch[:, gene_idx]
            batch_scores_list.append(peak_sub * gene_sub)

        # 3. TF-Gene
        if self.has_tf_gene:
            if self.peak_agg_matrix is not None:
                atac_sq = atac_mat_batch.power(2) 
                sum_peak_sq = (atac_sq @ self.peak_agg_matrix).toarray()
                
                tf_vals = tf_mat_batch[:, self.target_tf_idx]
                gene_vals = rna_mat_batch[:, self.target_gene_idx]
                
                batch_scores_list.append(tf_vals * gene_vals * sum_peak_sq)
            else:
                batch_scores_list.append(np.zeros((batch_size, len(self.target_tf_idx))))

        if not batch_scores_list:
            return

        batch_scores_mat = np.concatenate(batch_scores_list, axis=1)
        
        batch_sum = batch_scores_mat.sum(axis=0)
        self.global_sum += batch_sum
        
        unique_cts = np.unique(cell_types_batch)
        for ct in unique_cts:
            mask = (cell_types_batch == ct)
            ct_sum = batch_scores_mat[mask].sum(axis=0)
            ct_count = np.sum(mask)
            
            if ct not in self.ct_accumulators:
                self.ct_accumulators[ct] = {
                    'sum': np.zeros(self.n_edges, dtype=np.float64),
                    'count': 0
                }
            self.ct_accumulators[ct]['sum'] += ct_sum
            self.ct_accumulators[ct]['count'] += ct_count
            
        del batch_scores_mat, batch_scores_list

    def finalize(self):
        sample_grn_score = self.global_sum / self.total_cells
        sample_grn = pd.DataFrame({'score': sample_grn_score}, index=self.all_edge_names_flat)
        
        ct_res = {}
        for ct, data in self.ct_accumulators.items():
            ct_res[ct] = data['sum'] / data['count']
        
        celltype_grn = pd.DataFrame(ct_res, index=self.all_edge_names_flat)
        
        return sample_grn, celltype_grn

def parse_edge_index(index_series):
    parts = index_series.to_series().str.split("_", n=2, expand=True)
    parts.columns = ['group_subtype', 'feature1', 'feature2']
    return parts

def summarize_grn(grn_scores_df, cell_types_series):

    # sample-wise
    sample_grn = grn_scores_df.mean(axis=0) #.sort_values(ascending=False)
    sample_grn = pd.DataFrame({'score': sample_grn})
    
    # cell-type specific
    grn_with_ct = grn_scores_df.copy()
    grn_with_ct['cell_type'] = cell_types_series.values
    
    celltype_grn = grn_with_ct.groupby('cell_type').mean().T
    
    return sample_grn, celltype_grn


def parse_edge_index(index_series):

    parts = index_series.to_series().str.split("_", n=2, expand=True)
    parts.columns = ['group_subtype', 'feature1', 'feature2']
    return parts

def format_sample_grn(sample_grn):

    edge_df = parse_edge_index(sample_grn.index)
    edge_df = edge_df.join(sample_grn['score'])

    # --- TF-Gene ---
    tf_gene = edge_df[edge_df['group_subtype'] == 'TF-Gene'].copy()
    tf_gene_res = tf_gene.rename(columns={
        'feature1': 'TF',
        'feature2': 'Gene',
        'score': 'Score'
    })[['TF', 'Gene', 'Score']].nlargest(10000, "Score")

    # --- TF-Peak ---
    tf_peak = edge_df[edge_df['group_subtype'] == 'TF-Peak'].copy()
    tf_peak_res = tf_peak.rename(columns={
        'feature1': 'TF',
        'feature2': 'Peak',
        'score': 'Score'
    })[['TF', 'Peak', 'Score']].nlargest(20000, "Score")

    peak_gene = edge_df[edge_df['group_subtype'] == 'Peak-Gene'].copy()

    peak_gene_res = peak_gene.rename(columns={
        'feature1': 'Peak',
        'feature2': 'Gene',
        'score': 'Score'
    })

    gene_peak_res = peak_gene_res[['Gene', 'Peak', 'Score']].nlargest(20000, "Score")

    return tf_gene_res, tf_peak_res, gene_peak_res

def format_celltype_grn(celltype_grn,
                             k_tf_gene=10000,
                             k_tf_peak=20000,
                             k_gene_peak=20000):

    ct_df = celltype_grn.copy()
    ct_df['edge'] = ct_df.index

    long_df = ct_df.melt(
        id_vars='edge',
        var_name='cell_type',
        value_name='Score'
    )

    parts = long_df['edge'].str.split("_", n=2, expand=True)
    parts.columns = ['group_subtype', 'feature1', 'feature2']
    long_df = pd.concat([parts, long_df[['cell_type', 'Score']]], axis=1)

    # ===== TF-Gene =====
    tf_gene = long_df[long_df['group_subtype'] == 'TF-Gene'].copy()
    tf_gene = tf_gene.rename(columns={'feature1': 'TF', 'feature2': 'Gene'})
    tf_gene = tf_gene[['TF', 'Gene', 'cell_type', 'Score']]

    tf_gene_ct_res = (
        tf_gene
        .sort_values(['cell_type', 'Score'], ascending=[True, False])
        .groupby('cell_type', group_keys=False)
        .head(k_tf_gene)
    )

    # ===== TF-Peak =====
    tf_peak = long_df[long_df['group_subtype'] == 'TF-Peak'].copy()
    tf_peak = tf_peak.rename(columns={'feature1': 'TF', 'feature2': 'Peak'})
    tf_peak = tf_peak[['TF', 'Peak', 'cell_type', 'Score']]

    tf_peak_ct_res = (
        tf_peak
        .sort_values(['cell_type', 'Score'], ascending=[True, False])
        .groupby('cell_type', group_keys=False)
        .head(k_tf_peak)
    )

    # ===== Peak-Gene -> Gene-Peak =====
    peak_gene = long_df[long_df['group_subtype'] == 'Peak-Gene'].copy()
    peak_gene = peak_gene.rename(columns={'feature1': 'Peak', 'feature2': 'Gene'})
    gene_peak = peak_gene[['Gene', 'Peak', 'cell_type', 'Score']]

    gene_peak_ct_res = (
        gene_peak
        .sort_values(['cell_type', 'Score'], ascending=[True, False])
        .groupby('cell_type', group_keys=False)
        .head(k_gene_peak)
    )

    return tf_gene_ct_res, tf_peak_ct_res, gene_peak_ct_res


def pred_perturb_gene(df_gene, tf_name,ntop=1000):
    df_tf = df_gene[tf_name]
    tf_norm = (df_tf - df_tf.mean()) / df_tf.std(ddof=1)
    gene_norm = (df_gene - df_gene.mean()) / df_gene.std(ddof=1)
    n_samples = df_tf.shape[0]
    correlation_matrix = np.dot(tf_norm.T, gene_norm) / (n_samples - 1)

    df_corr = pd.DataFrame(
        correlation_matrix,
        index=df_tf.columns,  
        columns=df_gene.columns 
    )
    stacked_series = df_corr.stack() 

    long_df = stacked_series.reset_index()
    long_df.columns = ['TF', 'Gene', 'Score']

    result_df = (
        long_df
        .groupby('TF', group_keys=False)
        .apply(lambda x: x.nlargest(ntop+1, 'Score'))
    )
    result_df = result_df[result_df['TF']!=result_df['Gene']]
    return result_df

def compute_tf_gene_score(rna_data,cand_df,cell_types):
    cell_meta = pd.DataFrame({"barcode":rna_data.index.values,"cell_type":cell_types})
    tf_data = rna_data.T.loc[cand_df['feature1']]
    gene_data = rna_data.T.loc[cand_df['feature2']]
    tf_gene_sample = cand_df[['feature1','feature2']]

    tf_data_aligned = tf_data.reset_index(drop=True)
    gene_data_aligned = gene_data.reset_index(drop=True)

    scores = tf_data_aligned.corrwith(gene_data_aligned, axis=1, method='pearson')

    tf_gene_sample['score'] = scores.values
    tf_gene_sample = tf_gene_sample[tf_gene_sample['feature1']!=tf_gene_sample['feature2']]
    tf_gene_sample.columns = ['TF',"Gene","Score"]
    tf_gene_sample = tf_gene_sample[tf_gene_sample['Score']<2]
    tf_gene_sample = tf_gene_sample.nlargest(10000,"Score")
    tf_gene_ctx = pd.DataFrame()
    for ctx in set(cell_types.unique()):
        sel_cell = cell_meta[cell_meta['cell_type']==ctx]['barcode'].values
        tf_data = rna_data.T.loc[cand_df['feature1'],sel_cell]
        gene_data = rna_data.T.loc[cand_df['feature2'],sel_cell]
        pairs_df = cand_df[['feature1','feature2']]

        tf_data_aligned = tf_data.reset_index(drop=True)
        gene_data_aligned = gene_data.reset_index(drop=True)

        scores = tf_data_aligned.corrwith(gene_data_aligned, axis=1, method='pearson')

        pairs_df['score'] = scores.values

        pairs_df = pairs_df[pairs_df['feature1']!=pairs_df['feature2']]
        pairs_df.columns = ['TF',"Gene","Score"]
        pairs_df= pairs_df[pairs_df['Score']<2]
        pairs_df = pairs_df.nlargest(10000,"Score")
        pairs_df['cell_type'] = ctx
        tf_gene_ctx = pd.concat([tf_gene_ctx,pairs_df],axis=0)
    return tf_gene_sample, tf_gene_ctx

def calc_cellgrn_top_target_score(df_grn, cand_df, gene_list, sample_grn_res,
                                col_name='TF', ntop=500):
    out_df = pd.DataFrame()
    for test_tf in gene_list:
        sel_grn  =cand_df[cand_df[col_name]==test_tf]
        sel_names = sel_grn.apply(lambda row: "_".join(row.astype(str)), axis=1).tolist()
        sel_df = df_grn.copy()[sel_names]
        sel_names2 = sample_grn_res.loc[sel_names,:].nlargest(ntop, "score").index.values 
        sel_df2 = sel_df[sel_names2].sum(axis=1)
        out_df = pd.concat([out_df,sel_df2],axis=1)
    out_df.columns = gene_list
    return out_df 

def calc_cellgrn_top_target_connectivity(df_grn, cand_df, gene_list, 
                                col_name='TF', rank=5000):
    out_df = pd.DataFrame()
    for test_tf in gene_list:
        sel_grn  =cand_df[cand_df[col_name]==test_tf]
        sel_names = sel_grn.apply(lambda row: "_".join(row.astype(str)), axis=1).tolist()
        sel_df = df_grn.copy()[sel_names]
        df_row_ranks = df_grn.rank(axis=1, ascending=False, method="first")
        selected_ranks = df_row_ranks[sel_names]
        sel_df2 = ((sel_df > 0) & (selected_ranks <= rank)).sum(axis=1)
        out_df = pd.concat([out_df,sel_df2],axis=1)
    out_df.columns = gene_list
    return out_df 

    