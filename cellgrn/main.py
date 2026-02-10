import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, MaxAbsScaler
import os
import anndata as ad
from scipy import sparse


def normalzie_rna(gene_expr, temperature=0.1,rna_thres=0.5):
    """
    对 RNA 表达矩阵做归一化，并基于 (x - threshold) / temperature 做 sigmoid 转换。

    输入:
        gene_expr: DataFrame, index=cells, columns=genes
        temperature: float, sigmoid 温度参数 (默认 1.0, 要 > 0)

    输出:
        gene_scaled:  第一阶段归一化结果 (QuantileTransformer 到 [0,1] 的均匀分布)
        gene_scaled2: 使用全局固定阈值 0.5:  sigmoid((gene_scaled - 0.5) / temperature)
        gene_scaled3: 使用每个基因自己的均值作为阈值:
                      对 gene g: sigmoid((gene_scaled[:, g] - mean_g) / temperature)
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    gene_name = gene_expr.columns.values
    cell_name = gene_expr.index.values
    gene_expr2 = gene_expr.to_numpy()

    # -------- Step 1: 对 cell 做 Quantile 归一化 (与原先一致, 输出 ~U[0,1]) --------
    # rna_scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    rna_scaler = MinMaxScaler(feature_range=(0,1))
    gene_scaled_array = rna_scaler.fit_transform(gene_expr2.T).T   # shape: (n_cell, n_gene)

    gene_scaled = pd.DataFrame(gene_scaled_array, index=cell_name, columns=gene_name)
    # return gene_scaled
    # -------- Step 2: 基于 (x - tf_th) / temperature 做 sigmoid 转换 --------

    # 2.1 tf_th = 0.5 (固定阈值)
    # tf_th_fixed = 0.5
    sig_input_fixed = (gene_scaled_array - rna_thres) / temperature
    gene_scaled2_array = 1.0 / (1.0 + np.exp(-sig_input_fixed))
    gene_scaled2 = pd.DataFrame(gene_scaled2_array, index=cell_name, columns=gene_name)
    return gene_scaled, gene_scaled2


def normalzie_rna_sparse(gene_expr_sparse, temperature=0.1, rna_thres=0.5):
    """
    对 RNA 表达矩阵做归一化 (支持稀疏矩阵输入)。
    输入:
        gene_expr_sparse: scipy.sparse matrix (cells x genes)
    输出:
        gene_scaled:  scipy.sparse matrix (MaxAbs scaled, 保持稀疏)
        gene_scaled2: numpy array (Sigmoid transformed, 必然是 Dense)
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Step 1: MaxAbsScaler
    # 对于 Count 数据 (min=0)，MaxAbsScaler 等价于 MinMaxScaler(0,1)
    # 它的好处是支持稀疏输入，且输出依然是稀疏矩阵
    scaler = MaxAbsScaler()
    gene_scaled_sparse = scaler.fit_transform(gene_expr_sparse)

    # Step 2: Sigmoid
    # 注意：Sigmoid 变换会将 0 变为非 0 值 (1/(1+e^(-(-0.5/0.1))) != 0)，
    # 因此结果必然是 Dense 的。
    # 我们在这里将稀疏矩阵转为 Dense，并强制使用 float32 节省内存。
    
    # 获取 Dense 数组用于计算 Sigmoid
    if sparse.issparse(gene_scaled_sparse):
        gene_scaled_dense = gene_scaled_sparse.toarray().astype(np.float32)
    else:
        gene_scaled_dense = gene_scaled_sparse.astype(np.float32)

    # 计算 Sigmoid
    sig_input_fixed = (gene_scaled_dense - rna_thres) / temperature
    # 截断避免溢出
    sig_input_fixed = np.clip(sig_input_fixed, -50, 50) 
    gene_scaled2_array = 1.0 / (1.0 + np.exp(-sig_input_fixed))

    # 返回值：
    # gene_scaled_sparse: 保持稀疏，用于某些可能只需要线性缩放的地方（虽然本流程主要用 sigmoid）
    # gene_scaled2_array: Dense 矩阵
    return gene_scaled_sparse, gene_scaled2_array


def parse_edges(cand_df, tf_list, gene_list, peak_list):
    """
    解析边，返回三种边的索引和名称：
    - edges_idx: dict with keys 'tf_gene', 'tf_peak', 'peak_gene'
        每个 value 是 (list_of_tf_idx, list_of_gene_or_peak_idx)
    - edges_name: dict with keys 'tf_gene', 'tf_peak', 'peak_gene'
        每个 value 是 [ "TF-Gene:tf:gene", ...] 这样的名字列表
    """
    tf_set = set(tf_list)
    gene_set = set(gene_list)
    peak_set = set(peak_list)

    # 用于从名字找到列索引
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


# 核心函数：计算单个细胞的边得分

def compute_cell_grn_scores(cell_id, tf_df, rna_df, atac_df, edges):
    """
    计算单个细胞的 GRN 边得分
    """
    
    tf_values = tf_df.loc[cell_id]
    gene_values = rna_df.loc[cell_id]
    peak_values = atac_df.loc[cell_id]
    
    scores = {}
    
    # 方法1: 直接相乘
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

# 计算所有细胞的 GRN 得分


def compute_all_cells_grn(tf_df, rna_df, atac_df, 
                          edges_idx, edges_name,
                          tf_list, gene_list, peak_list):
    """
    向量化计算所有细胞的 GRN 边得分
    修改点：TF-Gene 的得分现在基于共享 Peak 的路径总和计算
    公式：Score = TF * Gene * Sum(Peak^2)
    """
    cell_ids = tf_df.index.tolist()

    # 转成矩阵
    tf_mat = tf_df[tf_list].to_numpy()      # (n_cell, n_tf)
    rna_mat = rna_df[gene_list].to_numpy()  # (n_cell, n_gene)
    
    # 必须有 ATAC 数据才能计算基于 Peak 的路径
    if atac_df is None:
        raise ValueError("Must input atac_df to calculate peak-based scores.")
    
    atac_mat = atac_df[peak_list].to_numpy() # (n_cell, n_peak)

    all_edge_scores = []
    all_edge_names = []

    # ==========================================
    # 1. TF-Peak (保持原样)
    # ==========================================
    tf_idx, peak_idx = edges_idx['tf_peak']
    if len(tf_idx) > 0:
        tf_sub = tf_mat[:, tf_idx]
        peak_sub = atac_mat[:, peak_idx]
        tf_peak_scores = tf_sub * peak_sub
        all_edge_scores.append(tf_peak_scores)
        all_edge_names.extend(edges_name['tf_peak'])

    # ==========================================
    # 2. Peak-Gene (保持原样)
    # ==========================================
    peak_idx, gene_idx = edges_idx['peak_gene']
    if len(peak_idx) > 0:
        peak_sub = atac_mat[:, peak_idx]
        gene_sub = rna_mat[:, gene_idx]
        peak_gene_scores = peak_sub * gene_sub
        all_edge_scores.append(peak_gene_scores)
        all_edge_names.extend(edges_name['peak_gene'])

    # ==========================================
    # 3. TF-Gene (核心修改部分)
    # ==========================================
    # 目标：对于每一个 TF-Gene 边，找到中间所有的 Peak，计算 sum(TF * Peak^2 * Gene)
    # 优化公式 = (TF * Gene) * sum(Peak^2)
    
    target_tf_idx, target_gene_idx = edges_idx['tf_gene']
    
    if len(target_tf_idx) > 0:
        # --- A. 构建拓扑映射关系 ---
        # 1. 构建 TF-Peak 关系表
        tp_tf, tp_peak = edges_idx['tf_peak']
        df_tp = pd.DataFrame({'tf': tp_tf, 'peak': tp_peak})
        
        # 2. 构建 Peak-Gene 关系表
        pg_peak, pg_gene = edges_idx['peak_gene']
        df_pg = pd.DataFrame({'peak': pg_peak, 'gene': pg_gene})
        
        # 3. 构建目标 TF-Gene 索引表 (我们需要保持这个顺序)
        df_target = pd.DataFrame({'tf': target_tf_idx, 'gene': target_gene_idx})
        df_target['edge_id'] = range(len(df_target)) # 记录原始顺序 ID
        
        # --- B. 寻找共有 Peak (Merge操作) ---
        # 逻辑：(TF, Peak) join (Peak, Gene) -> (TF, Peak, Gene)
        df_paths = pd.merge(df_tp, df_pg, on='peak')
        
        # 逻辑：将找到的路径与我们要计算的目标 TF-Gene 边进行匹配
        # 只有在 edges_idx['tf_gene'] 中存在的组合才会被计算
        df_valid_paths = pd.merge(df_paths, df_target, on=['tf', 'gene'])
        
        # df_valid_paths 现在包含列: [tf, peak, gene, edge_id]
        # 每一行代表一条通路：TF -> Peak -> Gene，属于第 edge_id 个 TF-Gene 边
        
        if len(df_valid_paths) > 0:
            # --- C. 向量化计算 Sum(Peak^2) ---
            # 我们需要构建一个稀疏矩阵 M，形状为 (n_peaks, n_tf_gene_edges)
            # M[p, e] = 1 表示 Peak p 是边 e 的中间桥梁
            
            # 提取映射索引
            path_peak_indices = df_valid_paths['peak'].values
            path_edge_indices = df_valid_paths['edge_id'].values
            
            # 构造稀疏矩阵 (行是Peak，列是目标Edge)
            # n_peak 是总 peak 数量 (peak_list长度)，n_edges 是 tf_gene 边的数量
            n_total_peaks = atac_mat.shape[1]
            n_target_edges = len(df_target)
            
            # 创建稀疏权重矩阵 (Values都是1，表示存在连接)
            # coo_matrix((data, (row, col)), shape=...)
            peak_agg_matrix = sparse.coo_matrix(
                (np.ones(len(df_valid_paths)), (path_peak_indices, path_edge_indices)),
                shape=(n_total_peaks, n_target_edges)
            ).tocsr() # 转为 CSR 格式以支持矩阵乘法
            
            # 计算 Peak^2 的聚合
            # 矩阵乘法: (Cells x Peaks) @ (Peaks x Edges) -> (Cells x Edges)
            # 这一步直接算出了每个细胞、每个 TF-Gene 对中间所有 Peak 的平方和
            sum_peak_sq = (atac_mat ** 2) @ peak_agg_matrix
            
            # --- D. 结合 TF 和 Gene ---
            # 提取对应的 TF 和 Gene 表达量
            # (Cells x Edges)
            tf_vals = tf_mat[:, target_tf_idx]
            gene_vals = rna_mat[:, target_gene_idx]
            
            # 最终计算: TF * Gene * Sum_Peak_Sq
            tf_gene_scores = tf_vals * gene_vals * sum_peak_sq
            
        else:
            # 如果没有找到任何中间 Peak 通路，得分为 0
            tf_gene_scores = np.zeros((len(cell_ids), len(target_tf_idx)))

        all_edge_scores.append(tf_gene_scores)
        all_edge_names.extend(edges_name['tf_gene'])

    if len(all_edge_scores) == 0:
        return pd.DataFrame(index=cell_ids)

    # 沿列拼接
    scores_mat = np.concatenate(all_edge_scores, axis=1)
    scores_df = pd.DataFrame(scores_mat, index=cell_ids, columns=all_edge_names)
    
    return scores_df

class SparseGRNCalculator:
    """
    用于分块计算 GRN 得分的类
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
        
        # 累加
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

# 辅助格式化函数
def parse_edge_index(index_series):
    parts = index_series.to_series().str.split("_", n=2, expand=True)
    parts.columns = ['group_subtype', 'feature1', 'feature2']
    return parts

def summarize_grn(grn_scores_df, cell_types_series):
    """
    汇总 GRN 得分
    返回:
        - sample_grn: 全样本平均得分
        - celltype_grn: 按细胞类型的平均得分
    """
    # 全样本平均
    sample_grn = grn_scores_df.mean(axis=0) #.sort_values(ascending=False)
    sample_grn = pd.DataFrame({'score': sample_grn})
    
    # 按细胞类型平均
    grn_with_ct = grn_scores_df.copy()
    grn_with_ct['cell_type'] = cell_types_series.values
    
    celltype_grn = grn_with_ct.groupby('cell_type').mean().T
    
    return sample_grn, celltype_grn


def parse_edge_index(index_series):
    """
    输入: index_series 是一个类似 Index 或 Series，元素形如 'TF-Gene:TF:Gene'
    输出: DataFrame, 三列: ['group_subtype', 'feature1', 'feature2']
    """
    parts = index_series.to_series().str.split("_", n=2, expand=True)
    parts.columns = ['group_subtype', 'feature1', 'feature2']
    return parts

def format_sample_grn(sample_grn):
    """
    输入:
        sample_grn: DataFrame, index: 'TF-Gene:TF:Gene' 字符串, 列: ['score']
    输出:
        tf_gene_res:  columns = ['TF', 'Gene', 'Score']
        tf_peak_res:  columns = ['TF', 'Peak', 'Score']
        gene_peak_res:columns = ['Gene', 'Peak', 'Score']
    """
    # 解析 index
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

    # --- Peak-Gene -> Gene-Peak (注意顺序反转) ---
    peak_gene = edge_df[edge_df['group_subtype'] == 'Peak-Gene'].copy()
    # 交换 feature1 / feature2 的语义
    # 原来: feature1=Peak, feature2=Gene
    # 需求: Gene, Peak, Score
    peak_gene_res = peak_gene.rename(columns={
        'feature1': 'Peak',
        'feature2': 'Gene',
        'score': 'Score'
    })
    # 交换列顺序时注意 Gene/Peak 的位置
    gene_peak_res = peak_gene_res[['Gene', 'Peak', 'Score']].nlargest(20000, "Score")

    return tf_gene_res, tf_peak_res, gene_peak_res

def format_celltype_grn(celltype_grn,
                             k_tf_gene=10000,
                             k_tf_peak=20000,
                             k_gene_peak=20000):
    """
    输入:
        celltype_grn: DataFrame
            index: 'group_subtype:feature1:feature2'
                    例如: 'TF-Gene:TFX:GENE1', 'TF-Peak:TFY:chr1:100-200', ...
            列: 各个 cell_type 名称

    参数:
        k_tf_gene:   每个 cell_type 输出的 TF-Gene 边的数量上限
        k_tf_peak:   每个 cell_type 输出的 TF-Peak 边的数量上限
        k_gene_peak: 每个 cell_type 输出的 Gene-Peak 边的数量上限

    输出:
        tf_gene_ct_res:   columns = ['TF', 'Gene', 'cell_type', 'Score']
        tf_peak_ct_res:   columns = ['TF', 'Peak', 'cell_type', 'Score']
        gene_peak_ct_res: columns = ['Gene', 'Peak', 'cell_type', 'Score']

        每个输出 DataFrame 是“每个 cell_type 内按 Score 取 top-k 后的拼接结果”。
    """
    ct_df = celltype_grn.copy()
    ct_df['edge'] = ct_df.index

    # 宽表 -> 长表: 一行对应 (edge, cell_type, Score)
    long_df = ct_df.melt(
        id_vars='edge',
        var_name='cell_type',
        value_name='Score'
    )

    # 解析 edge 字符串
    parts = long_df['edge'].str.split("_", n=2, expand=True)
    parts.columns = ['group_subtype', 'feature1', 'feature2']
    long_df = pd.concat([parts, long_df[['cell_type', 'Score']]], axis=1)

    # ===== TF-Gene =====
    tf_gene = long_df[long_df['group_subtype'] == 'TF-Gene'].copy()
    tf_gene = tf_gene.rename(columns={'feature1': 'TF', 'feature2': 'Gene'})
    tf_gene = tf_gene[['TF', 'Gene', 'cell_type', 'Score']]

    # 按 cell_type 分组，在每个组内按 Score 取 top k_tf_gene
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

    # 4. 转回 DataFrame 以便查看
    df_corr = pd.DataFrame(
        correlation_matrix,
        index=df_tf.columns,  # 行是 TF
        columns=df_gene.columns # 列是 Gene
    )
    stacked_series = df_corr.stack() 
    # 此时 stacked_series 的索引是 MultiIndex (TF, Gene)，值是 Score

    # 3. 重置索引以便操作
    long_df = stacked_series.reset_index()
    long_df.columns = ['TF', 'Gene', 'Score']

    # 4. 分组并取 Top 1000
    # 逻辑：按 TF 分组 -> 按 Score 降序排列 -> 取前 1000
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

    # 第四步：计算相关性 (axis=1 代表按行计算)
    # 这一步是并行的，极快
    scores = tf_data_aligned.corrwith(gene_data_aligned, axis=1, method='pearson')

    # 第五步：填入结果
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

        # 第四步：计算相关性 (axis=1 代表按行计算)
        # 这一步是并行的，极快
        scores = tf_data_aligned.corrwith(gene_data_aligned, axis=1, method='pearson')

        # 第五步：填入结果
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

    