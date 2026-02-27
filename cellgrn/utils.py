

import pandas as pd
import numpy as np
import heapq
import os
from sklearn.metrics import precision_recall_curve, auc
import pyranges as pr
import re
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


def get_top_regu(regu_file, rowitem, colitem, ntop=10000, chunksize=2000):

    heap = []

    for chunk in pd.read_csv(regu_file, sep="\t", index_col=0, chunksize=chunksize):
        row_names = chunk.index.to_numpy()
        col_names = chunk.columns.to_numpy()
        matrix = chunk.to_numpy()

        n_col = len(col_names)

        for i, row in enumerate(matrix):

            top_idx = np.argpartition(row, -min(ntop, n_col))[-min(ntop, n_col):]
            for j in top_idx:
                val = row[j]
                if len(heap) < ntop:
                    heapq.heappush(heap, (val, row_names[i], col_names[j]))
                else:
                    if val > heap[0][0]:
                        heapq.heapreplace(heap, (val, row_names[i], col_names[j]))

    top_res = heapq.nlargest(ntop, heap)
    df_top = pd.DataFrame(top_res, columns=["Score", rowitem, colitem])
    df_top = df_top[[colitem, rowitem, "Score"]]  # 与原函数保持一致

    return df_top

def get_top_regu2(regu_file,ntop=20000):
    df = pd.read_csv(regu_file, sep="\t",header=None)
    df.columns = ['Peak', 'Gene', "Score"]
    df = df[['Gene', 'Peak', "Score"]]
    top_res = df.nlargest(ntop, "Score")
    return top_res

def load_scenic2(res_path, soft_path, *args): 
    scenic2_grn1 = pd.read_csv(f"{res_path}/{soft_path}/Snakemake/eRegulon_direct.tsv", header=0, sep='\t')
    scenic2_grn2 = pd.read_csv(f"{res_path}/{soft_path}/Snakemake/eRegulons_extended.tsv", header=0, sep='\t')
    scenic2_grn = pd.concat([scenic2_grn1,scenic2_grn2],axis=0)
    scenic2_grn_out = scenic2_grn[['TF',"Gene",'importance_TF2G']]
    scenic2_grn_out.columns = ["TF", "Gene", "Score"]
    scenic2_grn_out.drop_duplicates(inplace=True)
    scenic2_gene_peak_out  = scenic2_grn[['Gene',"Region",'importance_R2G']]
    scenic2_gene_peak_out.columns = ["Gene","Peak","Score"]
    scenic2_gene_peak_out.drop_duplicates(inplace=True)
    scenic2_tf_peak_out  = scenic2_grn[['TF',"Region",'regulation']] # regulation只记录，不使用
    scenic2_tf_peak_out.columns = ['TF',"Peak","Score"]
    scenic2_tf_peak_out.drop_duplicates(inplace=True)
    return scenic2_gene_peak_out, scenic2_grn_out, scenic2_tf_peak_out


def load_figr(res_path, soft_path, *args): 
    figr_gene_peak = pd.read_csv(f"{res_path}/{soft_path}/figr_gene_peak.txt", header=0, sep='\t')
    figr_grn = pd.read_csv(f"{res_path}/{soft_path}/figr_grn.txt", header=0, sep='\t')
    figr_grn = figr_grn[figr_grn["Score"]>1]
    figr_gene_peak_out = figr_gene_peak[["Gene","PeakRanges","rObs"]]
    figr_gene_peak_out.columns = ["Gene","Peak","Score"]
    figr_grn_out = figr_grn[["Motif","DORC","Score"]]
    figr_grn_out.columns = ["TF","Gene","Score"]
    return figr_gene_peak_out, figr_grn_out, None

def load_linger_all(res_path, soft_path, *args): 
    gene_peak_res = get_top_regu2(f"{res_path}/{soft_path}/cell_population_cis_regulatory.txt")
    tf_peak_res = get_top_regu(f"{res_path}/{soft_path}/cell_population_TF_RE_binding.txt","Peak","TF",ntop=20000)

    tf_gene_res = get_top_regu(f"{res_path}/{soft_path}/cell_population_trans_regulatory.txt","Gene","TF")
    tf_gene_res = tf_gene_res[tf_gene_res['Gene']!=tf_gene_res['TF']]
    return gene_peak_res, tf_gene_res, tf_peak_res

def load_linger_ctx(res_path, soft_path, *args): 
    linger_gene_peak_out = pd.DataFrame()
    if not os.path.exists(f"{res_path}/{soft_path}/cell_population_cis_regulatory.txt"):
        return None, None, None
    

    for myf in os.listdir(f"{res_path}/{soft_path}/"):
        if myf.startswith("cell_type_specific_cis_regulatory_"):
            cts = myf.lstrip("cell_type_specific_cis_regulatory_")[:-4]

            gene_peak_res = get_top_regu2(f"{res_path}/{soft_path}/{myf}")
            gene_peak_res[['cell_type']] = cts
            linger_gene_peak_out = pd.concat([linger_gene_peak_out, gene_peak_res],axis=0)

    linger_tf_peak_out = pd.DataFrame()
    for myf in os.listdir(f"{res_path}/{soft_path}/"):
        if myf.startswith("cell_type_specific_TF_RE_binding_"):
            cts = myf.lstrip("cell_type_specific_TF_RE_binding_")[:-4]

            tf_peak_res = get_top_regu(f"{res_path}/{soft_path}/{myf}","Peak","TF",ntop=20000)
            tf_peak_res[['cell_type']] = cts
            linger_tf_peak_out = pd.concat([linger_tf_peak_out, tf_peak_res],axis=0)

    linger_tf_gene_out = pd.DataFrame()
    for myf in os.listdir(f"{res_path}/{soft_path}/"):
        if myf.startswith("cell_type_specific_trans_regulatory_"):
            cts = myf.lstrip("cell_type_specific_trans_regulatory_")[:-4]

            tf_gene_res = get_top_regu(f"{res_path}/{soft_path}/{myf}","Gene","TF")
            tf_gene_res[['cell_type']] = cts
            tf_gene_res = tf_gene_res[tf_gene_res['Gene']!=tf_gene_res['TF']]
            linger_tf_gene_out = pd.concat([linger_tf_gene_out, tf_gene_res],axis=0)
    return linger_gene_peak_out, linger_tf_gene_out, linger_tf_peak_out

def load_thres_grn(res_path,scale="sample",suffix="raw",peak_rev=False):
    gene_peak_res_out = pd.read_csv(f"{res_path}/gene_peak_{scale}_{suffix}.csv",header=0)
    if peak_rev:
        gene_peak_res_out['Peak'] = gene_peak_res_out['Peak'].str.replace(r'^([^-\s]+)-', r'\1:', regex=True)
    tf_gene_res_out = pd.read_csv(f"{res_path}/tf_gene_{scale}_{suffix}.csv",header=0)
    tf_gene_res_out = tf_gene_res_out[tf_gene_res_out['TF'] !=tf_gene_res_out['Gene']]
    tf_peak_res_out = pd.read_csv(f"{res_path}/tf_peak_{scale}_{suffix}.csv",header=0)
    if peak_rev:
        tf_peak_res_out['Peak'] = tf_peak_res_out['Peak'].str.replace(r'^([^-\s]+)-', r'\1:', regex=True)
    return gene_peak_res_out, tf_gene_res_out, tf_peak_res_out


def generate_candidate_groups(original_dfs) -> pd.DataFrame:

    groups = []
    for original_df in original_dfs:

        for _, row in original_df.iterrows():
            
            tf = row["TF"] if "TF" in original_df.columns  else None
            peak = row["Peak"] if "Peak" in original_df.columns else None
            gene = row["Gene"] if "Gene" in original_df.columns else None
            
            
            if not peak:

                groups.append({
                    "group_subtype": "TF-Gene",
                    "feature1": tf,
                    "feature2": gene
                })
            if not gene:

                groups.append({
                    "group_subtype": "TF-Peak",
                    "feature1": tf,
                    "feature2": peak
                })
            if not tf:

                groups.append({
                    "group_subtype": "Peak-Gene",
                    "feature1": peak,
                    "feature2": gene
                })
    
    candidate_groups_df = pd.DataFrame(groups).drop_duplicates()
    
    return candidate_groups_df



def enhancer_eval(enhancer_gold, gene_peak_res, soft="SparseGRN"):

    p_count = enhancer_gold.iloc[:, enhancer_gold.shape[1]-2].sum()
    # print(p_count)
    n_count = enhancer_gold.shape[0] - p_count

    gene_peak_res2 = pd.merge(gene_peak_res, enhancer_gold, on='Peak')
    gene_peak_res2 = gene_peak_res2[gene_peak_res2['Score'] != np.inf]
   
    gene_peak_res2 = gene_peak_res2.groupby(['Peak'], as_index=False)[['Score', 3]].max()
    gene_peak_res2[3] = np.array(gene_peak_res2[3], dtype=np.int32)
    gene_peak_res2.sort_values(by="Score", inplace=True, ascending=False)
    # print(soft)
    label_loc = gene_peak_res2.shape[1]-1
    
    if gene_peak_res2.shape[0] > 1:
        label = gene_peak_res2.iloc[:, label_loc]

        label.loc[label > 1] = 1 

        tp_count = sum(label)
        fp_count = len(label) - sum(label)
        fn_count = p_count - tp_count
        tn_count = n_count - fp_count # T-TP
        neg_score = min(gene_peak_res2['Score']) - 1 

        if len(label) >= p_count:
            epr = gene_peak_res2.iloc[:int(p_count), label_loc].sum() / p_count
        else:
            epr = label.sum() / len(label)

        y_true = list(label) + [1 for i in range(int(fn_count))] + [0 for i in range(int(tn_count))]
        y_scores = list(gene_peak_res2['Score']) + [neg_score for i in range(int(fn_count))] + [neg_score for i in range(int(tn_count))]

        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        pr_auc = auc(recall, precision)
        
        numerator = 2 * precision * recall
        denominator = precision + recall
        f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        f1 = np.max(f1_scores)
        # ------------------------

        pr_table = pd.DataFrame({"prec": precision,
                                 "recall": recall,
                                 "prauc": pr_auc,
                                 "method": soft})
    else:
        pr_table = pd.DataFrame({"prec": [0],
                                 "recall": [0],
                                 "prauc": [0],
                                 "method": [soft]})
        pr_auc = 0
        epr = 0
        f1 = 0 
        
    return pr_table, pr_auc, epr, f1



def eval_gene_peak(gene_peak_gold, gene_peak_res, gene_peak_dist, soft="SparseGRN"):
    peaks_region = {
        "Chromosome": [],
        "Start": [],
        "End": []
    }
    
    peaks_spec=[re.split(pattern="[:|-]",string=i) for i in gene_peak_res['Peak'].values]

    peaks_region["Chromosome"]= [i[0] for i in peaks_spec]
    peaks_region["Start"] = [i[1] for i in peaks_spec]
    peaks_region["End"] = [i[2] for i in peaks_spec]
    peaks_region["Name"] = list(gene_peak_res['Gene'])
    
    peak2 = gene_peak_gold.join(pr.from_dict(peaks_region))
    peak2=peak2.df

    peak2['Name_b'] = [i.split(".")[0] for i in  peak2['Name_b']]

    peak_gene_sel = peak2[peak2['Name']==peak2['Name_b']].iloc[:,[0,4,5,6]]
    peak_gene_sel['peak_gene'] = peak_gene_sel.apply(lambda row: f"{row[0]}:{row[1]}-{row[2]}_{row[3]}", axis=1)
    
    peak_gene_sel['Gene'] = peak_gene_sel['Name_b']
    peak_gene_sel['Peak'] = peak_gene_sel.apply(lambda row: f"{row[0]}:{row[1]}-{row[2]}", axis=1)

    peak_gene_sel = pd.merge(peak_gene_sel,gene_peak_dist,on=["Gene","Peak"],how="left")
    
    gene_peak_res['peak_gene'] = gene_peak_res.apply(lambda row: f"{row[1]}_{row[0]}", axis=1)
    gene_peak_res['label'] = 0
    gene_peak_res.dropna(inplace=True)

    gene_peak_res.loc[gene_peak_res['peak_gene'].isin(peak_gene_sel['peak_gene'].values),'label'] = 1
    gene_peak_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    gene_peak_res = gene_peak_res.groupby(['Gene', 'Peak'], as_index=False)[['Score','label','Dist']].max()   
    gene_peak_res = gene_peak_res.astype({"label":"int32"})
    
    gene_peak_res.sort_values(by="Score",inplace=True, ascending=False)
    gene_peak_res.drop_duplicates(inplace=True)
    
    p_count = gene_peak_gold.df.shape[0]
    n_count = p_count*50
    tp_count = sum(gene_peak_res['label'])
    
    fp_count = gene_peak_res.shape[0] -tp_count
    fn_count = p_count - tp_count
    tn_count = n_count - fp_count 
    neg_score = min(gene_peak_res['Score'])- 1 
    
    if gene_peak_res.shape[0]>=p_count:
        epr = gene_peak_res['label'].iloc[0:p_count].sum()/p_count
    else:
        epr = gene_peak_res['label'].sum()/gene_peak_res.shape[0]

    precision, recall, _ = precision_recall_curve(list(gene_peak_res['label'])+[1 for i in range(fn_count)]+[0 for i in range(tn_count)],
                                            list(gene_peak_res['Score'])+[neg_score for i in range(fn_count)]+[neg_score for i in range(tn_count)])
    
    pr_auc = auc(recall, precision)

    # --- Add Global F1 Calculation ---
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0
    # ---------------------------------

    pr_table = pd.DataFrame({"prec":precision,
                            "recall":recall,
                            "prauc":pr_auc,
                            "method":soft})
    
    bins = [0,5000,10000,20000,50000,100000,200000,1000000]
    range_sel = ['0-5k','5-10k','10-20k','20-50k','50-100k','100-200k','>200k']

    gene_peak_res['range'] = pd.cut(gene_peak_res['Dist'],bins=bins,labels=range_sel,right=True)
    peak_gene_sel['range'] = pd.cut(peak_gene_sel['Dist'],bins=bins,labels=range_sel,right=True)
    range_result = pd.DataFrame()
    
    for range_sel_ in range_sel:
        gene_peak_res_ = gene_peak_res[gene_peak_res['range']==range_sel_]
        peak_gene_sel_ = peak_gene_sel[peak_gene_sel['range']==range_sel_]
        p_count = peak_gene_sel_.shape[0]
        n_count = p_count*50
        tp_count = sum(gene_peak_res_['label'])

        if tp_count==0:
            # Add f1: 0
            out_res = pd.DataFrame({"prauc":[0], "epr":[0], "f1":[0], "method":[soft],"range":[range_sel_]})
        else:
            fp_count = gene_peak_res_.shape[0] -tp_count
            fn_count = p_count - tp_count
            tn_count = n_count - fp_count 
            if gene_peak_res_.shape[0]>=p_count:
                epr_range = gene_peak_res_['label'].iloc[0:p_count].sum()/p_count
            else:
                epr_range = gene_peak_res_['label'].sum()/gene_peak_res_.shape[0]

            precision, recall, _ = precision_recall_curve(list(gene_peak_res_['label'])+[1 for i in range(fn_count)]+[0 for i in range(tn_count)],
                                                    list(gene_peak_res_['Score'])+[neg_score for i in range(fn_count)]+[neg_score for i in range(tn_count)])

            pr_auc_range = auc(recall, precision)
            
            # --- Add Range F1 Calculation ---
            num_r = 2 * precision * recall
            den_r = precision + recall
            f1_scores_r = np.divide(num_r, den_r, out=np.zeros_like(num_r), where=den_r != 0)
            f1_range = np.max(f1_scores_r) if len(f1_scores_r) > 0 else 0
            # --------------------------------

            out_res = pd.DataFrame({"prauc":[pr_auc_range], "epr":[epr_range], "f1":[f1_range], "method":[soft],"range":[range_sel_]})
        range_result = pd.concat([range_result,out_res], axis=0)
        
    return pr_table, pr_auc, epr, f1, range_result


# do not restrict gene peak distance
def eval_gene_peak2(gene_peak_gold, gene_peak_res, soft="SparseGRN"): 

    peaks_region = {
        "Chromosome": [],
        "Start": [],
        "End": []
    }
    
    peaks_spec=[re.split(pattern="[:|-]",string=i) for i in gene_peak_res['Peak'].values]

    peaks_region["Chromosome"]= [i[0] for i in peaks_spec]
    peaks_region["Start"] = [i[1] for i in peaks_spec]
    peaks_region["End"] = [i[2] for i in peaks_spec]
    peaks_region["Name"] = list(gene_peak_res['Gene'])
    
    peak2 = gene_peak_gold.join(pr.from_dict(peaks_region))
    peak2=peak2.df

    peak2['Name_b'] = [i.split(".")[0] for i in  peak2['Name_b']]


    peak_gene_sel = peak2[peak2['Name']==peak2['Name_b']].iloc[:,[0,4,5,6]]
    peak_gene_sel['peak_gene'] = peak_gene_sel.apply(lambda row: f"{row[0]}:{row[1]}-{row[2]}_{row[3]}", axis=1)
    
    peak_gene_sel['Gene'] = peak_gene_sel['Name_b']
    peak_gene_sel['Peak'] = peak_gene_sel.apply(lambda row: f"{row[0]}:{row[1]}-{row[2]}", axis=1)

    gene_peak_res['peak_gene'] = gene_peak_res.apply(lambda row: f"{row[1]}_{row[0]}", axis=1)
    gene_peak_res['label'] = 0
    gene_peak_res.dropna(inplace=True)

    gene_peak_res.loc[gene_peak_res['peak_gene'].isin(peak_gene_sel['peak_gene'].values),'label'] = 1
    gene_peak_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    gene_peak_res = gene_peak_res.groupby(['Gene', 'Peak'], as_index=False)[['Score','label']].max()   
    gene_peak_res = gene_peak_res.astype({"label":"int32"})
    
    gene_peak_res.sort_values(by="Score",inplace=True, ascending=False)
    gene_peak_res.drop_duplicates(inplace=True)
    
    p_count = gene_peak_gold.df.shape[0]
    n_count = p_count*50
    tp_count = sum(gene_peak_res['label'])
    print(tp_count)
    fp_count = gene_peak_res.shape[0] -tp_count
    fn_count = p_count - tp_count
    tn_count = n_count - fp_count 
    neg_score = min(gene_peak_res['Score'])- 1 
    
    if gene_peak_res.shape[0]>=p_count:
        epr = gene_peak_res['label'].iloc[0:p_count].sum()/p_count
    else:
        epr = gene_peak_res['label'].sum()/gene_peak_res.shape[0]

    precision, recall, _ = precision_recall_curve(list(gene_peak_res['label'])+[1 for i in range(fn_count)]+[0 for i in range(tn_count)],
                                            list(gene_peak_res['Score'])+[neg_score for i in range(fn_count)]+[neg_score for i in range(tn_count)])
    
    pr_auc = auc(recall, precision)

    # --- Add F1 Calculation ---
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0
    # --------------------------

    pr_table = pd.DataFrame({"prec":precision,
                            "recall":recall,
                            "prauc":pr_auc,
                            "method":soft})
    
    return pr_table, pr_auc, epr, f1


def eval_tf_recovery(grn_res, gold_data, label, log = False):
    grn_res = grn_res.groupby(['TF'], as_index=False)['Score'].max()    
    grn_res = grn_res.nlargest(min(200,grn_res.shape[0]), "Score")
    best_f1_list = []
    best_prec_list = []
    best_recall_list = []
    best_n_list = []

    cell_type = list(gold_data.keys())
    for ctx in cell_type:
        sel_TF = set(gold_data[ctx])

        grn_sorted = grn_res.sort_values('Score', ascending=False)
        tf_list = list(grn_sorted['TF'])
        
        max_f1 = 0
        best_prec = 0
        best_recall = 0
        best_n = 0

        max_top_n = min(200, len(tf_list))
        for n in range(1, max_top_n+1):
            pred_TF_set = set(tf_list[:n])
            overlap = len(pred_TF_set & sel_TF)
            prec = overlap / n
            recall = overlap / len(sel_TF) if len(sel_TF)>0 else 0
            if prec + recall > 0:
                f1 = 2 * prec * recall / (prec + recall)
            else:
                f1 = 0
            if f1 > max_f1:
                max_f1 = f1
                best_prec = prec
                best_recall = recall
                best_n = n
        best_f1_list.append(max_f1)
        best_prec_list.append(best_prec)
        best_recall_list.append(best_recall)
        best_n_list.append(best_n)
        if log:
            print(f"{label} in {ctx}: best F1 {max_f1:.3f} at Top {best_n}, Precision {best_prec:.3f}, Recall {best_recall:.3f}")

    tf_recover = pd.DataFrame({
        "method": [label for _ in cell_type],
        "cell_type": cell_type,
        "Best_F1": best_f1_list,
        "Best_Precision": best_prec_list,
        "Best_Recall": best_recall_list,
        "Best_N": best_n_list
    })

    tf_recovery_summary = tf_recover.groupby("method").agg(
        Best_F1_avg=("Best_F1", "mean"),
        Best_Precision_avg=("Best_Precision", "mean"),
        Best_Recall_avg=("Best_Recall", "mean"),
        Best_N_avg=("Best_N", "mean")
    ).round(4)
    return tf_recover, tf_recovery_summary


def eval_tf_gene(grn_res, gold_data, all_comb, label):

    grn_res = grn_res.groupby(['TF','Gene'], as_index=False)['Score'].max()  
    soft_pred = grn_res.nlargest(min(20000,grn_res.shape[0]), "Score")
    soft_pred['pred'] = 0
    soft_pred['pair'] = soft_pred.apply(lambda row: f"{row[0]}_{row[1]}", axis=1)
    soft_pred.loc[soft_pred['pair'].isin(gold_data),'pred'] = 1
    p_count = len(set(gold_data))
    tp_count = sum(soft_pred['pred'])
    fp_count = soft_pred.shape[0] - sum(soft_pred['pred'])
    fn_count = p_count - tp_count # len(set(gold_data) - set(soft_pred['pair'].values))
    tn_count = all_comb - len(set(gold_data))- fp_count 
    neg_score = min(soft_pred['Score'])- 1 

    precision, recall, _ = precision_recall_curve(list(soft_pred['pred'])+[1 for i in range(fn_count)]+[0 for i in range(tn_count)],
                                                list(soft_pred['Score'])+[neg_score for i in range(fn_count)]+[neg_score for i in range(tn_count)])
    pr_auc = auc(recall, precision)
    
    # --- Add F1 Calculation ---
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0
    # --------------------------

    if soft_pred.shape[0] >= p_count:
        epr = soft_pred['pred'].iloc[0:p_count].sum()/p_count
    else:
        epr = soft_pred['pred'].sum()/p_count
        
    pr_curve = pd.DataFrame({"prec":precision,
                            "recall":recall,
                            "prauc":pr_auc,
                            "method":label})
    return pr_auc, epr, f1, pr_curve


def eval_tf_recovery_ctx(grn_res_ctx, gold_data, label, log = False):
    
    best_f1_list = []
    best_prec_list = []
    best_recall_list = []
    best_n_list = []

    cell_type = list(gold_data.keys())
    for ctx in cell_type:

        grn_res = grn_res_ctx[grn_res_ctx['cell_type']==ctx]
        grn_res = grn_res.groupby(['TF'], as_index=False)['Score'].max()    
        grn_res = grn_res.nlargest(min(200,grn_res.shape[0]), "Score")
        sel_TF = set(gold_data[ctx])

        grn_sorted = grn_res.sort_values('Score', ascending=False)
        tf_list = list(grn_sorted['TF'])
        
        max_f1 = 0
        best_prec = 0
        best_recall = 0
        best_n = 0

        max_top_n = min(200, len(tf_list))
        for n in range(1, max_top_n+1):
            pred_TF_set = set(tf_list[:n])
            overlap = len(pred_TF_set & sel_TF)
            prec = overlap / n
            recall = overlap / len(sel_TF) if len(sel_TF)>0 else 0
            if prec + recall > 0:
                f1 = 2 * prec * recall / (prec + recall)
            else:
                f1 = 0
            if f1 > max_f1:
                max_f1 = f1
                best_prec = prec
                best_recall = recall
                best_n = n
        best_f1_list.append(max_f1)
        best_prec_list.append(best_prec)
        best_recall_list.append(best_recall)
        best_n_list.append(best_n)
        if log:
            print(f"{label} in {ctx}: best F1 {max_f1:.3f} at Top {best_n}, Precision {best_prec:.3f}, Recall {best_recall:.3f}")

    tf_recover = pd.DataFrame({
        "method": [label for _ in cell_type],
        "cell_type": cell_type,
        "Best_F1": best_f1_list,
        "Best_Precision": best_prec_list,
        "Best_Recall": best_recall_list,
        "Best_N": best_n_list
    })

    tf_recovery_summary = tf_recover.groupby("method").agg(
        Best_F1_avg=("Best_F1", "mean"),
        Best_Precision_avg=("Best_Precision", "mean"),
        Best_Recall_avg=("Best_Recall", "mean"),
        Best_N_avg=("Best_N", "mean")
    ).round(4)
    return tf_recover, tf_recovery_summary


# obseleted
def grn_umap(df_grn,n_pcs=50):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df_grn.values)
    pca = PCA(n_components=n_pcs)
    pca_result = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        data=pca_result, 
        index=df_grn.index, 
        columns=[f'PC_{i+1}' for i in range(n_pcs)]
    )

    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.3,   
        n_components=2, 
        random_state=42 
    )
    umap_result = reducer.fit_transform(pca_result)

    umap_df = pd.DataFrame(
        data=umap_result, 
        index=df_grn.index, 
        columns=['UMAP_1', 'UMAP_2']
    )
    return umap_df, pca_df

def compute_full_distance_matrix_global(gene_df, peak_df, gene_names):

    gene_df = gene_df[gene_df["gene_name"].isin(gene_names)]
    gene_name_out = gene_df["gene_name"].drop_duplicates()
    gene_df["gene_idx"] = gene_df["gene_name"].map({name: i for i, name in enumerate(gene_name_out)})
    gene_df.dropna(inplace=True)
    peak_df["peak_idx"] = np.arange(len(peak_df))
    peak_name_out = peak_df.iloc[:, :3].apply(lambda row: '-'.join(map(str,row)), axis=1)

    rows = []
    cols = []
    data = []

    prom_rows = []
    prom_cols = []
    prom_data = []

    for chrom in gene_df["chr"].unique():
    # for chrom in ['chr11']:
        print(f"Processing chromosome: {chrom}")
        gene_chrom = gene_df[gene_df["chr"] == chrom]
        peak_chrom = peak_df[peak_df["chr"] == chrom]

        if gene_chrom.empty or peak_chrom.empty:
            continue  

        for gene_idx in set(gene_chrom["gene_idx"]):
            gene_tab = gene_chrom[gene_chrom["gene_idx"] == gene_idx]
            gene_starts = []
            gene_ends = []
            for gene_row in gene_tab.itertuples():
                _gene_start, _gene_end = gene_row.start, gene_row.end
                gene_starts.append(_gene_start)
                gene_ends.append(_gene_end)
            gene_start = np.min(gene_starts)
            gene_end = np.max(gene_ends) 
            
            peak_start_idx = 0

            while peak_start_idx < len(peak_chrom) and (peak_chrom.iloc[peak_start_idx].end < gene_start - 1000000):
                peak_start_idx += 1
                
            for peak_row in peak_chrom.iloc[peak_start_idx:].itertuples():
                peak_start, peak_end, peak_idx = peak_row.start, peak_row.end, peak_row.peak_idx


                if peak_start > gene_end + 1000000:  
                    break
                
                if peak_start <= gene_end and peak_end >= gene_start:

                    prom_rows.append(gene_idx)
                    prom_cols.append(peak_idx)
                    prom_data.append(-1)
                else:
                    distance = min(abs(peak_start - gene_end), abs(peak_end - gene_start))

                    rows.append(gene_idx)
                    cols.append(peak_idx)
                    data.append(distance)
                    if distance <= 0:
                        print(f"gene: start {gene_start}, end {gene_end}")
                        print(f"gene: start {peak_start}, end {peak_end}")


    n_genes = len(gene_name_out)
    n_peaks = len(peak_df)
    distance_matrix = csr_matrix((data, (rows, cols)), shape=(n_genes, n_peaks))
    prom_matrix = csr_matrix((prom_data,(prom_rows,prom_cols)), shape=(n_genes, n_peaks))
    return distance_matrix, prom_matrix, gene_name_out, peak_name_out

def compute_distance_matrix_all(gene_bed_file, peak_bed_file, gene_names):

    gene_cols = ["chr", "start", "end", "gene_name"]
    peak_cols = ["chr", "start", "end"]

    gene_df = pd.read_csv(gene_bed_file, sep="\t", header=None, names=gene_cols)
    peak_df = pd.read_csv(peak_bed_file, sep="\t", header=None, names=peak_cols)

    gene_df = gene_df.sort_values(by=["chr", "start", "end"]).reset_index(drop=True)
    peak_df = peak_df.sort_values(by=["chr", "start", "end"]).reset_index(drop=True)

    distance_matrix, prom_matrix, gene_name_out, peak_name_out = compute_full_distance_matrix_global(gene_df, peak_df, gene_names)
    return distance_matrix, prom_matrix, gene_name_out, peak_name_out



def construct_pseudo_bulk_pstime(rna_tab, atac_tab, pseudo_time, cell_types, cell_latent, n_cell=10):

    if not isinstance(rna_tab, pd.DataFrame):
        raise ValueError("rna_tab 必须是 Pandas DataFrame 类型")
    if not isinstance(atac_tab, pd.DataFrame):
        raise ValueError("atac_tab 必须是 Pandas DataFrame 类型")
    if not isinstance(cell_latent, pd.DataFrame):
        raise ValueError("cell_latent 必须是 Pandas DataFrame 类型")
    if isinstance(pseudo_time, (list, np.ndarray)):
        pseudo_time = pd.Series(pseudo_time, index=rna_tab.index)
    if isinstance(cell_types, (list, np.ndarray)):
        cell_types = pd.Series(cell_types, index=rna_tab.index)
    
    if not (rna_tab.index.equals(atac_tab.index) and 
            rna_tab.index.equals(pseudo_time.index) and 
            rna_tab.index.equals(cell_types.index)):
        raise ValueError("rna_tab、atac_tab、pseudo_time 和 cell_types 的索引必须一致")
    
    pseudo_bulk_rna = []
    pseudo_bulk_atac = []
    pseudo_bulk_meta = []
    pseduo_bulk_latent = []

    for cell_type in cell_types.unique():
        mask = cell_types == cell_type
        rna_sub = rna_tab[mask]
        atac_sub = atac_tab[mask]
        latent_sub = cell_latent[mask]
        pseudo_time_sub = pseudo_time[mask]

        order = pseudo_time_sub.sort_values().index
        rna_sub = rna_sub.loc[order]
        atac_sub = atac_sub.loc[order]
        latent_sub = latent_sub.loc[order]
        pseudo_time_sub = pseudo_time_sub.loc[order]

        n_cells_total = len(rna_sub)
        n_groups = int(np.ceil(n_cells_total / n_cell))

        for i in range(n_groups):
            start = i * n_cell
            end = min((i + 1) * n_cell, n_cells_total)
            cell_indices = rna_sub.index[start:end]

            rna_mean = rna_sub.loc[cell_indices].mean(axis=0)
            atac_mean = atac_sub.loc[cell_indices].mean(axis=0)
            latent_mean = latent_sub.loc[cell_indices].mean(axis=0)
            time_mean = pseudo_time_sub.loc[cell_indices].mean()
            time_min = pseudo_time_sub.loc[cell_indices].min()
            time_max = pseudo_time_sub.loc[cell_indices].max()

            pseudo_bulk_rna.append(rna_mean)
            pseudo_bulk_atac.append(atac_mean)
            pseduo_bulk_latent.append(latent_mean)
            pseudo_bulk_meta.append({
                "cell_type": cell_type,
                "num_cells": len(cell_indices),
                "pseudo_time_mean": time_mean,
                "pseudo_time_min": time_min,
                "pseudo_time_max": time_max
            })

    pseudo_bulk_rna = pd.DataFrame(pseudo_bulk_rna).reset_index(drop=True)
    pseudo_bulk_atac = pd.DataFrame(pseudo_bulk_atac).reset_index(drop=True)
    pseduo_bulk_latent = pd.DataFrame(pseduo_bulk_latent).reset_index(drop=True)
    pseudo_bulk_meta = pd.DataFrame(pseudo_bulk_meta)

    return pseudo_bulk_rna, pseudo_bulk_atac, pseudo_bulk_meta, pseduo_bulk_latent