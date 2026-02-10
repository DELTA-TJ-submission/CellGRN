method_path="/home/shaliu_fu/multireg/cellGRN/code/grn_method/"
input_path="/home/shaliu_fu/multireg/cellGRN/data/spatial_brain/"
output_path="/home/shaliu_fu/multireg/cellGRN/data/spatial_brain/"
soft_config="/home/shaliu_fu/multireg/benchmark/multiGRN_bench/algor_config.json"


mkdir -p ${output_path}/
# mkdir -p ${output_path}/scenic2_output

source ~/miniconda3/etc/profile.d/conda.sh

conda activate scenic2_env

cd ${output_path}

mkdir -p scenic2_output
# rm -rf ./scenic2_output # ensure empty

# scenicplus init_snakemake --out_dir scenic2_output

cd ./scenic2_output
# which python

(/usr/bin/time -f 'Elapsed time: %E\nMemory usage: %M KB\nCPU usage: %P' \
python ${method_path}/scenic2_grn.py \
${input_path} \
${output_path}scenic2_output \
$soft_config \
# # ~{write_json(config)} \
> /dev/null ) \
2> ${output_path}/scenic2_output/prepare_log.txt

cd Snakemake
/usr/bin/time -f 'Elapsed time: %E\nMemory usage: %M KB\nCPU usage: %P' \
snakemake --cores 20 > ${output_path}/scenic2_output/log.txt 2>&1


conda deactivate