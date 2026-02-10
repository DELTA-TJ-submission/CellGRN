method_path="/home/shaliu_fu/multireg/cellGRN/code/grn_method/"
input_path="/home/shaliu_fu/multireg/cellGRN/data/spatial_brain/"
output_path="/home/shaliu_fu/multireg/cellGRN/data/spatial_brain/"
soft_config="/home/shaliu_fu/multireg/benchmark/multiGRN_bench/algor_config.json"



mkdir -p ${output_path}/

source ~/miniconda3/etc/profile.d/conda.sh

conda activate LINGER_env

cd ${output_path}

mkdir -p linger_output

(/usr/bin/time -f 'Elapsed time: %E\nMemory usage: %M KB\nCPU usage: %P' \
python ${method_path}/linger_grn.py \
${input_path} \
${output_path}linger_output/ \
# # ~{write_json(config)} \
> /dev/null ) \
2> ${output_path}/linger_output/log.txt

conda deactivate