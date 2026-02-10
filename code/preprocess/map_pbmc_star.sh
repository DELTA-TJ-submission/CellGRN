outdir='/home/shaliu_fu/multireg/benchmark/datasets/pbmc/CD4_STARR/mapping/'
mkdir -p ${outdir}

for sample_id in "SRR23649084" "SRR23649086" "SRR23649087" "SRR23649089" "SRR23649090" "SRR23649092" "SRR23649085" "SRR23649088" "SRR23649091" "SRR23649093" "SRR23649094" "SRR23649095" "SRR23649096";do
	# bowtie2 -p 16 -x /home/shaliu_fu/genomes/NCBI_GRCh38/Sequence/Bowtie2Index/genome -1 /home/shaliu_fu/multireg/benchmark/datasets/pbmc/CD4_STARR/${sample_id}_1.fastq -2  /home/shaliu_fu/multireg/benchmark/datasets/pbmc/CD4_STARR/${sample_id}_2.fastq -S ${outdir}/${sample_id}.sam
    samtools view -@ 16 -bS ${outdir}/${sample_id}.sam > ${outdir}/${sample_id}.bam
    samtools sort -@ 16 ${outdir}/${sample_id}.bam -o ${outdir}/${sample_id}_sorted.bam
    samtools index ${outdir}/${sample_id}_sorted.bam
    echo
done

# samtools merge -f  ${outdir}/CD4_STARR.bam -@ 16 ${outdir}/SRR23649084_sorted.bam ${outdir}/SRR23649086_sorted.bam ${outdir}/SRR23649087_sorted.bam ${outdir}/SRR23649089_sorted.bam ${outdir}/SRR23649090_sorted.bam ${outdir}/SRR23649092_sorted.bam ${outdir}/SRR23649085_sorted.bam ${outdir}/SRR23649088_sorted.bam ${outdir}/SRR23649091_sorted.bam ${outdir}/SRR23649093_sorted.bam 

samtools merge -f ${outdir}/plasmid_input.bam -@ 16 ${outdir}/SRR23649094_sorted.bam ${outdir}/SRR23649095_sorted.bam ${outdir}/SRR23649096_sorted.bam 

for sample_id in "SRR23649084" "SRR23649086" "SRR23649087" "SRR23649089" "SRR23649090" "SRR23649092" "SRR23649085" "SRR23649088" "SRR23649091" "SRR23649093";do

    macs2 callpeak -t ${outdir}/${sample_id}_sorted.bam -c ${outdir}/plasmid_input.bam -f BAM -g hs -n ${sample_id} -q 0.01 --outdir ${outdir}
done

cat ${outdir}/SRR*.narrowPeak | sort -k1,1 -k2,2g | mergeBed -i stdin > ${outdir}/STARR_peak.bed

