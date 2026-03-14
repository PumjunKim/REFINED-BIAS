srun --partition=A6000 \
     --job-name=_chest_arch \
     --time=192:00:00 \
     --qos=gpu_qos_4 \
     --gres=gpu:1 \
     --mem=16G \
     --cpus-per-task=16 \
     --chdir=/workspace/seungah.lee/shape-texture/REFINED-BIAS/refined_bias \
     --nodelist=potato-02 
     --pty bash -lc "\
       source ~/anaconda3/etc/profile.d/conda.sh && \
       conda activate refined-bias  && \
    jupyter lab --no-browser --ip=0.0.0.0 --port=8888"
