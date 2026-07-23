for i in {300..310..20}; do
    python /apdcephfs_zwfy3/share_302867165/ewencai/CODE/G-OPD/verl/scripts/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir /apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/code_rl/code_qwen3-4b-fullparam_lr1e-6_DAPO/global_step_300/actor \
            --target_dir /apdcephfs_zwfy3/share_302867165/ewencai/CODE/GOPD/verl/examples/code_rl/code_qwen3-4b-fullparam_lr1e-6_DAPO/global_step_300/hf_model
done