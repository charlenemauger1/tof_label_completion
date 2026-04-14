from huggingface_hub import snapshot_download
 
# Download the segmentation checkpoints
snapshot_download(
    repo_id="charlenemauger1/complete-me",
    allow_patterns="dynUnet_segmentation/**",
    local_dir="./src/completeme/segmentation/checkpoints/"
)

# Download label completion checkpoint
snapshot_download(
    repo_id="charlenemauger1/complete-me",
    allow_patterns="label_completion/**",
    local_dir="./src/completeme/label_completion/checkpoints/"
)