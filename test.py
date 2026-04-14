from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="TanishqMittal/MedicalNet",
    filename="resnet50.pth"
)
print(path)
