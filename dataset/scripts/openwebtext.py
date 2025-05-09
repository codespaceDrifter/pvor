from datasets import load_dataset
from usb import usb_path
import os

openwebtext_dataset = load_dataset("openwebtext", trust_remote_code=True)
openwebtext_dataset.save_to_disk(os.path.join(usb_path(), "openwebtext"))

#clear huggingface cache later
#rm -rf ~/.cache/huggingface/datasets
