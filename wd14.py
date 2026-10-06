import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

WD14_REPO = 'SmilingWolf/wd-swinv2-tagger-v3'
_wd14_instance = None

class WD14Tagger:
    def __init__(self, repo_id=WD14_REPO):
        self.repo_id = repo_id
        self.target_size = 448
        self._load_model()

    def _load_model(self):
        print(f"Loading WD14 Tagger ({self.repo_id})...")
        try:
            csv_path = hf_hub_download(repo_id=self.repo_id, filename="selected_tags.csv", local_files_only=True)
            model_path = hf_hub_download(repo_id=self.repo_id, filename="model.onnx", local_files_only=True)
        except Exception:
            csv_path = hf_hub_download(repo_id=self.repo_id, filename="selected_tags.csv")
            model_path = hf_hub_download(repo_id=self.repo_id, filename="model.onnx")

        self.df = pd.read_csv(csv_path)
        # Category 0 = general tags (pose, clothing, background, etc.)
        self.general_indices = self.df[self.df["category"] == 0].index.to_numpy()
        self.general_tags = self.df.loc[self.general_indices, "name"].tolist()

        # Run on CPU to preserve VRAM for JoyCaption and prevent CUDA DLL conflicts
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("WD14 Tagger loaded cleanly on: CPU (0 VRAM used)")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert('RGB')
        width, height = image.size
        max_dim = max(width, height)
        scale = self.target_size / max_dim
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        
        resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        padded = Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))

        # Convert to BGR float32 format expected by SmilingWolf taggers
        img_array = np.array(padded, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        return np.expand_dims(img_array, axis=0)

    def predict(self, image: Image.Image, threshold: float = 0.35) -> list[str]:
        input_tensor = self.preprocess_image(image)
        raw_output = self.session.run(None, {'input': input_tensor})[0][0]

        predicted_tags = []
        for idx, tag in zip(self.general_indices, self.general_tags):
            score = float(raw_output[idx])
            if score >= threshold:
                predicted_tags.append((tag, score))

        # Sort by confidence descending
        predicted_tags.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in predicted_tags]

def get_wd14_tagger():
    global _wd14_instance
    if _wd14_instance is None:
        _wd14_instance = WD14Tagger()
    return _wd14_instance

def predict_wd14_tags(image: Image.Image, threshold: float = 0.35) -> list[str]:
    tagger = get_wd14_tagger()
    return tagger.predict(image, threshold=threshold)
