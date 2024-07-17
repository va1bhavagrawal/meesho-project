import torch
from PIL import Image
from clip import clip
import dino 
from torchvision import transforms

class MetricEvaluator: 
    def __init__(self, device, batch_size=8):
        self.device = device
        self.batch_size = batch_size
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.dino_model, self.dino_preprocess = dino.load_model(device=self.device)
        self.cleanup() 
        self.clip_similarities = [] 
        self.dino_similarities = [] 

    def register(self, images, prompts, gt_images): 
        self.images = images
        self.prompts = prompts
        self.gt_images = gt_images
    
    def cleanup(self): 
        self.images = []
        self.prompts = []
        self.gt_images = []

    def compute_clip_similarities(self):
        """
        Computes the CLIP similarities for all registered images and prompts in batches.
        
        Returns:
            list of float: List of CLIP similarity scores.
        """
        similarities = []
        for i in range(0, len(self.images), self.batch_size):
            batch_images = self.images[i:i + self.batch_size]
            batch_prompts = self.prompts[i:i + self.batch_size]

            image_tensors = torch.stack([self.clip_preprocess(img).to(self.device) for img in batch_images])
            text_tensors = clip.tokenize(batch_prompts).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensors)
                text_features = self.clip_model.encode_text(text_tensors)

            batch_similarities = (100.0 * image_features @ text_features.T).cpu().numpy().diagonal().tolist()
            similarities.extend(batch_similarities)
        
        return similarities

    def compute_dino_similarities(self):
        """
        Computes the DINO similarities for all registered images and ground truth images in batches.
        
        Returns:
            list of float: List of DINO similarity scores.
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.dino_preprocess.image_mean, std=self.dino_preprocess.image_std)
        ])

        similarities = []
        for i in range(0, len(self.images), self.batch_size):
            batch_generated_images = self.images[i:i + self.batch_size]
            batch_ground_truth_images = self.gt_images[i:i + self.batch_size]

            generated_tensors = torch.stack([preprocess(img).to(self.device) for img in batch_generated_images])
            ground_truth_tensors = torch.stack([preprocess(img).to(self.device) for img in batch_ground_truth_images])

            with torch.no_grad():
                generated_features = self.dino_model(generated_tensors)[0]
                ground_truth_features = self.dino_model(ground_truth_tensors)[0]

            batch_similarities = torch.nn.functional.cosine_similarity(generated_features, ground_truth_features).cpu().numpy().tolist()
            similarities.extend(batch_similarities)
        
        return similarities

# Example usage
if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = MetricEvaluator(device, batch_size=8)

    generated_images = [Image.open("gen_image1.jpg"), Image.open("gen_image2.jpg")]  # Replace with your images
    ground_truth_images = [Image.open("gt_image1.jpg"), Image.open("gt_image2.jpg")]  # Replace with your images
    prompts = ["prompt1", "prompt2"]  # Replace with your prompts

    evaluator.register(generated_images, prompts, ground_truth_images)

    clip_similarities = evaluator.compute_clip_similarities()
    dino_similarities = evaluator.compute_dino_similarities()

    print("CLIP Similarities:", clip_similarities)
    print("DINO Similarities:", dino_similarities)
