from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import random
import json

class PrecomputedTrainingDataset:
    """Fast-loading dataset for training, using precomputed crops with optional augmentation."""
    
    def __init__(self, processed_dir, augment):
        """Initialize the dataset with the processed crop directory and augmentation setting."""
        self.processed_dir = Path(processed_dir)
        self.augment = augment
        
        with open(self.processed_dir / "manifest.json", "r") as f:
            self.manifest = json.load(f)
    
    def __len__(self):
        """Return the total number of precomputed building crops in the dataset.""" 
        return len(self.manifest)
    
    def __getitem__(self, index):
        """Load, optionally augment, and return one building crop with its damage label."""
        filename, label = self.manifest[index]
        image = Image.open(self.processed_dir / filename) # open the image
        image = transforms.ToTensor()(image) # covert into tensor
        
        if self.augment: # only train_ids will be augmented
            
            if random.random() < 0.5:
                image = TF.hflip(image)
                
            if random.random() < 0.5:
                image = TF.vflip(image)
            
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
        
        return image, label