from PIL import Image
from pathlib import Path
import random
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "images"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"

class SegmentationDataset:
    """PyTorch-style Dataset supplying (pre-disaster image, binary building-mask) pairs for segmentation training."""
    
    def __init__(self, location_ids, augment):
        """Stores the list of location IDs this dataset instance will serve."""
        self.location_ids = location_ids
        self.augment = augment # augmentation only applies to training set
    
    def __len__(self):
        """Returns the total number of locations in this dataset."""
        return len(self.location_ids)    
    
    def __getitem__(self, index):
        """Loads and returns the pre-disaster image and its matching target mask for the location at the given index."""
        located_id = self.location_ids[index] # e.g. hurricane-harvey_00000042
        image_path = IMAGES_DIR / f"{located_id}_pre_disaster.png"
        target_path = TARGETS_DIR / f"{located_id}_pre_disaster_target.png"
        
        # Original (1024, 1024) image and target
        image = Image.open(image_path)
        target = Image.open(target_path)
        
        # Downsampling (1024, 1024) to (512, 512)
        image = image.resize((512, 512), Image.Resampling.LANCZOS) 
        target = target.resize((512, 512), Image.Resampling.NEAREST) # preserving [0, 1]
        
        image = transforms.ToTensor()(image)
        target = torch.from_numpy(np.array(target)).float().unsqueeze(0)  # FIXED: no /255 scaling
        
        # ------- Augmentation Pipeline Design ------------
        if self.augment:
            # 1. Horizontal flip
            if random.random() < 0.5: # one random decimal between 0 and 1
                image = TF.hflip(image)
                target = TF.hflip(target)
            
            # 2. Vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                target = TF.vflip(target)
            
            # 3. Random Rotation
            angle = random.uniform(-15, 15) # one random decimal between -15 deg and +15 deg
            
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR
            )   
            
            target = TF.rotate(
                target,
                angle,
                interpolation=InterpolationMode.NEAREST
            )
            
            # 4. Brightness Jitter
            brightness_factor = random.uniform(0.8, 1.2)
            
            image = TF.adjust_brightness(
                image, 
                brightness_factor
            )
            
            # 5. Contrast Jitter
            contrast_factor = random.uniform(0.8, 1.2)
            
            image = TF.adjust_contrast(
                image, 
                contrast_factor
            )
            
        return image, target
    