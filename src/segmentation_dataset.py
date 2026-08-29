from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "images"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"

class SegmentationDataset:
    """PyTorch-style Dataset supplying (pre-disaster image, binary building-mask) pairs for segmentation training."""
    
    def __init__(self, location_ids):
        """Stores the list of location IDs this dataset instance will serve."""
        self.location_ids = location_ids
    
    def __len__(self):
        """Returns the total number of locations in this dataset."""
        return len(self.location_ids)    
    
    def __getitem__(self, index):
        """Loads and returns the pre-disaster image and its matching target mask for the location at the given index."""
        located_id = self.location_ids[index] # e.g. hurricane-harvey_00000042
        image_path = IMAGES_DIR / f"{located_id}_pre_disaster.png"
        target_path = TARGETS_DIR / f"{located_id}_pre_disaster_target.png"
        
        image = Image.open(image_path)
        target = Image.open(target_path)
        
        return image, target