import json
from shapely.wkt import loads as wkt_loads
from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "labels"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "images"

class ClassificationDataset:
    """Dataset for extracting building crops and their damage-class labels from disaster images."""
    damage_class_labels = {
        "no-damage": 0,
        "minor-damage" : 1,
        "major-damage" : 2,
        "destroyed" : 3
    }
    
    def __init__(self, location_ids):
        """Initializes the dataset with building records from the given image IDs."""
        self.records = []
        
        for location_id in location_ids: #santa-rosa-wildfire_00000073
            json_path = LABELS_DIR / f"{location_id}_post_disaster.json"
            
            with open(json_path, "r") as f:
                d = json.load(f)
                
                for building in d.get("features", {}).get("xy", []):
                    wkt_str = building["wkt"]
                    subtype = building.get("properties", {}).get("subtype") # damage_class
                    polygon = wkt_loads(wkt_str) # shapely polygon object
                    
                    if subtype in ClassificationDataset.damage_class_labels:  # only keep the 4 real damage classes
                        self.records.append((location_id, polygon, subtype))
                        

    def __len__(self):
        """Returns the total number of building samples in the dataset."""
        return len(self.records)
    
    def __getitem__(self, index):
        """Returns a resized building crop tensor and its corresponding damage-class label.""" 
        # Get building's record from the list
        location_id, polygon, subtype = self.records[index]

        # Turn the building's polygon into a bounding box
        min_x, min_y, max_x, max_y = polygon.bounds 
        
        # Load the post-disaster image
        image_path = IMAGES_DIR / f"{location_id}_post_disaster.png"
        image = Image.open(image_path)
        
        # Add padding proportional to the building's own size (e.g. 15% of width/height)
        box_width = max_x - min_x
        box_height = max_y - min_y
        padding_x = box_width * 0.15
        padding_y = box_height * 0.15

        left = max(0, int(min_x - padding_x))
        top = max(0, int(min_y - padding_y))
        right = min(image.width, int(max_x + padding_x))
        bottom = min(image.height, int(max_y + padding_y))

        # Crop the building region
        crop = image.crop((left, top, right, bottom))
        
        # Resize crop to the fixed input size expected by the pretrained model
        crop = crop.resize((224, 224), Image.Resampling.LANCZOS) # helps produce smoother enlargement
        
        # Convert crop to tensor
        cropped_image_tensor = transforms.ToTensor()(crop)
        
        # Damage class label number
        damage_label_number = ClassificationDataset.damage_class_labels[subtype]
        
        return cropped_image_tensor, damage_label_number


        

        
        
        
    
    
                            
            
    