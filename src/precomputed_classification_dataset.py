from pathlib import Path
from PIL import Image
from src.classification_dataset import ClassificationDataset
import json

class PrecomputedClassificationDataset:
    """Precomputes and saves building crops for classification."""
    
    def __init__(self, ids, processed_dir):
        """Initialize the precomputation dataset using the specified IDs and output folder."""
        self.dataset = ClassificationDataset(ids) # create the classification dataset object
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess(self):
        """Create and save all building crops."""   
        manifest = [] # track (filename, label) pairs
        
        for building_index in range(len(self.dataset)): # loop through all buildings
            
            # Get location id from existing dataset record
            location_id = self.dataset.records[building_index][0]
            
            crop_tensor, damage_label = self.dataset[building_index]
            
            # Convert [C, H, W] tensor → [H, W, C], then numpy array 
            crop_array = crop_tensor.permute(1, 2, 0).numpy()
            
            # Convert tensor pixel values 
            crop_array = (crop_array * 255).astype("uint8")
            
            # Convert numpy array to PIL 
            crop_image = Image.fromarray(crop_array)
            
            # Create and save the filename
            filename = f"{location_id}_{building_index}.png"
            save_path = self.processed_dir / filename
            crop_image.save(save_path)
            
            manifest.append((filename, damage_label))
            
            if building_index % 5000 == 0: # for indication of working
                print(f"Processed {building_index}/{len(self.dataset)}")

        # Save the json file
        with open(self.processed_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        print("Manifest saved.")

             
