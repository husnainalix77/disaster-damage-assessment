import os
from collections import defaultdict
class DisasterInspector:
    """Inspects xBD dataset directories to count images per disaster type and phase."""
    def __init__(self, root_dir:str):

        self.root_dir = root_dir
        self.folders = ["images", "labels", "targets"]
        self.counts = defaultdict(lambda:{f: 0 for f in self.folders}) 
        # Structure: {'disaster_name': {'images': count, 'labels': count, 'targets': count}}
          
    def _validate_structure(self):
        """Ensures all three expected subdirectories exist."""
        for folder in self.folders:
            folder_path = os.path.join(self.root_dir, folder) # e.g: data/raw/train/images
            if not os.path.exists(folder_path):
                print(f"Directory not found: {folder_path}")
                return False
        return True
    
    def _extract_disaster_name(self, fname: str) -> str:
        """Extracts disaster prefix from filename regardless of extension."""
        parts = fname.split('_')
        return parts[0] if len(parts) >= 3 else None
        
    def analyze_directory(self):
        """Scans images, labels, and targets folders to accumulate file counts."""
        if not self._validate_structure():
            return
        
        for folder in self.folders:
            folder_path = os.path.join(self.root_dir, folder)
            valid_extns = ('png', 'json')
            for fname in os.listdir(folder_path):
                if not fname.endswith(valid_extns):
                    continue
                
                disaster_name = self._extract_disaster_name(fname)
                if disaster_name:
                    self.counts[disaster_name][folder] += 1
    
    def display_summary(self):
        """Prints a comparison table verifying folder file parity."""
        if not self.counts:
            print("No data to display.")
            return
        
        # Print the result
        print(f"{'Disaster Name':<22} | {'Images':<8} | {'Labels':<8} | {'Targets':<8} | {'Parity Status'}")
        print("-" * 65)
        
        # Check the matching
        for disaster, f_counts in sorted(self.counts.items()):
            images_count = f_counts['images']
            labels_count = f_counts['labels']
            targets_count = f_counts['targets']
            # Match check: images count must equal labels count and targets count
            status = "OK" if (images_count == labels_count == targets_count and images_count > 0) else "MISMATCH!"
            print(f"{disaster:<22} | {images_count:<8} | {labels_count:<8} | {targets_count:<8} | {status}")
        
# Main Execution Workflow
if __name__ == "__main__":
    print("=== TRAIN SET ===")
    inspector = DisasterInspector("data/raw/train")
    inspector.analyze_directory()
    inspector.display_summary()
    print("\n=== TEST SET ===")
    inspector = DisasterInspector("data/raw/test")
    inspector.analyze_directory()
    inspector.display_summary()