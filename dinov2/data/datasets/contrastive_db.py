import glob 
from typing import Any

from PIL import Image 
from torch.utils.data import Dataset

class TrainContrastiveDataset(Dataset):
    def __init__(self, path_image, transforms = None) -> None:
        super().__init__()

        self.path_image = path_image 

        if str(path_image).endswith("/"):
            self.list_image = glob.glob(path_image + "*.jpg")
        else:
            self.list_image = glob.glob(path_image + "/*.jpg")
        
        assert len(self.list_image) > 0, self.path_image
        
        self.transforms = transforms

    
    def __getitem__(self, index) -> Any:
        image = Image.open(self.list_image[index]).convert("RGB")

        if self.transforms is not None: 
            image = self.transforms(image) 
        
        return image, None # 2th returned value is index class, but contrastive is not need
    
    def __len__(self) -> int: 
        return len(self.list_image)

        


