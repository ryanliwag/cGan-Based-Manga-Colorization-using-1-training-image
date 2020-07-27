import glob
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

transforms_orig = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Resize((128, 128), Image.BICUBIC),
                transforms.RandomHorizontalFlip()])

class ImageDataset(Dataset):
    def __init__(self, root, transforms_orig=None):
        '''
        Args:
            root: folder containing the training Images
            transform (callable, optional)
        '''
        
        files = glob.glob('%s/*.*' % root)
        self.training_files = files * 4
        self.transforms_orig = transforms_orig

        #transforms for rgb
        self.transforms_rgb = transforms.Compose([
                transforms.ToTensor()])

        #transforms for greyimage
        self.transforms_bw = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()])
        
    def __len__(self):
        return len(self.training_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.training_files[idx])
        
        orig_image = self.transforms_orig(img)
        
        true_image = self.transforms_rgb(orig_image)
        bw_image = self.transforms_bw(orig_image)
        
        return {"true_image": true_image, "conditional_image": bw_image}