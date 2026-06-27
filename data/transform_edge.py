import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class Transforms(object):
    def __init__(self, input_size=256):
        self.input_size = input_size
        self.final_resize = transforms.Resize((self.input_size, self.input_size))

    def __call__(self, _data):
                                     
        img1, img2, cd_label, edge_label = _data['img1'], _data['img2'], _data['cd_label'], _data['edge_label']

              
        if random.random() < 0.5:
            img1_, img1 = img1, img2
            img2 = img1_

              
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            cd_label = TF.hflip(cd_label)
            edge_label = TF.hflip(edge_label)                                    

              
        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            cd_label = TF.vflip(cd_label)
            edge_label = TF.vflip(edge_label)                                    

            
        if random.random() < 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            cd_label = TF.rotate(cd_label, angle)
            edge_label = TF.rotate(edge_label, angle)                                    
            
              
                                   
                                                                                                                 
                                                                                                       
                                                                                                                 
                                                                                                                 
                                                                                                                        
                                                                                                                                                                
        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=(self.input_size, self.input_size)).get_params(img=img1, scale=[0.333, 1.0], ratio=[0.75, 1.333])
            img1 = TF.resized_crop(img1, i, j, h, w, size=(self.input_size, self.input_size), interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resized_crop(img2, i, j, h, w, size=(self.input_size, self.input_size), interpolation=InterpolationMode.BILINEAR)
            cd_label = TF.resized_crop(cd_label, i, j, h, w, size=(self.input_size, self.input_size), interpolation=InterpolationMode.NEAREST)
            edge_label = TF.resized_crop(edge_label, i, j, h, w, size=(self.input_size, self.input_size), interpolation=InterpolationMode.NEAREST)


                                            
        return {'img1': img1, 'img2': img2, 'cd_label': cd_label, 'edge_label': edge_label}
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
