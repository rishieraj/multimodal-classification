import os.path as osp
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class MaterialDataset(Dataset):
    def __init__(self, args, set_type='train'):
        self.args = args
        self.set_type = set_type
        self.modalities = args.modality_list
        
        # Data paths
        self.data_root = args.data_location
        self.paths = {
            'vision': osp.join(self.data_root, 'vision'),
            'touch': osp.join(self.data_root, 'touch'),
            'audio': osp.join(self.data_root, 'audio_examples')
        }
        
        # Load splits and labels
        with open(osp.join(self.data_root, 'label.json')) as f:
            self.label_dict = json.load(f)
        with open(args.split_location) as f:
            splits = json.load(f)
            self.samples = splits[set_type]
        
        # Print dataset statistics
        print(f"\n{set_type.capitalize()} set size: {len(self.samples)}")
        print(f"Using modalities: {self.modalities}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        obj, instance = self.samples[index]
        data = {
            'names': (obj, instance),
            'label': int(self.label_dict[obj])
        }
        
        # Load modalities
        if 'vision' in self.modalities:
            data['vision'] = self._load_vision(obj, instance)
        if 'touch' in self.modalities:
            data['touch'] = self._load_touch(obj, instance)
        if 'audio' in self.modalities:
            data['audio'] = self._load_audio(obj, instance)
            
        return data

    def _load_vision(self, obj, instance):
        img_path = osp.join(self.paths['vision'], obj, f'{instance}.png')
        img = Image.open(img_path).convert('RGB')
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def _load_touch(self, obj, instance):
        img_path = osp.join(self.paths['touch'], obj, f'{instance}.png')
        img = Image.open(img_path).convert('RGB')
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def _load_audio(self, obj, instance):
        spec_path = osp.join(self.paths['audio'], obj, f'{instance}.npy')
        return torch.tensor(np.load(spec_path)).float()

    def collate_fn(self, batch):
        data = {}
        data['names'] = [item['names'] for item in batch]
        data['label'] = torch.tensor([item['label'] for item in batch])
        
        for modality in self.modalities:
            if modality == 'audio':
                data[modality] = torch.stack([item[modality].unsqueeze(0) for item in batch])
            else:
                data[modality] = torch.stack([item[modality] for item in batch])
        
        return data

class ContrastiveDataset(Dataset):
    def __init__(self, args, set_type='train'):
        self.args = args
        self.set_type = set_type
        self.modalities = args.modality_list
        
        # Data paths
        self.data_root = args.data_location
        self.paths = {
            'vision': osp.join(self.data_root, 'vision'),
            'touch': osp.join(self.data_root, 'touch'),
            'audio': osp.join(self.data_root, 'audio_examples')
        }
        
        # Load splits and labels
        with open(osp.join(self.data_root, 'label.json')) as f:
            self.label_dict = json.load(f)
        with open(args.split_location) as f:
            splits = json.load(f)
            # Reorganize samples into (obj_id, instance_pairs) format
            self.samples = []
            for obj_id, instances in splits[set_type].items():
                if len(instances) > 1:  # Ensure multiple instances for pairing
                    for i in range(len(instances)):
                        for j in range(i+1, len(instances)):
                            self.samples.append((
                                obj_id,
                                (str(instances[i]), str(instances[j]))
                            ))
        
        print(f"\n{set_type.capitalize()} set:")
        print(f"Number of pairs: {len(self.samples)}")
        print(f"Using modalities: {self.modalities}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        obj_id, (instance1, instance2) = self.samples[index]
        
        # Create one-hot label
        label = int(self.label_dict[obj_id])
        label_onehot = torch.zeros(len(self.label_dict))
        label_onehot[label] = 1
        
        data = {
            'label': label,
            'label_onehot': label_onehot,
            'names': (obj_id, (instance1, instance2))
        }
        
        # Load two views for each modality
        for modality in self.modalities:
            data[f'{modality}_1'] = self._load_modality(modality, obj_id, instance1)
            data[f'{modality}_2'] = self._load_modality(modality, obj_id, instance2)
            
        return data

    def _load_modality(self, modality, obj_id, instance):
        if modality in ['vision', 'touch']:
            img_path = osp.join(self.paths[modality], str(obj_id), f'{instance}.png')
            img = Image.open(img_path).convert('RGB')
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        else:  # audio
            spec_path = osp.join(self.paths['audio'], str(obj_id), f'{instance}.npy')
            return torch.tensor(np.load(spec_path)).float()

    def collate_fn(self, batch):
        data = {}
        data['label'] = torch.tensor([item['label'] for item in batch])
        data['label_onehot'] = torch.stack([item['label_onehot'] for item in batch])
        data['names'] = [item['names'] for item in batch]
        
        for modality in self.modalities:
            if modality == 'audio':
                data[f'{modality}_1'] = torch.stack([item[f'{modality}_1'].unsqueeze(0) for item in batch])
                data[f'{modality}_2'] = torch.stack([item[f'{modality}_2'].unsqueeze(0) for item in batch])
            else:
                data[f'{modality}_1'] = torch.stack([item[f'{modality}_1'] for item in batch])
                data[f'{modality}_2'] = torch.stack([item[f'{modality}_2'] for item in batch])
        
        return data