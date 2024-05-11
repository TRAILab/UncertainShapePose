#
# This file is part of https://github.com/TRAILab/UncertainShapePose
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#


import torch
import numpy as np
from pkg_resources import load_entry_point

from torch.utils.data import Dataset

import json
from PIL import Image
import random
import os
import cv2

from torchvision import transforms

# training/eval/test 
def get_dataset_loader_shapenet(config_dataset_dir, dataset_type='test', shuffle_train=True, 
    transform_type=None, random_resized_scale=(0.08,1), label='chairs', gt_data_dir = './data/'):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224,scale=random_resized_scale),
        transforms.RandomHorizontalFlip(),  # augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    # config_dataset_dir = '/media/lzw/HardDisk/Ubuntu/Datasets/ShapeNetRendering/'

    # all those are under ./data. they are gt info
    # gt_data_dir = './data/'

    # training/eval/test 
    if dataset_type == 'test':
        dataset_test = shapenet(config_dataset_dir, gt_data_dir, data_transforms, 
            train=False, test=True, shuffle_train=shuffle_train, transform_type=transform_type, label=label)
    elif (dataset_type == 'training') or (dataset_type == 'train'):
        dataset_test = shapenet(config_dataset_dir, gt_data_dir, data_transforms, 
            train=True, test=False, shuffle_train=shuffle_train, transform_type=transform_type, label=label)
    elif dataset_type == 'eval':
        dataset_test = shapenet(config_dataset_dir, gt_data_dir, data_transforms, 
            train=False, test=False, shuffle_train=shuffle_train, transform_type=transform_type, label=label)

    return dataset_test


class shapenet(Dataset):
    def __init__(self, config_dataset_dir, gt_data_dir, data_transforms, train=True, test=False, subset_ratio=1.0,
        shuffle_train=True, transform_type=None, label='chairs'):
        super(shapenet, self).__init__()

        self.class_name = self.generate_label_id(label)
        self.dataset = 'ShapeNetCore.v2'

        self.init_dir(config_dataset_dir, gt_data_dir, label)

        print('Initializing shapenet with a ratio of:', subset_ratio)
        self.subset_ratio = subset_ratio

        self.generate_dataset(shuffle_train=shuffle_train)

        auto_transform = (transform_type is None) or (transform_type=='auto')
        if train:
            self.dataset = self.training_dataset
            self.dataset_type = 'train'
            if auto_transform:
                transform_type = 'train'
        elif test:
            self.dataset = self.test_dataset
            self.dataset_type = 'test'
            if auto_transform:
                transform_type = 'val'
        else:
            self.dataset = self.eval_dataset
            self.dataset_type = 'eval'
            if auto_transform:
                transform_type = 'val'
        if transform_type == 'eval':
            transform_type = 'val'

        self.transform_type = transform_type
        self.data_transform = data_transforms[transform_type]

        # process data_transforms
        # Dangerous to load all the images.
        # self.images = self.load_images_from_names(self.dataset['images'], data_transform)
        # self.images = process_data_transforms(self.images, data_transforms)

        self.images_names = self.dataset['images']
        self.codes = self.dataset['codes']
        self.images_instance_names = self.dataset['names']

        print('image num:', len(self.images_names))

        # Get partial pointcloud for each images.
        

    def generate_label_id(self, label):
        label_to_id = \
        {
            'chairs': '03001627',
            'cars': '02958343',
            'planes': '02691156',
            'lamps': '03636649'
        }
        return label_to_id[label]

    def init_dir(self, config_dataset_dir, gt_data_dir, label):
        class_name = self.generate_label_id(label)

        render_dataset_dir = config_dataset_dir + '/ShapeNetRendering/'
        render_data_dir = render_dataset_dir + f'/{class_name}/'

        # load groundtruth 
        sdf_gt_dir = gt_data_dir + f'/gt_sdf/{class_name}/'
        self.load_gt_sdf_from_dir(sdf_gt_dir)

        # load gt code
        gt_code_dir = gt_data_dir + f'/gt_code/{class_name}/'
        self.set_gt_code_dir(gt_code_dir)

        # load gt_mesh
        gt_mesh_dir = gt_data_dir + '/gt_mesh/'
        self.load_gt_mesh_from_dir(gt_mesh_dir)

        self.gt_normalization_dir = gt_data_dir + f'/NormalizationParameters/{class_name}/'
        self.gt_sdf_samples_dir = gt_data_dir + f'/SdfSamples/{class_name}/'

        self.load_render_images(render_data_dir)

        # Currently only chair has gt 
        if label == "chairs":
            code_dir = gt_data_dir + f'/codes/chairs_64/2000.pth'
        else:
            code_dir = None
        code_name = gt_data_dir + f'/splits/sv2_{label}_train.json'
        self.load_codes(code_dir, code_name)

    def load_gt_sdf_from_dir(self,dir):
        self.dir_gt_sdf = dir

    # ./data/gt_mesh
    def load_gt_mesh_from_dir(self,dir):
        self.dir_gt_mesh = dir

    def __len__(self):
        return len(self.images_names)

    def get_ins_num(self):
        image_num = self.__len__()
        assert(image_num % self.num_render == 0)
        ins_num = image_num // self.num_render
        return ins_num

    def __getitem__(self, index, use_data_transform = True):
        # image_name = self.images[index]
        imageName = self.images_names[index] # load
        if use_data_transform:
            data_transform = self.data_transform
        else:
            data_transform = None
        images = self.load_images_from_names([imageName], data_transform)
        code = self.codes[index]
        # resize code [1,64] -> [64]
        code_squeeze = np.squeeze(code, axis=0)
        return images[0], code_squeeze

    def get_name(self, index):
        return self.images_instance_names[index]

    def get_gt_sdf(self, index):
        if self.dir_gt_sdf is None:
            raise Exception('Please set gt_sdf dir with load_gt_sdf_from_dir().')  
       
        dir = self.dir_gt_sdf + '/' + self.get_name(index) + '.npy'
        # load sdf
        sdf_gt = np.load(dir)

        return sdf_gt

    def set_gt_code_dir(self, dir):
        self.dir_gt_code = dir

    def get_gt_code(self, index):
        if self.dir_gt_code is None:
            raise Exception('Please set gt code dir with set_gt_code_dir().')  
       
        dir = self.dir_gt_code + '/' + self.get_name(index) + '.pth'
        # load sdf
        gt_code = torch.load(dir).numpy()

        return gt_code

    def load_images_from_names(self,image_name_list, data_transform=None):
        images = []
        for name in image_name_list:
            image = Image.open(name).convert('RGB')

            if data_transform is not None:
                # process data transofrm on image
                image = data_transform(image)

            # save to image list
            images.append(image)

        return images



    def load_render_images(self, render_data_dir):
        self.render_data_dir = render_data_dir
    
    def load_codes(self, code_dir_car, code_name_dir):
        with open(code_name_dir, "r") as f:
            file = json.load(f)
            self.code_names = file['ShapeNetV2'][self.class_name]  # not repeated

        self.code_dir_car = code_dir_car

        if code_dir_car is None:
            self.codes = [None for i in range(len(self.code_names))]
        else:
            code_training = torch.load(code_dir_car)
            self.codes = code_training['latent_codes'].detach().numpy()



    def generate_dataset(self, shuffle_train = True):
        self.num_render = 24
        image_datasets = []

        # load codes
        image_names = []
        code_list = []
        code_names = []
        for id,code in enumerate(self.codes):
            # load images of the name
            name = self.code_names[id]

            image_dir = self.render_data_dir + '/' + name + '/rendering/' 

            # load all iamges under the dir , 00-23.png
            for x in range(0,self.num_render):
                name_id = "{:0>2d}".format(x)
                image_names.append(image_dir + name_id + ".png")
                code_list.append(code)
                code_names.append(name)

        # code_list = self.codes
        # get a small subset
        # subset_ratio = 0.001
        # subset_ratio = 0.05
        subset_ratio = self.subset_ratio
        print('Generate dataset with a ratio of', subset_ratio)

        # divide to training and eval
        training_ratio = 0.7
        eval_ratio = 0.2
        used_ratio = training_ratio+eval_ratio

        total_ins_num = len(self.codes)
        valid_ins_num = round(total_ins_num * subset_ratio)
        training_ins_num = round(valid_ins_num * training_ratio)
        eval_ins_num = round(valid_ins_num * eval_ratio)
        test_ins_num = round(valid_ins_num * (1-training_ratio-eval_ratio))
        # test_ins_num = valid_ins_num - training_ins_num - eval_ins_num
        id_training = training_ins_num * 24
        id_eval = id_training + eval_ins_num * 24
        id_test = id_eval + test_ins_num * 24

        # Check if id_test == id_total
        assert(id_test == (training_ins_num+eval_ins_num+test_ins_num)*24)

        # total_num = round(len(image_names))
        # total_valid_num = round(total_num * subset_ratio * used_ratio)
        # id_training = round(total_valid_num*training_ratio)

        self.training_dataset = {'images': image_names[:id_training], 'codes': code_list[:id_training], 'names': code_names[:id_training]}
        self.eval_dataset = {'images': image_names[id_training:id_eval], 
                'codes': code_list[id_training:id_eval],
                'names': code_names[id_training:id_eval]}
                
        self.test_dataset = {'images': image_names[id_eval:id_test], 'codes': code_list[id_eval:id_test],
                'names': code_names[id_eval:id_test]}

        # random shuffle training datasets
        # use the same seed for image and code
        if shuffle_train:
            random.seed(0)
            random.shuffle(self.training_dataset['images'])
            random.seed(0)
            random.shuffle(self.training_dataset['codes'])
            random.seed(0)
            random.shuffle(self.training_dataset['names'])

        # lst = [1,2,3,4,5,6]
        # lst2 = [1,2,3,4,5,6]
        # random.seed(0)
        # random.shuffle(lst)
        # random.seed(0)
        # random.shuffle(lst2)
        # print(lst)
        # print(lst2)
        
        # return self.training_dataset

        # save the train and eval name list to files
        print('save training and eval images names to ./logs/...')
        os.makedirs('./logs/',exist_ok=True)
        with open("./logs/training_images.txt", "w") as fp:   #Pickling
            fp.write('\n'.join(self.training_dataset['images']))
        with open("./logs/eval_images.txt", "w") as fp:   #Pickling
            fp.write('\n'.join(self.eval_dataset['images']))
        with open("./logs/test_images.txt", "w") as fp:   #Pickling
            fp.write('\n'.join(self.test_dataset['images']))

    def training_datasets(self):  # for code, find corresponding image dir
        return self.training_dataset
    
    def eval_datasets(self):
        return self.eval_dataset

    def get_instance_id_from_name(self,name):
        repeat_id = self.get_index_from_name(name)
        ins_id = repeat_id // (self.num_render-1)
        return ins_id

    def get_index_from_name(self,name):
        # find name in self.code_names, which is the first id among all the 24 images of the instance
        return self.images_instance_names.index(name)

    def get_index_from_code(self,code):
        return self.codes.index(code)

    def get_gt_mesh_path(self,index):
        if self.dir_gt_mesh is None:
            raise Exception('Please set gt_mesh dir with load_gt_mesh_from_dir().')  
       
        instance_name = self.get_name(index)
        ground_truth_samples_filename = os.path.join(
            self.dir_gt_mesh,
            self.class_name,
            instance_name + ".ply",
        )
        return ground_truth_samples_filename

    def get_normalization_path(self,index):
        instance_name = self.get_name(index)
        n_filename = self.gt_normalization_dir + instance_name + '.npz'
        return n_filename

    def get_first_index_from_instance(self, ins):
        return self.generate_index_list([0], ins)[0]

    def generate_index_list(self, view_list, instance):
        return instance * self.num_render + np.array(view_list)

    def get_instance_num(self):
        ins_num = len(self.images_names) / self.num_render
        return round(ins_num)

    def get_root_dir(self):
        return self.render_data_dir

    def load_saved_aug_image(self, index):
        root_dir = self.get_root_dir() + '/augmentation/'
        im = torch.load(root_dir + f'/{index}.pt')
        im_vis = cv2.imread(root_dir+f'/{index}_vis.png') / 255.0

        return im, im_vis