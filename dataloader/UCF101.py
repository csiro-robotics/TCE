'''
Create train and test split dataset loaders for UCF101
'''
from __future__ import print_function

import os
import pickle 
import random 
from PIL import Image 

import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 

from split_train_test_video import UCF101_splitter

class UCF_dataset(Dataset):
    def __init__(self, dic, root_dir, mode, transform=None):
        '''
        Creates a dataset for training or testing on the UCF101 dataset

        Arguments:
            dic : Dictionary relating a video name to a list containing paths to the frames in that video
            root_dir : Path to the directory containing frames images for the UCF101 dataset
            mode : Set dataset to either training or evalution
            transform : Transformations to be applied to the frames before passing them to the network
        '''
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        '''
        Returns length of dataset
        '''
        return len(self.keys)

    def load_ucf_image(self,video_name, index):
        '''
        Loads a frame at a given frame index in a given video
        
        Arguments:
            video_name : The name of the video being sampled from
            index : The index of that frame within the video
        '''
        # Create path template for the video's frames
        path = self.root_dir + 'v_' + video_name + '/frame'
        
        # While loop to try and repeatedly load the image in case of a temporary filestore error
        unloaded = True
        count = 0
        while unloaded:
            try:
                # Load image at given index
                img = Image.open('{}{:06d}.jpg'.format(path,index))
                unloaded = False
            except:
                count += 1
                if count-1 % 100 == 0:
                    print("Image failed : {}{} Count : ", path +str(index).zfill(6)+'.jpg', "count", count)
        
        # Print new line if previous print statement occured
        if count > 0 :
            print('')
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):
        '''
        Get training sample for a video at index idx

        Arguments:
            idx : index for training video to be loaded from the dataset
        '''
        if self.mode == 'train':
            # Load training sample
            # Get video name, number of frames and video ground truth
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            label = self.values[idx]
            label = int(label)-1

            # Randomly sample a starting frame for the sampled frames
            start = random.randint(1, nb_clips-6)

            # Create stack-of-differences input to the network
            data = []
            for index in range(start, start+5):
                data.append(self.load_ucf_image(video_name, index) - self.load_ucf_image(video_name, index+1))
            stacked = torch.cat(data, dim=0)
            sample = (stacked, label)
            return sample

        else:
            # Load validation sample
            # Get video name, index to start the validation stack at, and ground truth
            video_name, start = self.keys[idx].split(' ')
            start = abs(int(start))
            label = self.values[idx]
            label = int(label)-1

            # Create stack-of-differences input to the network
            data = []
            for index in range(start, start+5):
                data.append(self.load_ucf_image(video_name, index) - self.load_ucf_image(video_name, index+1))
            stacked = torch.cat(data, dim=0)
            sample = (video_name, stacked, label)
            return sample

class UCF101():
    def __init__(self, batch_size, num_workers, path, ucf_list, ucf_split):
        '''
        Wrapper class for train and testing datasets for UCF101

        Arguments:
            batch_size : Batch size
            num_workers : Number of CPU workers for multiprocessing
            path : Path to root directory for UCF101 dataset with videos split into frames 
            ucf_list : Path to folder containing splits for UCF101
            ucf_split : Dataset split to be used for training and testing
        '''
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = path
        self.ucf_list = ucf_list 
        self.ucf_split = ucf_split 
        self.frame_count = {}
        # Split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()

    def load_frame_count(self):
        '''
        Loads a dictionary relating the name of each video to the number of frames in that video from a pickle file
        If the pickle file does not already exists, creates the dictionary and saves it to the split file directory
        '''
        if not os.path.exists(os.path.join(self.ucf_list, 'frame_count.pickle')):
            
            # Build dictionary if no pickle file found 
            print('Creating frame count dictionary for UCF101 : ')
            dic_frame = {}
            for idx, video_directory in enumerate(os.listdir(self.data_path)):
                key = video_directory + '.avi'
                frame_count = len(os.listdir(os.path.join(self.data_path, video_directory)))
                dic_frame[key] = frame_count
                print('Counted frames for video {} of {}'.format(idx + 1, len(os.listdir(self.data_path))), end = '\r')
            print('\nCreated frame count dictionary')
            
            # Save dictionary
            print('Saving dictionary to disk : ')
            with open(os.path.join(self.ucf_list, 'frame_count.pickle'), 'wb') as file:
                pickle.dump(dic_frame, file)
            file.close()
            print('Done')

        else:
            # Load dictionary from pickle file if found
            print('Loading frame count dictionary for UCF101 from pickle file : ')
            with open(os.path.join(self.ucf_list, 'frame_count.pickle'), 'rb') as file:
                dic_frame = pickle.load(file)
            file.close()
            print('Done')

        # Handle capitalisation mismatch for HandstandPushups
        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line]
        
        return None

    def get_training_dic(self):
        '''
        Get dictionary relating the names of videos in the training split to lists of paths to their frames
        '''
        self.dic_training={}
        for video in self.train_video:
            nb_frame = self.frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
        return None

    def get_validation_dic(self):
        '''
        Get dictionary for constructing the dataset for the test split
        This dictionary contains 19 items for each video, for the 19 stacks averaged over for validation
        Key names are video + starting frame for the stack, value is a list of the paths to frames for that video
        '''
        self.dic_testing={}
        for video in self.test_video:
            # Retrieve number of frames
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)
            itera = 19
            for i in range(itera):
                # Create key-value pair for each of the 19 validation stacks in the video
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]


    def train(self):
        '''
        Create the dataset loader for the train split
        '''
        # Create the image transform
        tran =  transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        
        # Create the pytorch dataset for the train split
        training_split = UCF_dataset(dic=self.dic_training, 
                                       root_dir=self.data_path, 
                                       mode='train',
                                       transform = tran
                                       )
        print('==> Training data : {} Videos'.format(len(training_split)))

        # Create the dataset loader for the train split
        train_loader = DataLoader(
            dataset=training_split, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
            )
        
        return train_loader

    def validate(self):
        '''
        Create the dataset loader for the test split
        '''
        # Create the image transform
        tran = transforms.Compose([transforms.Resize([224,224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                   ])

        # Create the pytorch dataset for the validation split
        validation_set = UCF_dataset(dic=self.dic_testing,
                                         root_dir=self.data_path, 
                                         mode='val',
                                         transform = tran
                                         )
        print('==> Validation data : {} frames'.format(len(validation_set ) / 19))

        # Create the dataset loader for the train split
        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

    def run(self):
        '''
        Returns dataset loaders for train and test split, as well as labels for the videos in the test split
        '''
        self.load_frame_count()
        self.get_training_dic()
        self.get_validation_dic()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video
