import os, pickle 

class UCF101_splitter():
    def __init__(self, root, split):
        self.root = root
        self.split = split

    def get_action_index(self):
        '''
        Create a dictionary relating integer labels (e.g. 1, 2, ... 101) to their respective action classes
        '''
        self.action_label = {}

        # ===== Open class index text file and retrieve lines =====
        with open(os.path.join(self.root,'classInd.txt')) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()

        # ===== Process the text file to associate labels with classes =====
        for line in content:
            label,action = line.split(' ')
            if action not in self.action_label.keys():
                self.action_label[action]=label

        return None 


    def split_video(self):
        '''
        Create dictionaries for the train and testing splits, relating the video name to the paths to the frames contained within that video
        '''
        # ===== Create dictionary relating labels to action classes =====
        self.get_action_index()
        train_video = None
        for path,subdir,files in os.walk(self.root):
            for filename in files:
                # ===== Create dictionary for train split ======
                if filename.split('.')[0] == 'trainlist{:02d}'.format(self.split):
                    train_video = self.file2_dic(os.path.join(self.root,filename))
                
                # ===== Create dictionary for test split =====
                if filename.split('.')[0] == 'testlist{:02d}'.format(self.split):
                    test_video = self.file2_dic(os.path.join(self.root,filename))

        # ===== Correct issue with handstandpushups class : Inconsistency between text file and folder name capitalisation =====
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)
        return self.train_video, self.test_video

    def file2_dic(self,fname):
        '''
        Get a text file containing the list of videos, and return a dictionary relating the video name to the paths to the frames in the video

        Arguments:
            fname : Path to the text file containing a list of videos in the split
        '''

        # ===== Create a list of videonames from the file =====
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()

        # ===== Create a dictionary relating every video name to a list of paths to the frames within that video =====
        dic={}
        for line in content:
            video = line.split('/',1)[1].split(' ',1)[0]
            key = 'v_' + video.split('_',1)[1].split('.',1)[0]
            label = self.action_label[line.split('/')[0]]   
            dic[key] = int(label)
        return dic

    def name_HandstandPushups(self,dic):
        '''
        Account for a discrepancy between the capitalisation of HandstandPushups in the split file and in the files in the dataset
        '''
        dic = {k.replace('HandStandPushups', 'HandstandPushups'):v for k,v in dic.items()}
        return dic 