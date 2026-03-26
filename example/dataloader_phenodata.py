import torch
import glob
import csv
import numpy as np


def identity(sample, label):
    return sample, label

class Standardizer:
    def __init__(self, data, labels):

        self.mean = data.mean(axis = 0, keepdims = True)
        self.std = data.std(axis = 0)

        self.label_mean = labels.mean(axis = 0)
        self.label_std = labels.std(axis = 0)

    def standardize(self, sample, label):

        sample = sample - self.mean
        sample = sample / self.std

        label = label - self.label_mean
        label = label / self.label_std

        return sample, label

    def destandardize(self, sample, label):
        sample = sample * self.std
        sample = sample + self.mean
    
        label = label * self.label_std
        label = label + self.label_mean

        return sample, label

class Normalizer:
    def __init__(self, data, labels):

        self.min = torch.min(data, dim = 0)[0]
        self.max = torch.max(data, dim = 0)[0]

        self.label_min = torch.min(labels, dim = 0)[0]
        self.label_max = torch.max(labels, dim = 0)[0]

    def normalize(self, sample, label):
        sample = sample - self.min
        sample = sample / (self.max - self.min)

        label = label - self.label_min
        label = label / (self.label_max - self.label_min)

        return sample, label

    def denormalize(self, sample, label):
        sample = sample * (self.max - self.min)
        sample = sample + self.min
    
        label = label * (self.label_max - self.label_min)
        label = label + self.label_min

        return sample, label


class PhenoDataset(torch.utils.data.Dataset):
    def __init__(self, 
            data_path, 
            pft = 'DB', 
            device = None,
            warmupdays = 365,
            preprocessing = identity):

        if device is None:
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'

        self.device = device

        print('device set to {}'.format(device))

        if warmupdays > 365:
            print('can not warm up for more than a year. Warmup is set to 365'
                'days.')
            self.warmupdays = 365
        else:
            self.warmupdays = warmupdays

            
        self.preprocessing = identity
        self.pft = pft
        self.data_path = data_path
        file_list = glob.glob(self.data_path + '{}_*.csv'.format(self.pft))
        print(file_list)

        self.siteyears = []
        for file in file_list:
            with open(file, 'r') as fp:
                reader = csv.reader(fp)
                sitename = next(reader)[1]
                row = next(reader)
                minyear = int(row[1])
                maxyear = int(row[2])
                
            for year in range(minyear, maxyear):
                self.siteyears.append((sitename, year + 1))
        
        if preprocessing == 'standardize':
            data = []
            labels = []
            for i in range(len(self.siteyears)):
                data.append(self.__getitem__(i)['features'])
                labels.append(self.__getitem__(i)['target'])

            data = torch.cat(data, axis = 0)
            labels = torch.cat(labels, axis = 0)
            print(data.shape)


            self.preprocessor = Standardizer(data, labels)
            self.preprocessing = self.preprocessor.standardize

        elif preprocessing == 'normalize':
            data = []
            labels = []
            for i in range(len(self.siteyears)):
                data.append(self.__getitem__(i)['features'])
                labels.append(self.__getitem__(i)['target'])

            data = torch.cat(data, axis = 0)
            labels = torch.cat(labels, axis = 0)

            self.preprocessor = Normalizer(data, labels)
            self.preprocessing = self.preprocessor.normalize

        else:
            self.preprocessing = preprocessing


    def __len__(self):
        return len(self.siteyears)

    def __getitem__(self, idx):
        site, year = self.siteyears[idx]

        # print(site, year)


        sample = np.zeros((730, 15))
        label = np.zeros((365, 8))

        with open('{}{}_{}.csv'.format(self.data_path, self.pft, site), 'r') as fp:
            reader = csv.reader(fp)
            next(reader)
            row = next(reader)
            minyear = int(row[1])
            maxyear = int(row[2])
            next(reader)
            next(reader)
            next(reader)

            for i in range(365 * (year - minyear -1)):
                next(reader)
            
            for i, row in enumerate(reader):
                if i >= 730: 
                    break
                for col in range(1,16): 
                    sample[i,col-1] = float(row[col])
                
                if i >= 365:
                    label[i - 365, 0] = float(row[16])
                    label[i - 365, 1] = float(row[17])
                    label[i - 365, 2] = float(row[18])
                    label[i - 365, 3] = float(row[19])
                    label[i - 365, 4] = float(row[20])
                    label[i - 365, 5] = float(row[21])
                    label[i - 365, 6] = float(row[22])
                    label[i - 365, 7] = float(row[23])


        sample = torch.Tensor(sample).to(self.device)
        label = torch.Tensor(label).to(self.device)
        sample, label = self.preprocessing(sample, label)

        start_idx = 365 - self.warmupdays
        sample = sample[start_idx:]
        label = label[start_idx:]

        return  {'features': sample, 'target': label, 'sitename': site, 'year':year}
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, parent, idxs):
        self.parent = parent
        self.idxs = idxs
        self.preprocessor = parent.preprocessor

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.parent.__getitem__(self.idxs[idx])

def leave_onesite_out_crossvalidation(
            data_path,
            pft = 'DB',
            device = None,
            warmupdays = 365,
            preprocessing = identity):

    phenodataset = PhenoDataset(
            data_path = data_path,
            pft = pft,
            device = device,
            warmupdays = warmupdays,
            preprocessing = preprocessing)

    sites = set([])
    siteyears = []
    for siteyear in phenodataset.siteyears:
        siteyears.append(siteyear[0])
        sites.add(siteyear[0])

    training_sets = []
    validation_sets = []
    for site in sites:
        val_idxs = np.where(np.array(siteyears) == site)[0]
        train_idxs = np.where(np.array(siteyears) != site)[0]

        training_sets.append(IndexDataset(phenodataset, train_idxs))
        validation_sets.append(IndexDataset(phenodataset, val_idxs))

    return training_sets, validation_sets


if __name__ == '__main__':
    phenodataset = PhenoDataset('../data/split_dataset/', 'EN') #, preprocessing = 'standardize')
    sample, label = phenodataset.__getitem__(10)
    print(sample[:, 0])
