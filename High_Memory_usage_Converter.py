'''
This Version of code, stores the one-hot transactions
on RAM, After storing the one-hot transactions on RAM
it writes the whole dataset to the file.
'''

import numpy as np
import os


def Find_Path():
    '''
    Return the current path of files
    '''
    path = os.getcwd()+'/Datasets'
    return path


def Get_List_Of_Dataset_Names(path):
    '''
    Extracting the list of Dataset files in the Actual Datasets directory
    To convert into equivalent one-hot
    '''
    datasets = os.listdir(path+'/Actual Datasets')
    return datasets


def Read_Dataset_From_CSV(path, dataset, delimiter):
    '''
    '''
    df = []
    maximum = 0
    with open(path+'/Actual Datasets/'+dataset) as file:
        for line in file:
            transaction = [int(item)
                           for item in line.replace('\n', '').split(delimiter)]
            df.append(transaction)
            maximum = np.max([maximum, np.max(transaction)])
    return maximum, df


def Make_Equivalent_OneHot_Dataset(df, maximum):
    dataset = []
    for i in range(len(df)):
        transaction = np.zeros(maximum+1)
        for j in range(len(df[i])):
            transaction[df[i][j]] = 1
        transaction = np.array(transaction, dtype=np.int)
        dataset.append(list(transaction))
    return dataset


def Write_OneHot_Dataset_To_File(path, dataset, name_of_dataset, delimiter):
    dataset = np.array(dataset, dtype=np.int8)
    np.savetxt(path+'{}/{}01.csv'.format('/One-hot', name_of_dataset),
               dataset, fmt='%i', delimiter=delimiter)


if __name__ == '__main__':
    path = Find_Path()
    Datasets = Get_List_Of_Dataset_Names(path)
    delimiter = ','
    for dataset in Datasets:
        print(dataset)
        if dataset.split('.')[-1] == 'csv':
            name_of_dataset = os.path.splitext(dataset)[0]
            maximum, df = Read_Dataset_From_CSV(path, dataset, delimiter)
            one_hot_dataset = Make_Equivalent_OneHot_Dataset(df, maximum)
            Write_OneHot_Dataset_To_File(
                path, one_hot_dataset, name_of_dataset, delimiter)
            print('Done !')
