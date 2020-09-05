'''
This Version of code, doesn't store the one-hot transactions
on RAM, it directly writes the transaction into the file.
'''

import numpy as np
import os


def Find_Path():
    path = os.getcwd()+'/Datasets'
    return path


def Get_List_Of_Dataset_Names(path):
    # Extracting the list of Dataset files in the Actual Datasets directory
    # To convert into equivalent one-hot
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


def Make_Equivalent_OneHot_Dataset(path, name_of_dataset, df, maximum, delimiter):
    create_file(path, name_of_dataset)
    path_to_write = path+'{}/{}01.csv'.format('/One-hot', name_of_dataset)
    with open(path_to_write, 'a') as file:
        for i in range(len(df)):
            transaction = np.zeros(maximum+1, dtype=np.int)
            for j in range(len(df[i])):
                transaction[df[i][j]] = 1
            transaction = transaction.tolist()
            transaction = [str(item) for item in transaction]
            str_transaction = delimiter.join(transaction)
            file.write(str_transaction+'\n')


def create_file(path, name_of_dataset):
    f = open(path+'{}/{}01.csv'.format('/One-hot', name_of_dataset), "w+")
    f.close()


if __name__ == '__main__':
    path = Find_Path()
    Datasets = Get_List_Of_Dataset_Names(path)
    delimiter = ','
    for dataset in Datasets:
        print(dataset)
        name_of_dataset = os.path.splitext(dataset)[0]
        maximum, df = Read_Dataset_From_CSV(path, dataset, delimiter)
        Make_Equivalent_OneHot_Dataset(
            path, name_of_dataset, df, maximum, delimiter)
        print('Done !')
