from torch.utils.data import DataLoader

def load(type,dataset,batch_size,num_workers,type_save='standard'):
    train, test = [],[]
    if (type_save=='standard') :
        file_dir = 'save/'
    if (type_save=='same_size') :
        file_dir = 'save_same/'
    
    if type==0 :
        with open(file_dir + 'old_train.txt','r') as file:
            train_id=[line for line in file]
        with open(file_dir + 'old_val.txt','r') as file:
            test_id =[line for line in file]
    if type==1 :
        with open(file_dir + 'recent_train.txt','r') as file:
            train_id=[line for line in file]
        with open(file_dir + 'recent_val.txt','r') as file:
            test_id =[line for line in file]
    if type==2 :
        with open(file_dir + 'now_train.txt','r') as file:
            train_id=[line for line in file]
        with open(file_dir + 'now_val.txt','r') as file:
            test_id =[line for line in file]
    if type==3 :        
        with open(file_dir + 'now_train.txt','r') as file:
            train_id1=[line for line in file]
        with open(file_dir + 'now_val.txt','r') as file:
            test_id1 =[line for line in file]            
        with open(file_dir + 'recent_train.txt','r') as file:
            train_id2=[line for line in file]
        with open(file_dir + 'recent_val.txt','r') as file:
            test_id2 =[line for line in file]
        train_id = train_id1 +train_id2
        test_id = test_id1+ test_id2
            
    train_id = [x.replace('\n', '') for x in train_id]    
    test_id = [x.replace('\n','') for x in test_id]
    train = DataLoader(dataset.images_from_sequences(train_id),batch_size= batch_size,num_workers=num_workers,shuffle=True)
    test = DataLoader(dataset.images_from_sequences(test_id),batch_size= batch_size,num_workers=num_workers,shuffle=False)
    
    
    return train, test
