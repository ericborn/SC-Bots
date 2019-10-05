# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:04:19 2019

@author: Eric Born
"""

path = 'G:/test/'

# list of all files
files = os.listdir(path)

# object for single array file, which equals a single game
#full_array = np.load(path + files[0], allow_pickle=True)

# column list for all match data
#cols = ['minerals','gas','supply_cap', 'supply_army', 'supply_workers',
#        'nexus', 'c_pylons', 'assimilators', 'gateways', 'cybercore', 'robofac',
#        'stargate', 'robobay', 'k-structures', 'k-units', 'attack',
#        'assimilators', 'offensive_force', 'b_pylons', 'workers', 'distribute',
#        'nothing', 'expand', 'buildings', 'ZEALOT', 'STALKER', 'ADEPT',
#        'IMMORTAL', 'VOIDRAY', 'COLOSSUS', 'difficulty', 'outcome']
#
## Single array df
##full_df = pd.DataFrame(data=full_array,columns=cols)
#
## define empty dataframe using columns previously defined
#full_df = pd.DataFrame(columns=cols)

#del big_array
big_array = np.memmap('myFile',dtype=np.int,mode='w+',shape=(5000000,32))

a = np.load(path+files[0])
b = np.load(path+files[1])

c = np.concatenate([a,b])

d = np.load(path+files[2])

e = np.concatenate([c,d])

e.shape

for i in range(files):
    a = np.load(path+files[i])
    b = np.load(path+files[i+1])


a = np.concatenate(np.load(path+files[1]), axis = 0)

a[:,:]
c.shape[:,32]

big_array[a.shape[0],a.shape[1]] = np.load(path+files[0])

big_array.shape

array = np.array()

for i, file in enumerate(files):
    #print(file)
    big_array[i,:] = np.load(path+file)
big_array.flush()