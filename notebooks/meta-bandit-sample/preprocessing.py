import pandas as pd
import numpy as np
import time
import datetime

from os import listdir
from os.path import isfile, join

import random


def preprocess(path, timetrack=False):
    # the path can be both a file or a folder containing multiple files
    # Should it be a folder, be sure that the folder contains ONLY the dataset files unzipped
    try:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        onlyfiles.sort()
        files=[path+"/"+str(file) for file in onlyfiles]
    except:
        files=[path]

    if timetrack:
        count = 0
        t0 = time.time()
        lt = time.time()
        print(count)
    data = []
    for file_path in files:

        file = open(file_path, "r")
        if timetrack:
            print("--New File--")
            print(str(file_path))

        for line in file:
        #     Removing 'not data' parts of the line string
            line = line.replace('id-', '')
            line = line.replace('\n', '')
            line = line.replace('user', '')
            line = line.replace('|', '')
            line = line.replace('  ', ' ')

            aux_str = ''
            info = 0 # 0 = time; 1 = clicked_article; 2 = click; 3 = user_features; 4 = articles_list
            features = np.zeros(136, dtype=np.bool)
            articles_list = []
            for i in line:
                if i == ' ':
                    if info == 0:
                        timestamp = int(aux_str)
                        aux_str = ''
                        info+=1
                    elif info == 1:
                        clicked_article = int(aux_str)
                        aux_str = ''
                        info+=1
                    elif info == 2:
                        click = int(aux_str)
                        aux_str = ''
                        info+=1
                    elif info == 3:
                        try:
                            features[int(aux_str)-1] = 1
                            aux_str = ''
                        except:
                            articles_list.append(int(aux_str))
                            aux_str = ''
                            info+=1
                    elif info == 4:
                        articles_list.append(int(aux_str))
                        aux_str = ''
                else:
                    aux_str+=i

            articles_list.append(int(aux_str))
            aux_str = ''
            if timetrack:
                count += 1
                if count%100000 == 0:
                    t1 = time.time()
                    dt = t1-t0
                    print(str(count)+" - "+str(datetime.timedelta(seconds=t1-lt))+" - "+str(datetime.timedelta(seconds=dt)))
                    lt = t1
            data.append({'Timestamp': timestamp, 'Clicked_Article': clicked_article, 'Click': click, 'User_Features': features, 'Article_List': np.asarray(articles_list)})

        file.close()

    df = pd.DataFrame(data)
    return df


def read_sample(filename, p=0.01):
    # keep the header, then take only 1% of lines
    # if random from [0,1] interval is greater than 0.01 the row will be skipped
    df = pd.read_csv(
        filename,
        skiprows=lambda i: i > 0 and random.random() > p
    )

    return df
