import pandas as pd
import numpy as np

def preprocess(path):

    file = open(path, "r")

    df = pd.DataFrame(columns=['Timestamp', 'Clicked_Article', 'Click', 'User_Features', 'Article_List'])

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
        df = df.append({'Timestamp': timestamp, 'Clicked_Article': clicked_article, 'Click': click, 'User_Features': features, 'Article_List': np.asarray(articles_list)}, ignore_index=True)

    file.close()

    return df
