import sys
import csv
import numpy as np
import os
from sklearn.model_selection import KFold



def main(csvfile):
    #reading data from csvfile and getting fieldnames/header and creating 3 different arrays for tweet_id, tweet_text and tweet_label.
    with open(csvfile, 'r' , encoding="utf-8") as file:
        tid = np.array([])
        text = np.array([])
        label = np.array([])
        rows = csv.DictReader(file)
        header = rows.fieldnames
        for row in rows:
            tid = np.append(tid , row[header[0]])
            text = np.append(text ,row[header[1]])
            label = np.append(label, row[header[2]])
    
    
    
    kf = KFold(n_splits=5) # n_splits = sum of train to dev to test ratio
    #splitting the data to train and test
    for train_index, test_index in kf.split(tid):
        train = zip(tid[train_index], text[train_index], label[train_index])
        test = zip(tid[test_index], text[test_index], label[test_index])
        
    #getting the filename
    filename = os.path.splitext(csvfile)[0]
    filename = os.path.basename(filename)
    
    #creating and outputing the train and test data to two different files.
    with open ("%s_train.csv" %filename , 'w', encoding= 'utf-8' , newline='') as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(train)
        
    with open ("%s_test.csv" %filename , 'w', encoding= 'utf-8' , newline='') as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(test)     
    
    
    
    
if __name__ == "__main__":
    main(sys.argv[1])