import os
import sys
import numpy as np
import csv

DATA_DIR = 'mixFiles'

def addToCsv(file , outFile):
    tid = np.array([])
    text = np.array([])
    label = np.array([])
    filename = os.path.splitext(file)[0]
    with open(file , 'r' , encoding = 'utf-8') as csvFile:
        csvReader = csv.DictReader(csvFile)
        fieldname = csvReader.fieldnames
        for row in csvReader:
            #tid = np.append(tid, row[fieldname[8]])
            #text = np.append(text, row[fieldname[9]])    
            #if (row[fieldname[5]] != 'not_related_or_irrelevant'):
            #   label = np.append(label, '1')
            #else:
            #    label = np.append(label , '0')
            tid = np.append(tid , row[fieldname[0]])
            text = np.append(text , row[fieldname[1]])
            label = np.append(label , row[fieldname[2]])
                
        print("The size of this file is: %s" %np.size(tid))
        zipped = zip(tid,text,label)
        fieldname = ['tweet_id', 'tweet_text', "label"]
        with open(outFile , 'a' , encoding= 'utf-8', newline='') as out:
            csvWriter = csv.writer(out)
            if (os.path.getsize(outFile) == 0):
                csvWriter.writerow(fieldname)
            csvWriter.writerows(zipped)
    

def main(fol):
    f = []
    for (dirpath , dirnames , filenames) in os.walk(fol):
        f.extend(filenames)
    print('The files in the folder are: %s' %f)
    #if needed to add from different folders just comment the next line out to just append both folders. 
    open('mixFiles/mix.csv', 'w').close()
    for file in ([file for file in f if not 'mix.csv'==file]):
        print("Adding CSV file %s to mix file:" %file)
        addToCsv(os.path.join(dirpath,file) , os.path.join(DATA_DIR , 'mix.csv'))
        print("File added")
    print("All files in folder added to mix.")
    
    
if __name__ == "__main__":
    main(sys.argv[1])