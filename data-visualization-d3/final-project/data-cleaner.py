__author__ = 'thomas.almenningen'

import csv

"""
  Cleans and removes superfluous data from the titanic data set

  Columns of interest:

    #Survival (Index 1)
    #Passenger class (Index 2)
    #Sex (Index 4)
    #Age (Index 5)
    #Siblings/Spouses aboard (Index 6)
    #Parents/Children aboard (Index 7)
    #Passenger fare (Index 9)
    #Port of Embarkation (Index 11)

"""

indices = (1, 2, 4, 5, 6, 7, 9, 11)

cleanData = []

with open('data.csv') as data:
    reader = csv.reader(data, delimiter=',', quotechar='"')
    for row in reader:
        cleanData.append([row[i] for i in indices])
        print(cleanData[-1][0])
        cleanData[-1][0] = "Perished" if cleanData[-1][0] == "0" else "Survived"

with open("clean-data.csv", 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(cleanData)


