import csv

with open("test_result-0.1.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)
