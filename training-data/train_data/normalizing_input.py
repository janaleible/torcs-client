import numpy as np
import csv
import pandas as pd




file_names = ["aalborg", "alpine-1", "f-speedway"]


for i in range(len(file_names)):
    with open("/home/student/Documents/torcs-server/torcs-client//training-data/train_data/" + file_names[i]+".csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        data_list = []
        for row in reader:
            data_list.append(row)

        num_row = len(data_list)
        num_col = len(data_list[0])
        data_mtx = np.zeros((num_row, num_col))

        for j in range(num_row):
            for k in range(num_col):
                data_mtx[j][k] = data_list[j][k]

        column_means = np.zeros(num_col)
        column_means = np.sum(data_mtx, axis=0)/(num_row)
        column_std = data_mtx.std(0)


        for l in range(3,num_col):
            data_mtx[:,l] = (data_mtx[:,l]-column_means[l])/column_std[l]

        df = pd.DataFrame(data_mtx, columns = header)
        df.to_csv(file_names[i]+"normalized.csv", index = False, header = True, sep = ",")

        # np.savetxt(file_names[i]+"normalized.csv", data_mtx, delimiter=",")

