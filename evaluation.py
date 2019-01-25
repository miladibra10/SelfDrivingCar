import os
import csv
import matplotlib.pyplot as plt


class Evaluation:

    @staticmethod
    def steering_metric(path, col=3):
        data = []

        # extract data from file
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                data.append(abs(float(row[col])))
                line_count += 1

        # calculation of information distribution
        GE_0_LE_10 = 0
        G_10_LE_30 = 1
        G_30_LE_60 = 2
        G_60_LE_100 = 3
        sizes = [0, 0, 0, 0]
        for steering in data:
            if 0 <= steering <= 0.1:
                sizes[GE_0_LE_10] += 1
            elif 0.1 < steering <= 0.3:
                sizes[G_10_LE_30] += 1
            elif 0.3 < steering <= 0.6:
                sizes[G_30_LE_60] += 1
            elif 0.6 < steering <= 1:
                sizes[G_60_LE_100] += 1

        labels = ['0 <= steering <= 10', '10 < steering <= 30', '30 < steering <= 60', '60 < steering <= 100']
        explode = (0.1, 0, 0, 0)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')

        # save result in file
        name = os.path.basename(path).split('.')[0]
        directory = os.path.dirname(path)
        plt.savefig(os.path.join(directory, name))
