import csv
class CarData:
    def __init__(self, dataList):
        self.outputdata = []
        self.sensordata = []
        self.outputdata.append(dataList[0])
        self.outputdata.append(dataList[1])
        self.outputdata.append(dataList[2])
        if (len(dataList) < 25):
            print(dataList)
            raise ValueError("Message")
        for i in range(3, 25):
            self.sensordata.append(dataList[i])

    def get_output_data(self):
        return self.outputdata

    def get_sensor_data(self):
        return self.sensordata


def createCarDataList():
    filepath = 'data_merge.csv'
    car_data_list = []
    with open(filepath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        cnt = 0
        for row in readCSV:
            if (cnt != 0):
                car_data_list.append(CarData(row))

            cnt += 1
    return car_data_list