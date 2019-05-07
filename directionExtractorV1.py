import re
from numpy.ma import array
from os import listdir
from os.path import isfile, join
import csv
import netifaces



def direction_finder(source_mac_address, destination_mac_address):
    # These strings have a "\n" appended to the end of them
    source_mac_address = source_mac_address.strip()
    destination_mac_address = destination_mac_address.strip()

    this_pc = netifaces.ifaddresses(netifaces.interfaces()[7])[netifaces.AF_LINK][0]["addr"]
    if this_pc in source_mac_address:
        # print("S", source_mac_address)
        return 1
    elif this_pc in destination_mac_address:
        # print("D", destination_mac_address)
        return -1
    else:
        # print("S and D", source_mac_address, destination_mac_address)
        return 0


files = array([file for file in listdir("D:\\VideoFingerprint\\Output\\") if
               isfile(join("D:\\VideoFingerprint\\Output\\", file))])
for file_counter in range(0, len(files)):
    with open("D:\\VideoFingerprint\\Output\\" + files[file_counter]) as file_name:
        # print(file_name)
        # readlines gives a list of rows of file
        line_array = []
        lines = file_name.read().split("\n")
        for line in lines:
            line = line.replace("\x00", "")
            # print(line)
            if line.strip().split("\t") is not []:
                line_array.append(line.strip().split("\t"))
        # dataread = csv.reader(file_name, delimiter=',')


        with open("D:\\VideoFingerprint\\NewOutput\\" + files[file_counter][0: files[file_counter].index(".")] + ".csv", "w+") as csv_file:
            csv_writer = csv.writer(csv_file)

            print("Starting writing into file "+files[file_counter])
            row = ["Direction"]
            csv_writer.writerow(row)

            for row_counter in range(0, len(line_array)):
                # print(len(line_array[row_counter]))
                if len(line_array[row_counter]) != 1:
                    # print("Line", line_array[row_counter])
                    row = [direction_finder(str(line_array[row_counter][0]), str(line_array[row_counter][1]))]
                    csv_writer.writerow(row)
            print("Writing into file "+files[file_counter]+" finished!")