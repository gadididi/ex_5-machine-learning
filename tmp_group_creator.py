import os
import shutil

path_train = "./gcommands/train"
path_valid = "./gcommands/valid"
path_test = "./gcommands/test"

# your path to the repository
compath = "C:/Users/Gadi/Documents/C grade/machine learning/ex5/ex_5/ex_5-machine-learning/"


def create(folder, kind, original, items):
    os.mkdir(folder)
    subdir = [x[0] for x in os.walk(original)]
    subdir = subdir[1:]
    for m_dir in subdir:
        num_file = 0
        the_dir = m_dir.split("\\")[1]
        new_place = folder + "/" + the_dir
        os.mkdir(new_place)
        for file in os.listdir(m_dir):
            if file == ".":
                continue
            if num_file > items:
                num_file = 0
                break
            num_file += 1
            shutil.copyfile(compath + "/gcommands/" + kind + "/" + the_dir + "/" + file,
                            compath + new_place + "/" + file)


create("short_train", "train", path_train, 250)
create("short_valid", "valid", path_valid, 50)
