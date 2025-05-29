import numpy as np
import os
import glob
import shutil

class debug():
    def __init__(self, isDebug):

        global cur_path
        cur_path = os.getcwd()
        global res_path
        global init_path

        # creat folder for outputs
        self.debug_path = os.path.join(cur_path, 'debug')
        self.init_path = os.path.join(os.getcwd(), 'init')
        if os.path.isdir(self.debug_path):
            # os.chdir(res_path)
            res_file = os.path.join(self.debug_path, '*')
            for file in glob.glob(res_file):
                if not os.path.isdir(file):
                    os.remove(file)
        else:
            os.mkdir(self.debug_path)

        self.isDebug = isDebug

        # if os.path.isfile('debug.log'):
        #     os.remove('debug.log')

    def print_results_to_file(self, results, resuls_name):
        if self.isDebug:
            res_path = os.path.join(self.debug_path, str(resuls_name))
            logfile = open(res_path+'.dat', 'a')
            logfile.write(str(results) + '\n')
            logfile.close()

    def read_results_from_file(self, project_name, resuls_name):
        if self.isDebug:
            res_path = os.path.join(self.init_path, project_name, str(resuls_name))
            with open(res_path+'.dat', 'r') as f:
                line = f.read().split('\n')
                c1 = []
                for i in line[:-1]:
                    var = i.split(': ')
                    c1.append(float(var[0]))
            return np.array(c1)

    def pause(self):
        if self.isDebug:
            wait = input("Press Enter to continue.")

    def print(self, results):
        if self.isDebug:
            print('the result is: ' + '\n')
            print(results)
            wait = input("Press Enter to continue.")