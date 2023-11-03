"""
读取各数据集
包含read_data,view_data方法
"""
import os

import numpy as np
import pandas as pd
from datetime import datetime
from src.myutils import *
import scipy.io as sio


class PanasonicData():

    """
    calce = CalceDataINR181650('data/Calce/INR 18650-20R Battery')
    data = calce.read_data('DST', "80", '0')
    """

    def __init__(self, path='data/Panasonic 18650PF Data'):
        self.root_path=path
        self.temp=[-20,-10,0,10,25]
        self.test_kind=['5 pulse test', 'Charges and Pauses', 'Drive Cycles', 'EIS']
        self.drive_cycles=['NN','Cycle1','Cycle2','Cycle3','Cycle4','LA92','UDDS','US06']
        self.pre_cut=pd.read_csv(os.path.join(path,'pre_cut.csv'),index_col=0)
        self.post_cut = pd.read_csv(os.path.join(path, 'post_cut.csv'),index_col=0)
        self.max_capacity=2.9


    def preprocess(self):

        for t in self.temp:
            filepath = os.path.join(self.root_path,'{}degC'.format(t), 'Drive Cycles')
            filenames=os.listdir(filepath)
            for f in filenames:
                if '-' not in f:
                    continue
                nf=f.split('_')[2:]
                nf=''.join(nf)
                os.rename(os.path.join(filepath,f),os.path.join(filepath,nf))

    def get_data(self,temp,params=['Drive Cycles','Cycle1']):

        test_kind,subkind=params
        if test_kind=='Drive Cycles':
            filepath=os.path.join(self.root_path,'{}degC'.format(temp),test_kind,'{}Pan18650PF.mat'.format(subkind))
            data = sio.loadmat(filepath)
        time=data['meas']['Time'][0][0].flatten()[self.pre_cut.loc[subkind,str(temp)]:self.post_cut.loc[subkind,str(temp)]]
        voltage=data['meas']['Voltage'][0][0].flatten()[self.pre_cut.loc[subkind,str(temp)]:self.post_cut.loc[subkind,str(temp)]]
        current=data['meas']['Current'][0][0].flatten()[self.pre_cut.loc[subkind,str(temp)]:self.post_cut.loc[subkind,str(temp)]]
        ah=data['meas']['Ah'][0][0].flatten()[self.pre_cut.loc[subkind,str(temp)]:self.post_cut.loc[subkind,str(temp)]]
        battery_temp=data['meas']['Battery_Temp_degC'][0][0].flatten()[self.pre_cut.loc[subkind,str(temp)]:self.post_cut.loc[subkind,str(temp)]]
        ah=[a+self.max_capacity for a in ah]
        soc=[a/self.max_capacity for a in ah]
        if time[0]!=0:
            t=time[0]
            time=[tt-t for tt in time]
        return time, voltage, current, soc, battery_temp

    def view_data(self):
        """
        查看数据，将原始数据绘图保存在 fig/Calce/INR 18650-20R 文件夹中
        :return:
        """
        save_path=os.path.join('fig','Panasonic')
        for t in self.temp:
            for dc in self.drive_cycles:
                time, ut, I, ah, battery_temp=self.get_data(t,['Drive Cycles',dc])

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # 宽，高
                data = pd.DataFrame({'time': time,
                                     'I': I,
                                     'ut': ut,
                                     'soc':ah,
                                     'temp':battery_temp})
                fig = plt.figure(num=1, figsize=(15, 8), dpi=120)
                ax1 = fig.add_subplot(111)
                ax1.plot(time, I, color='r', label='Current')
                # plt.plot(range(0,len(time)),time)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('I (A)')
                # ax1.set_ylim((-350,150))
                ax1.grid()
                ax1.legend(loc=1)

                ax2 = ax1.twinx()
                ax2.plot(time, ut, color='green', label='Voltage')
                # plt.plot(range(0,len(time)),time)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Ut (V)')
                # ax2.set_ylim((3, 4))
                ax2.grid()
                ax2.legend(loc=2)

                plt.savefig(os.path.join(save_path, '{}_{}C.png'.format(dc, t)))
                plt.show()
                data.to_csv(os.path.join(save_path, '{}_{}C.csv'.format(dc, t)))




