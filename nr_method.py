import sympy as sp
import numpy as np
import pandas as pd
from pprint import pprint as pp
from scipy.sparse import csc_matrix


class NRMethod:
    
    def __init__(self):
        # self.bus_names_indices = {'Allen':1,
        #                           'Betty':2,
        #                           'Clyde':3,
        #                           'Doug':4,
        #                           'Eve':5
        #                           }
        
        # self.z = {'Allen-Betty':0.009+0.0041j,
        #           'Allen-Doug':00.007+0.055j,
        #           'Doug-Clyde':0.011+0.061j,
        #           'Doug-Eve':0.006+0.045j,
        #           'Eve-Clyde':0.010+0.051j
        #           }
        
        self.data = pd.DataFrame({
            'Bus': ['Allen', 'Betty', 'Clyde', 'Doug', 'Eve'],
            'Type': ['slack', 'pv', 'pq', 'pq', 'pq'],
            'index': [1,2,3,4,5],
            'p_gen': [0,210,0,0,0],
            'q_gen': [0,50,0,0,0],
            'p_load': [0,0,110,100,150],
            'q_load': [0,0,85,95,120],
            'q_cap': [0,0,150,50,0]
            })
        
        self.tl_details = pd.DataFrame({
            'From': ['Allen', 'Betty','Allen', 'Doug', 'Doug', 'Clyde'],
            'To': ['Betty', 'Eve', 'Doug', 'Eve', 'Clyde', 'Eve'],
            'Rse': [0.009,0.006,0.007,0.006,0.011,0.010],
            'Xse': [0.041,0.037,0.055,0.045,0.061,0.051],
            'MVA': [125,250, 200,125,80,75]
            })

    def round_complex(self, num):
        '''
        Not called in main.
        '''
        real = round(num.real, 3)
        imag = round(num.imag, 3)
        return complex(real, imag)
    
    def rect_polar(self, num):
        '''
        Not called in main.
        '''
        mag = round(abs(num), 3)
        ang = round(np.angle(num, deg=True), 3)
        return mag, ang
    
    def y_bus (self):
        '''
        Called in main. 
            - Create the Y bus matrix. A fundamental part of the NR solution
            - Create the polar and rectangular forms of the Y values. Makes it easier to process later.
        '''
        self.y = {'Y_complex':[], 'Y_magnitude':[], 'Y_angle':[]}
        for index, row in self.tl_details.iterrows():
            r = row['Rse']
            x = row['Xse']
            z = complex(r,x)
            # y = 1/z
            y = self.round_complex(num=1/z)
            y_mag, y_ang = self.rect_polar(num=y)
            self.y['Y_complex'].append(y)
            self.y['Y_magnitude'].append(y_mag)
            self.y['Y_angle'].append(y_ang)
        # concatenate self.y into self.tl_details
        self.tl_details = pd.concat([self.tl_details, pd.DataFrame(self.y)], axis=1)

    def output_vector(self):
        y = sp.symbols(f"theta1:{len(self.data)+1}")
        theta_vec = sp.Matrix(y)
        print(theta_vec)
        
        pass
    
    
nr = NRMethod()
nr.y_bus()
nr.output_vector()
