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
            'From': ['Allen-0', 'Betty-1','Allen-0', 'Doug-3', 'Doug-3', 'Clyde-2'],
            'To': ['Betty-1', 'Eve-4', 'Doug-3', 'Eve-4', 'Clyde-2', 'Eve-4'],
            'Rse': [0.009,0.006,0.007,0.006,0.011,0.010],
            'Xse': [0.041,0.037,0.055,0.045,0.061,0.051],
            'MVA': [125,250, 200,125,80,75]
            })
        
        self.conn_map = {'1': [2, 4],
                         '2': [1,5],
                         '3': [4,5],
                         '4': [1,3,5],
                         '5': [2,3,4]
                         }

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
    
    def calc_admittances (self):
        '''
        Called in main. 
            - Create the Y bus matrix. A fundamental part of the NR solution
            - Create the polar and rectangular forms of the Y values. Makes it easier to process later.
        '''
        y = {'Y_complex':[], 'Y_magnitude':[], 'Y_angle':[]}
        for index, row in self.tl_details.iterrows():
            r = row['Rse']
            x = row['Xse']
            z = complex(r,x)
            y_comp = self.round_complex(num=1/z)
            y_mag, y_ang = self.rect_polar(num=y_comp)
            y['Y_complex'].append(y_comp)
            y['Y_magnitude'].append(y_mag)
            y['Y_angle'].append(y_ang)
        # concatenate self.y into self.tl_details
        self.tl_details_conc = pd.concat([self.tl_details, pd.DataFrame(y)], axis=1)

    def unknown_vector(self):
        '''
        Called in main. Creating the output vector, containing the voltage and the
        angle vectors for each unknown bus.
        '''
        
        unknown = sp.symbols(f"P1 theta2 theta3 theta4 theta5")
        unknown_vec = sp.Matrix(unknown)
        return unknown_vec
    
    def known_vector(self):
        known = sp.symbols("delta2 delta3 delta4 delta5 V3 V4 V5")
        known_vec = sp.Matrix(known)
        return known_vec
    
    def create_ybus_matrix(self):
        n = len(self.tl_details_conc)
        ybus = np.zeros((5,5), dtype=complex)
        for r in self.tl_details_conc.itertuples():
            i = int(r.From.split('-')[-1])
            j = int(r.To.split('-')[-1])
            yij = r.Y_complex
            # print(f'{i}\t{j}\t{yij}')
            # print(-yij)
            ybus[i,j] -= yij
            ybus[j,i] -= yij
            ybus[i,i] += yij
            ybus[j,j] += yij
        
        # print(f'ybus shape {ybus.shape}\n\nybus:\n\n {ybus}\n\ncompressed ybus:\n\n{csc_matrix(ybus)}')
        self.ybus = ybus
        
        pass
    def output_vector(self):
        '''
        Called in main. Creating the output vector, containing the voltage and the
        angle vectors for each unknown bus.
        
        
        '''
        
        y = sp.symbols(f"theta1:{len(self.data)+1}")
        theta_vec = sp.Matrix(y)
        print(self.tl_details_conc)
    
    def rem (self):
        
        # for key, value in self.conn_map.items():
        #     print(f"{key}\t{value}")
        ybus_symbols = []
        for index, row in self.tl_details_conc.iterrows():
            if index + 1 <= len(self.conn_map):
                # print(self.conn_map[str(index+1)])
                for to in self.conn_map[str(index+1)]:
                    
                    ybus_symbols.append(f"Y_{index+1}{to}")
                    
                # print(row['From'].split('-')[-1])
                # print(f"{self.conn_map[str(index+1)]}\t{row['From'].split('-')[-1]}")
                if row['From'].split('-')[-1] in self.conn_map[str(index+1)]:
                    print(f"{self.conn_map[str(index+1)]}\t{row['From']}")
                # self.conn_map[str(index+1)]
        print(ybus_symbols)
nr = NRMethod()
nr.calc_admittances()
nr.create_ybus_matrix()
# nr.output_vector()
# nr.rem()
