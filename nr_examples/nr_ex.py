# %%
# Import modules
import math
import cmath
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from pprint import pprint as pp
# %%
# Tabulate data:

def tabulate_results (data):
    # prints the data in a nice way.
    print(tabulate(data, headers='keys', tablefmt='grid'))
# %%

# Define the problem:
# The problem is defined and the flat start values are set. Can you identify the flat start values?
System_data = pd.DataFrame({'bus type':['Slack Bus','Load Bus', 'Gen Bus'],
                   'bus index':[0, 1, 2],
                   'del':[0,0,0],
                   'Voltage':[1.04, 1, 1.04],
                   'P Demand':[0, 0, 1.5],
                   'Q Demand':[0, 0, 0],
                   'P Gen':[0, 0.5, 0],
                   'Q Gen':[0, 1.0, 0]})

# System_data = pd.DataFrame({'bus type':['Slack Bus','Load Bus','Load Bus', 'Gen Bus'],
#                    'bus index':[0, 1, 2, 3],
#                    'del':[0,0,0, 0],
#                    'Voltage':[1.04, 0, 1.04, 2],
#                    'P Demand':[0, 0, 1.5, 0],
#                    'Q Demand':[0, 0, 0, 1],
#                    'P Gen':[0, 0.5, 0, 1],
#                    'Q Gen':[0, 1.0, 0, 0]})


impedance_data = pd.DataFrame({'From':['Slack bus', 'Slack bus', 'Load. bus'],
                               'To':['Load bus','Gen bus', 'Gen. bus'],
                               'impedance':[0.02+0.08j, 0.02+0.08j, 0.02+0.08j],
                               'index':[(0,1),(0,2),(1,2)]})

print(f"{'='*30} ALL VALUES ARE GIVEN IN P.U. {'='*30}")
print('\nProblem Data:')
tabulate_results(data=System_data)

print('\nImpedance Data:')
tabulate_results(data=impedance_data)

# %%
# Convert rectangular to polar

# Polar form is easy to read and calculate (personal opinion). However, Python does not like Polar;
# in fact, python does not have a polar structure format. 

# That being said, if you are doing your calculations in Polar,
# then this is a good way to ensure you're on the right track.

def to_polar (y):
    r, theta = cmath.polar(y)
    return pd.Series({'Y_mag (Polar)':r, 'Y_ang (Polar)':math.degrees(theta)})
    

# %%
# Calculate the admittance from the impedance: 1/z

# There are two "pythonic" ways to calculate the admittance (There are way more than that!).

# METHOD 1: The admittance can be calculated using a function, such as, calculate_admittance ()
def calculate_admittance (z:complex):
    '''
    Takes a complex X value in a rectangular form, calculates the admittance, extract the real and imag parts, and put them in a dataframe.
    
    The real and imaginary breakdown is simply because I need to round down my sig figs.
    
    You may notice a difference in the rectangular values. This is simply because
    python works with Radian angles in any given calculations. We, in the class, use
    degrees almost always.
    '''
    y = 1/z
    y_re = round(y.real, 3)
    y_im = round(y.imag, 3)
    return pd.Series({'Admittance (Rect.)':complex(y_re, y_im)})

# %%
# Calculate the admittance:
Y_rec = impedance_data['impedance'].apply(calculate_admittance)

# Appending the new admittance data to the dataframe:
impedance_data = pd.concat([impedance_data,Y_rec], axis=1)

# OPTIONAL: Convert the admittance from Rectangular to Polar
    # Please read the header in the to_polar function to understand why it is not neccessary.
Y_polar = impedance_data['Admittance (Rect.)'].apply(to_polar)

impedance_data = pd.concat([impedance_data,Y_polar], axis=1)

tabulate_results(data=impedance_data)
# %%
def build_ybus_rec (impedance_data, ybus_rect):
    for _, row in impedance_data.iterrows():
        i, j = row['index']
        z = row['impedance']
        y = 1/z
        ybus_rect[i,j] = -1*y
        ybus_rect[j,i] = -1*y
        ybus_rect[i,i] += y
        ybus_rect[j,j] += y
    return ybus_rect

# %%
def build_ybus_polar (ybus_rect):
    # Create a copy so we don't overwrite the rect Y-bus matrix
    ybus_polar = ybus_rect.copy()
    for i, j in np.ndenumerate(ybus_rect):
        y_polar = cmath.polar(j)
        y_re = y_polar[0]
        y_im = math.degrees(y_polar[1])
        ybus_polar[i[0],i[1]] = complex(y_re,y_im)
        ybus_polar[i[0],i[1]] = complex(y_re,y_im)
        ybus_polar[i[0],i[1]] = complex(y_re,y_im)
        ybus_polar[i[0],i[1]] = complex(y_re,y_im)
    return ybus_polar


# %%
# Build the y-bus matrix:

n = 3

ybus = np.zeros(shape=(n,n), dtype=np.complex128)

ybus_rect = build_ybus_rec (impedance_data=impedance_data, ybus_rect=ybus)

ybus_polar = build_ybus_polar (ybus_rect=ybus_rect)

# %%
# Display the Y-bus matrix:
bus_labels = ['Slack Bus', 'Load Bus', 'Gen. Bus']
ybus_df = pd.DataFrame(ybus_rect, index=bus_labels, columns=bus_labels)


# %%
# Create the mismatch matrix:

# The mismatch matrix is a "repeated step" in NR. Within the mismatch matrix, We have two vectors:

    # delta P
    # delta Q

# We learned in class that the elements of these vectors are the known P and Q values:

# Bus       | Known Params   | Unknown Params | Include it?
# Slack bus | delta and |V|  | P, Q           | No, we don't P and Q
# Load Bus  | P, Q           | |V|, del       | include both P and Q
# Gen Bus   | P, |V|         | Q, del         | only include P since it is a known param 

# Q) How many parameters should we include?

def get_bus_indices (system_data):
    '''
    This is a helper function. We get the indices for each bus type. This helps us
    avoid parameters that are not part of the Jacobian setup.
    
    For example, 
        1- We do not want to include the slack bus parameters.
        2- We do not want to include the Q values for the PV buses.
    
    To get a visualiation of this, print each variable to see how it works.
        print(pq_bus)
    
    you'll notice that the boolean values are true for the buses we want to include.
    '''
    pq_bus = system_data['bus type'].eq('Load Bus')
    pv_bus = system_data['bus type'].eq('Gen Bus')
    slack_bus = system_data['bus type'].eq('Slack Bus')
    
    return pq_bus, pv_bus, slack_bus

def claculate_specified_values (system_data):
    
    pq_bus, pv_bus, slack_bus = get_bus_indices(system_data=system_data)
    
    system_data['P_spec'] = np.where(
        pq_bus | pv_bus, system_data['P Gen'] - system_data['P Demand'],
        np.nan
    )
    
    # Calculate the specified values (reactive power) for only pq buses:
    system_data['Q_spec'] = np.where(
        pq_bus, system_data['Q Gen'] - system_data['Q Demand'],
        np.nan
    )
    
    # Extract the specified values:
    deltap2_vec = system_data.loc[pq_bus, 'P_spec'].to_numpy()
    deltap3_vec = system_data.loc[pv_bus, 'P_spec'].to_numpy()
    deltaq2_vec = system_data.loc[pq_bus, 'Q_spec'].to_numpy()
    
    return deltap2_vec, deltap3_vec, deltaq2_vec

def calculate_pq_bus_calc_values (v_mag, v_ang):
    '''
    The specified values are the know P and Q values. We can calculate them using the power flow
    equations. A helpful tip to know here is that some of these values do not change. For example,
        - The Y-bus (mag and ang) values remain the same.
        - The load bus is bus # 2 (index 1).
    
    NOTE: The q_calc equation index is i+1. This is mainly because we want to position the Q values in lower half
    of the mismatch matrix. 
    '''
    load_bus_index = 1                                                  # This will not change throught the iterations.
    
    p_calc = np.zeros(shape=(n,1), dtype=np.complex128)                 # Creating a P vector so we can easily merge them later.
    q_calc = np.zeros(shape=(n,1), dtype=np.complex128)                 # Creating a P vector so we can easily merge them later.

    i = load_bus_index
    for j in range(n):
        y_mag = np.abs(ybus[i,j])
        theta = np.angle(ybus[i,j])
        p_calc[i-1] += v_mag[i] * (y_mag * v_mag[j] * np.cos(theta + v_ang[j] - v_ang[i]))
        q_calc[i+1] -= v_mag[i] * (y_mag * v_mag[j] * np.sin(theta + v_ang[j] - v_ang[i]))
    
    return p_calc, q_calc

def calculate_pv_bus_calc_values (v_mag, v_ang):
    '''
    For a pv bus, we only need to calculate the P values. We omit the Q values.
    
    NOTE: The p_calc equation index is i-1. This is mainly because we want to position the Ppv values in the middle
    of the mismatch matrix. 
    '''
    pq_bus_index = 2
    p_calc = np.zeros(shape=(n,1), dtype=np.complex128)                 # Creating a P vector so we can easily merge them later.
    
    i = pq_bus_index
    for j in range(n):
        y_mag = np.abs(ybus[i,j])
        theta = np.angle(ybus[i,j])
        p_calc[i-1] += v_mag[i] * (y_mag * v_mag[j] * np.cos(theta + v_ang[j] - v_ang[i]))
    
    return p_calc

def calculate_mismatch_matrix (system_data, ybus):
    '''
    As we have seen in class, the general form of the mismatch matrix is:
    
    |Pi, specified - Pi, calculated|
    |Qi, specified - Qi, calculated|
    
    Therefore:
    
    1- Let's identify the indices of the buses:
        - To do that, I created a helped function called get_bus_indices ()
        
    2- Let's calculate the specified values:
        - To do that, I created a helper function called claculate_specified_values ()
        
    3- Let's calculate the P_calculated and Q_calculated values:
        A) For the PQ bus, I created a helper function called calculate_pq_bus_calc_values ()
        B) For the PV bus, I created a helper function called calculate_pv_bus_calc_values ()
    
    4- Finally, we build the mismatch matrix.
    '''
    v_mag = system_data['Voltage'].to_numpy().astype(float)             # This is a vector of all V mag 
    v_ang = np.deg2rad(system_data['del'].to_numpy().astype(float))     # Always convert angles to radians when using Python.
    
    p2_spec, p3_spec, q2_spec = claculate_specified_values (system_data=system_data)
    p_pq_calc, q_pq_calc  = calculate_pq_bus_calc_values(v_mag, v_ang)
    p_pv_calc = calculate_pv_bus_calc_values(v_mag, v_ang)
    
    specified_vals = np.concatenate([p2_spec, p3_spec, q2_spec]).reshape(-1,1)
    calculated_vals = p_pq_calc + p_pv_calc + q_pq_calc
    
    mismatch_mat = specified_vals - calculated_vals
    
    return mismatch_mat
  


mismatch_mat = calculate_mismatch_matrix(system_data=System_data, ybus=ybus)
