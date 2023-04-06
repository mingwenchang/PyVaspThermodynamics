#!/usr/bin/env python3
# coding=UTF-8
"""
Name: doscar_IO (ver 2.0)
Created on Fri Sep 18 10:57:09 2019
Developer: Ming-Wen Chang
E-mail: ming.wen.c@gmail.com
"""
#import re
#import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import simps
from scipy.interpolate import make_interp_spline
from collections import OrderedDict

TDOS_channels =  OrderedDict([
                    (2, ['energy', 'tdos']),
                    (3, ['energy', 'tdos_up', 'tdos_down'])
                    ])      


PDOS_channels = OrderedDict([
                (4, ['energy','s', 'p', 'd']),  
                 
                (5, ['energy','s', 'p', 'd', 'f']),  
                
                (7, ['energy','s_up', 's_down', 'p_up', 'p_down', 'd_up', 'd_down']),
                
                (9, ['energy','s_up', 's_down', 'p_up', 'p_down', 'd_up', 'd_down', 'f_up', 'f_down']),
                 
                (10, ['energy','s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z^2', 'd_xz', 'd_x^2-y^2']),  
                 
                (17, ['energy','s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z^2', 'd_xz', 'd_x^2-y^2', 
                     'f_y(3x^2-y^2)', 'f_xyz', 'f_yz^2', 'f_z^3', 'f_xz^2', 'f_z(x^2-y^2)', 'f_x(x^2-3y^2)']),
                
                (19, ['energy','s_up', 's_down', 'p_y_up', 'p_y_down', 'p_z_up', 'p_z_down', 'p_x_up', 'p_x_down', 
                    'd_xy_up', 'd_xy_down', 'd_yz_up', 'd_yz_down', 'd_z^2_up', 'd_z^2_down', 'd_xz_up', 
                    'd_xz_down', 'd_x^2-y^2_up', 'd_x^2-y^2_down']), 
                
                (33, ['energy','s_up', 's_down', 'p_y_up', 'p_y_down', 'p_z_up', 'p_z_down', 'p_x_up', 'p_x_down',
                    'd_xy_up', 'd_xy_down', 'd_yz_up', 'd_yz_down', 'd_z^2_up', 'd_z^2_down', 'd_xz_up',
                    'd_xz_down', 'd_x^2-y^2_up', 'd_x^2-y^2_down', 
                    'f_y(3x^2-y^2)_up', 'f_y(3x^2-y^2)_down', 'f_xyz_up', 'f_xyz_down', 'f_yz^2_up', 'f_yz^2_down', 
                    'f_z^3_up', 'f_z^3_down', 'f_xz^2_up', 'f_xz^2_down', 'f_z(x^2-y^2)_up', 'f_z(x^2-y^2)_down', 
                    'f_x(x^2-3y^2)_up', 'f_x(x^2-3y^2)_down'])
                ])   

lc  = '#212121' 
rc1 = {'axes.axisbelow': True,
       'axes.edgecolor': '#212121',
       'axes.facecolor': 'white',
       'axes.grid': False,
       'axes.labelcolor': '#212121',
       'axes.linewidth': 1.25,
       'figure.facecolor': 'white',
       'font.family': ['sans-serif'],
       'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
       'grid.color': '.8',
       'grid.linestyle': '-',
       'image.cmap': 'rocket',
       'legend.frameon': True,
       'legend.numpoints': 1,
       'legend.scatterpoints': 1,
       'lines.solid_capstyle': 'round',
       'text.color': '#212121',
       'xtick.color': '#212121',
       'xtick.direction': 'out',
       'xtick.major.size': 3.5,
       'xtick.minor.size': 3.5,
       'ytick.color': '#212121',
       'ytick.direction': 'out',
       'ytick.major.size': 3.5,
       'ytick.minor.size': 3.5}


#light blue:, Cerise, light green, orange, yellow, darkblue
#code = ['#02ACEF', '#DB1E83' , '#A8D808', '#FE9701', '#FEE303', '#023DFD']
         
        
# up to 24 colors stored   
try:
    import seaborn as sns
    sns.set_style(rc=rc1)
    #sns.set_context("paper")
    #sns.axes_style()
    clist = sns.color_palette("colorblind", 24)
except:
    clist =  ['blue', 'red', 'gold', 'salmon', 'lightcoral', 'lightskyblue', 'darkgreen', 'black',
             'orange','powderblue','olivedrab', 'burlywood',  'indianred', 
            'steelblue', 'lawngreen', 'y', 'hotpink', 'slategrey', 'yellowgreen','palegreen', 
            'sandybrown', 'tomato', 'darkviolet', 'lightgreen', 'tan','maroon']

#cdict = OrderedDict([
#       ('s', 'slategrey'), ('p', 'salmon'), ('d', 'gold'), ('f','lightgreen'), ('tdos', 'powderblue')
#        ])


class Doscar:
    
    def __init__(self, filename=None, efermi=None, undecomposed=True):  
        self.filename = filename
        self.undecomposed = undecomposed
        self.efermi = efermi
        self._efermi = 0.00
        self._energy = None
        self.data = None
        self.pdos = None
        self.analdbc = False
        
        if filename is not None:
            self.data = self.read_DOSCAR(self.filename)
            self._energy = self.data[0]['energy']
            
        if self.efermi is not None:
            self.set_efermi(self.efermi)
            
    def set_efermi(self, efermi):
        delta = efermi - self._efermi #shift value
        self._efermi  = efermi
        self._energy += delta 
        if self.data is not None:
            for df in self.data:
                df['energy'] = self._energy
                
        if self.pdos is not None:
            self.pdos['energy'] = self._energy
        
    def get_efermi(self):
        return self._efermi
            
    def read_DOSCAR(self, filename='DOSCAR'):   
        with open(filename, 'r' ) as txt: 
            natoms = int(txt.readline().split()[0]) #read line 1
            [txt.readline() for i in range(3)] #skip line2 to line4
            
            if 'LOBSTER' in txt.readline(): # read line5 
                lobster = True
            else:
                lobster = False
            
            data = []
            for i in range(0, natoms+1): #read line6 - end
                dos = []
                head = txt.readline().split()
                nedos = int(head[2])
                
                if not lobster:
                    efermi = float(head[3])
                else:
                    efermi = 0.00
            
                if len(head) == 5:#for vasp
                    factors = None
                else:#for lobster 
                    factors = head[7:]
    
                for j in range(nedos):
                    dos.append([float(value) for value in txt.readline().split()])
                dos = pd.DataFrame(data=dos)   
                    
                nn =  len(dos.columns)
                if i == 0:
                    if nn == 5: #spin-ploarized 
                        nn -= 2
                        self.ispin= True
                        dos = dos.iloc[:, 0:nn] #strip int_tdos_up and int_tdos_down 
                    else:#non spin ploarized 
                        nn -= 1
                        self.ispin= False
                        dos = dos.iloc[:, 0:nn] #strip int_tdos_up and int_tdos_down 
                        
                    dos.iloc[:, 0] -= efermi 
                    energy = dos.iloc[:, 0]
                    factors = TDOS_channels[nn]
                    
                else:
                    if factors is None: #vasp
                        factors = PDOS_channels[nn]
                    else: #lobster 
                        if nn - 1 > len(factors): #spin-ploarized 
                            ll = [ ]
                            for ft in factors:
                                ll.append(ft + '_up')
                                ll.append(ft + '_down')
                            factors = ll
                            factors.insert(0, 'energy')
                        else:
                            factors.insert(0, 'energy')
                    dos.iloc[:, 0] =energy
                dos.columns = factors  
                dos = self._invert_dos_values_of_spin_down(dos)
                data.append(dos)
                
            return data
                     
    def sum_atomic_pdos(self, dfobjs):
        energy = dfobjs[0]['energy']
        df = sum(dfobjs)
        if self.undecomposed :
            df = self._reduce_to_undecomposed(df)
        df['energy'] = energy
        return df
    
    def concat_atomic_pdos(self, dfobjs):
        energy = dfobjs[0]['energy']
        dfobjs = [df.iloc[:, 1:] for df in dfobjs]
        df = pd.concat(dfobjs, axis=1) 
        df.insert(loc=0, column='energy', value=energy)
        return df
                    
    def to_tdos(self, df=None, columns=None):
        if df is None:
            df = self.pdos
        if columns is None:
            if self.ispin:
                columns=['energy', 'tdos_up', 'tdos_down']
            else:
                columns=['energy', 'tdos']
         
        df2 = pd.DataFrame(data=0.00, index=range(len(df)), columns=columns)
        df2.iloc[:, 0] = df['energy']
        for idx in range(1, len(df.columns[1:]) + 1):
            if self.ispin:
                if idx%2 !=0:#spin up
                    df2.iloc[:, 1] += df.iloc[:,idx]
                else:#spin down
                    df2.iloc[:, 2] += df.iloc[:,idx]
            else:
                df2.iloc[:, 1] += df.iloc[:,idx]

        return df2
    
    def select_orbital(self, df, orb='d'):
        
        if len(orb) == 1: #i.e. s, p, d or f     
            #pattern = r'.*-?\d?%s_.*' %(orb)    
            pattern = r'.*-?\d?%s_.*' %(orb)    
            ids = [0]+ [idx for idx, ft in enumerate(df.columns) if re.match(pattern, ft)]
            df = df.iloc[:,ids]
        else:
            #pattern = re.sub('(_up|_down)','', orb) #strip "_up" and "_down" words
            pattern = '.*-?\d?' + re.sub('\^','\^', orb)
            ids = [0]+ [idx for idx, ft in enumerate(df.columns) if re.match(pattern, ft)]
            df = df.iloc[:,ids]
            pass
            
        return df  
    
    def get_pdos(self, kwargs, data=None):
        if data is None:
            data = self.data
            
        for idx, key in enumerate(kwargs.keys()):
            atom = key
            orb =  kwargs[key][0] 
            #sum the pdos of selected atoms 
            df = self.sum_atomic_pdos([data[idx] for idx in kwargs[key][1]]) 
            #Select target orbital and re-lable columns
            df = self.select_orbital(df, orb)
            df.columns = ['energy'] + [atom + '-' + ft for ft in df.columns[1:]]
            if idx == 0:
                df2 = df
            else:
                df2 = self.concat_atomic_pdos([df2, df])   
        self.pdos = df2
        return df2             
                
    def calculate_dbc(self, df=None, orbital='d', erange=None):
        if df is None:
            df = self.pdos
            
        df = self.select_orbital(df, orbital)
        epsilon = df['energy']  
        
        if erange is None:
            #erange = (epsilon.iloc[0], self._efermi+2.00)
            erange = (epsilon.iloc[0], epsilon.iloc[-1])
            
        if self.ispin: #spin polarize 
            rho1 = df.iloc[:,1].values
            rho2 = df.iloc[:,2].values
        else:#non-spin polarize 
            rho1 = df.iloc[:,1].values
            rho2 = df.iloc[:,1].values
          
        #Set intergration range for the calculation of nunber of electrons in d-band(ned)  
        mask = (epsilon >= erange[0]) & (epsilon <= self._efermi) 
        x = epsilon[mask] #integrating the DOS to the Fermil level
        y1 = rho1[mask] #density up
        y2 = rho2[mask] #density down
        ned_up = abs(simps(y1, x))  #d-band filling by up electrons 
        ned_down = abs(simps(y2, x)) #d-band filling by down electrons
        if self.ispin:
            self.ned = ned_up + ned_down
        else:
            self.ned = ned_up 
                                    
        #Set intergration range for the calculation of d-band center and width
        mask = (epsilon >= erange[0]) & (epsilon <= erange[1])
        x = epsilon[mask] #integrating the DOS to the Fermil level + n_eV
        y1 = rho1[mask] #density up
        y2 = rho2[mask] #density down
        dbc_up = simps(y1*x, x) / simps(y1, x)  #d-band center up 
        dbc_down = simps(y2*x, x) / simps(y2, x) #d-band center down 
        dbw_up =   np.sqrt(simps(y1*x**2, x) / simps(y1, x))  #d-band width up 
        dbw_down = np.sqrt(simps(y2*x**2, x) / simps(y2, x))  #d-band width down 
        self.dbc =  dbc_up, dbc_down
        self.dbw = dbw_up,  dbw_down
        
        self.analdbc = True   
        return self.dbc     
    
    def split_doscar(self):
        for idx, df in enumerate(self.data):
            if self.undecomposed:
                df = self._reduce_to_undecomposed(df)
            filename='dos' + str(idx)
            self.save_df_to_txt(df, filename)
            
    def save_df_to_txt(self, df, filename='df.txt'):
        #df = self._invert_dos_values_of_spin_down(df)
        with open(filename, 'w') as txt:
            txt.write(df.to_string(index=False))
            
    def save_dos(self, df=None, filename='pdos'):
        if df is None:
            try:
                df = self.pdos
                self.save_df_to_txt(df, 'pdos')
            except:
                df = self.data[0]
                self.save_df_to_txt(df, 'tdos')
        else:
            self.save_df_to_txt(df, filename)
        
    def _reduce_to_undecomposed(self, df):
        factors = [ ]
        for ft1 in df.columns[1:]:
            fft1 = ft1.split('_')[0] + '_' + ft1.split('_')[-1] 
            if fft1 not in  factors:
                factors.append(fft1)
        factors.insert(0, 'energy')
                
        df2 = pd.DataFrame(data=0.00, index=range(len(df)), columns=factors)
        for ft2 in df2.columns[1:]:
            for ft1 in df.columns[1:]:
                name = ft1.split('_')[0] + '_' + ft1.split('_')[-1]
                if ft2 == name:
                    df2[ft2] += df[ft1]
        df2['energy'] = df['energy']
        return df2
    
    def _invert_dos_values_of_spin_down(self, df):
        for ft in df.columns:
            if 'down' in ft:
                df[ft] *= -1
        return df
    
    def spline_df(self, df):
        lb = df.columns[0]
        df2 = pd.DataFrame()
        for ft in df.columns[1:]:
             xnew, ynew  = smooth_line(df.iloc[:,0], df[ft])
             df2[lb] = xnew
             df2[ft] = ynew
        return df2

    def plot_pdos(self, df=None, filename='pdos.png', erange=None, dosrange=None,smooth=True,
                  size=(1.5, 6), dpi=150, cmap=None, line_width=1.00, line_alpha=0.50, 
                  fill=True, fill_alpha=0.50, fontsize=10, nxticks=10, nyticks=5, stack='column', neat=False):
        
        if df is None:
            df = self.pdos
            
        if cmap is None:
            cmap = clist
            
        if smooth:
            df = self.spline_df(df)
        
        if erange is None:
            emin, emax  = (self._efermi-6.00, self._efermi+4.00)
        else:
            emin, emax = erange[0], erange[1] 
            
            
        #df = self._invert_dos_values_of_spin_down(df)
        mask = ( df.iloc[:,0].values  >= emin) & ( df.iloc[:,0].values  <= emax)  
        evalues =  df.iloc[:,0].values[mask] 
        dmax = 0.00

        coloridx = 0 
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        for ft in df.columns[1:]:
 
            #Color setting
            if self.undecomposed:
                label = ft.split('_')[0]
            else:
                label = ft.rstrip('_up').rstrip('_down')
            """
            if type(cmap) is dict: 
                try:
                    cc = cmap[label]
                except:
                    orb = re.sub("\d", "", label.split('-')[-1])
                    cc = cmap[orb] 
            """
            
            #Values for plotting
            cc = cmap[coloridx]
            dosvalues = df[ft].values[mask] 
            
            if stack.lower()[0] == 'c':
                x , y = dosvalues, evalues
                if 'down' not in ft:  
                    ax.plot(x, y, linewidth=line_width, alpha=line_alpha, label=label, color=cc)
                    coloridx -= 1
                else:
                    ax.plot(x, y, linewidth=line_width, alpha=line_alpha, label='', color=cc)
                    
                if fill:
                    ax.fill_betweenx(y, x, -0.000,  where=y <= 0, interpolate=True, color=cc, alpha=fill_alpha) 
                    ax.fill_betweenx(y, x, +0.000,  where=y >= 0, interpolate=True, color=cc, alpha=fill_alpha*0.1)
                    
            else:
                x , y = evalues, dosvalues
                if 'down' not in ft:  
                    ax.plot(x, y, linewidth=line_width, alpha=line_alpha, label=label, color=cc)
                    coloridx -= 1
                else:
                    ax.plot(x, y, linewidth=line_width, alpha=line_alpha, label='', color=cc)
                    
                if fill: 
                    ax.fill_between(x, y, -0.000,  where=x <= 0, interpolate=True, color=cc, alpha=fill_alpha) 
                    ax.fill_between(x, y, +0.000,  where=x >= 0, interpolate=True, color=cc, alpha=fill_alpha*0.1) 
           
            coloridx += 1 
            if np.max(np.abs(dosvalues)) > dmax:
                dmax = np.max(np.abs(dosvalues))

        if dosrange is None:
            if self.ispin:
                dmin, dmax = -1.10*dmax, 1.10*dmax
            else:
                dmin, dmax = -0.00, 1.10*dmax
        else:
            dmin, dmax = dosrange[0], dosrange[1]
        
        
        if stack.lower()[0] == 'c':
            ax.set_ylim([emin, emax])
            ax.set_xlim([dmin, dmax])
            ax.set_ylabel( '$E - E_{f}\ (eV)$', size=fontsize)
            ax.set_xlabel( 'DOS (electrons/eV)', size=fontsize)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nxticks))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks))
            #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(which='major', labelleft= True, left=True, labelbottom=True,  bottom=True, direction ='out', labelsize=fontsize) 
            ax.axvline(x=0, linewidth=1.00, linestyle='-', color='#393E46', alpha=0.90)
            ax.axhline(y=self._efermi, linewidth=1.50, linestyle='-', color='#393E46', alpha=0.90)
            if self.analdbc:
                ax.axhline(y=max(self.dbc), xmin=0.00, xmax=1.00, linewidth=1.50, linestyle='--', color='#393E46', alpha=0.50)
        else:
            ax.set_xlim([emin, emax])
            ax.set_ylim([dmin, dmax])
            ax.set_xlabel( '$E - E_{f}\ (eV)$', size=fontsize)
            ax.set_ylabel( 'DOS (electrons/eV)', size=fontsize)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nxticks))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks))
            #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(which='major', labelleft=True, left=True, labelbottom=True, bottom=True, direction ='in', labelsize=fontsize)
            ax.axvline(x=self._efermi, linewidth=1.50, linestyle='-', color='#393E46', alpha=1.00)
            ax.axhline(y=0, linewidth=1.00, linestyle='-', color='#393E46', alpha=0.90)
            if self.analdbc:
                ax.axvline(x=max(self.dbc), ymin=0.00, ymax=1.00, linewidth=1.50, linestyle='--', color='#393E46', alpha=0.50) #
                
        legend =ax.legend(bbox_to_anchor=(1.00, 1.02), loc= 'upper left', prop={'size':fontsize*0.8}, frameon=True) 
        [i.set_linewidth(3) for i in legend.legendHandles]
        
        if neat:
            ax.set_xlabel( '', size=fontsize)
            ax.set_ylabel( '', size=fontsize)
            ax.tick_params(which='major', labelleft= False, left=False, labelbottom=False, bottom=False, direction ='in', labelsize=fontsize)
            ax.legend().remove()

        fig.savefig(filename, bbox_inches="tight")
                         
def smooth_line(x, y, ngrids=None):
    if ngrids is None:
        ngrids = len(x) * 100
    xnew = np.linspace(x.min(), x.max(), ngrids) 
    bsplobj = make_interp_spline(x, y, k=3) #BSpline object
    ynew = bsplobj(xnew) #smoothed data
    return xnew, ynew    


def spline_df(df, ngrids=None):
    if ngrids is None:
        ngrids = 10 * len(df)
    lb = df.columns[0]
    df2 = pd.DataFrame()
    for ft in df.columns[1:]:
         xnew, ynew  = smooth_line(df.iloc[:,0], df[ft], ngrids)
         df2[lb] = xnew
         df2[ft] = ynew
    return df2

debug = False
if debug:
    atoms = list(range(1, 8+1))
    kwargs = {'Pt':('d_x^2-y^2', atoms)}
    a = Doscar('../DOSCAR.lobster', undecomposed=False)
    a.get_pdos(kwargs)
    a.pdos
    a.calculate_dbc(orbital='d_x^2-y^2')
    a.plot_pdos(erange=(-10,2), stack='row', size=(6, 1.5))
    a.save_dos()
    print(a.dbc)
    print(a.nelectrons_in_pdos)

 
        
        
 

    

