# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:42:00 2022

@author: jante
"""


class pvdwriter():
    def __init__(self, pvdfile):
        self.f = open(pvdfile, 'w')

        self.f.write('<?xml version="1.0"?>\n')
        self.f.write('<VTKFile type="Collection" version="0.1"\n')
        self.f.write('byte_order="LittleEndian"\n')
        self.f.write('compressor="vtkZLibDataCompressor">\n')
        self.f.write('<Collection>\n')

    def append(self, filename, timestamp):
        self.f.write(f'<DataSet timestep="{timestamp}"\n')
        self.f.write(f'file="{filename}"/>\n\n')

    def close(self):
        self.f.write('</Collection>\n')
        self.f.write('</VTKFile>\n')
        self.f.close()
