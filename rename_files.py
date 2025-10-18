#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 12:55:34 2025

@author: axel
"""
import os 
import glob
input_dir='/home/axel/ax_python_lib/thickCDR/extractDir/912X672/2c/'

files=glob.glob(input_dir+"*.nc")
for file in files: 
    basename=os.path.basename(file)
#    print(basename)
    p=basename.split(".")
#    print(p[0])
    q=p[0].split("_")
    p[0]=q[1]
    new_name=input_dir+".".join(p)
    print(new_name)    
    try:
    # Rename the file
        os.rename(file,new_name) 
        print(f"File '{file} successfully renamed to '{new_name}'.")
    except FileNotFoundError:
        print(f"Error: File '{file}' not found.")
    except PermissionError:
        print(f"Error: Permission denied. Unable to rename '{file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")