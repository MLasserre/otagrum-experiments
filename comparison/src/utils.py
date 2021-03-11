#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import openturns as ot
import otagrum as otagr
import numpy as np
import pyAgrum as gum
import os
import os.path as path
import graph_utils as gu


def structure_prospecting(structures, index):
    for s in structures[index]:
        print(s.dag())

def structural_score(true_structure, list_structures, score):
    # print(type(true_structure),flush=True)
    # print(type(list_structures[0]),flush=True)
    ref_dag = true_structure.getDAG()
    ref_names = [name for name in true_structure.getDescription()]
    result = []
    for l in list_structures:
        list_result = []
        for s in l: 
            #bn = named_dag_to_bn(s, Tstruct.names())
            test_dag = s.getDAG()
            test_names = [name for name in s.getDescription()]
            sc = gum.StructuralComparator()
            sc.compare(ref_dag, ref_names, test_dag, test_names)
            if score == 'skelP':
                scores = sc.precision_skeleton()
            elif score == 'skelR':
                scores = sc.recall_skeleton()
            elif score == 'skelF':
                scores = sc.f_score_skeleton()
            elif score == 'dagP':
                scores = sc.precision()
            elif score == 'dagR':
                scores = sc.recall()
            elif score == 'dagF':
                scores = sc.f_score()
            elif score == 'hamming':
                ref_cpdag = gu.dag_to_cpdag(ref_dag)
                test_cpdag = gu.dag_to_cpdag(test_dag)
                sc.compare(ref_cpdag, ref_names, test_cpdag, test_names)
                scores = sc.shd()
            else:
                print("Wrong entry for argument!")
            
            list_result.append(scores)
        
        result.append(list_result)
    return result

def struct_from_multiple_dataset(directory, method, parameters,
                                 start=10, end=1e4, num=10, restart=1):
    # Looking for which size we learn
    sizes = np.linspace(start, end, num, dtype=int)
    
    # Looking for all the files in the directory
    files_in_directory = [f for f in os.listdir(directory) \
                          if path.isfile(path.join(directory, f))]
    files_in_directory.sort()
    files_in_directory = files_in_directory[:restart]
    
    list_structures = []
    for f in files_in_directory:
        print("Processing file", f)
        # Loading file f
        data = ot.Sample.ImportFromTextFile(path.join(directory, f), ',')
        
        list_by_size = []
        for size in sizes:
            print("    Learning with", size, "data...")
            sample = data[0:size]
            bn = learning(sample, method, parameters)
            list_by_size.append(bn)

        list_structures.append(list_by_size)

    # Transposing result matrix
    list_structures = np.reshape(list_structures, (len(files_in_directory), num)).transpose()
    return list_structures

