#!/usr/bin/env python
# coding: utf-8

from PreprocessTrainingData import read_txt
from ExportTrainingDataWithEntities import list_as_txt
import os

cwd = os.getcwd()

abstracts_preprocessed = read_txt('abstracts_preprocessed')
abstracts_annotated_preprocessed = read_txt('abstracts_annotated_preprocessed')

titles_preprocessed = read_txt('titles_preprocessed')
titles_annotated_preprocessed = read_txt('titles_annotated_preprocessed')

concat_func = lambda x,y: x + " " + str(y)

simulation_IA = list(map(concat_func,titles_preprocessed,abstracts_preprocessed))
simulation_IB = list(map(concat_func,titles_annotated_preprocessed,abstracts_annotated_preprocessed))
simulation_IIA = abstracts_preprocessed
simulation_IIB = abstracts_annotated_preprocessed

list_as_txt(simulation_IA, cwd + '/simulations/simulation_IA_preprocessed.txt')
list_as_txt(simulation_IB, cwd + '/simulations/simulation_IB_preprocessed.txt')
list_as_txt(simulation_IIA, cwd + '/simulations/simulation_IIA_preprocessed.txt')
list_as_txt(simulation_IIB, cwd + '/simulations/simulation_IIB_preprocessed.txt')
