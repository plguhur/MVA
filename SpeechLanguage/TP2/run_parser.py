#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Algorithms for Speech and NLP TD 4 -- Pirashanth Ratnamogan
Fonction qui lance tout le parser
"""
import sys
from Utils import readfile,writefile
from ProbabilisticCYK import ProbabilisticCYK
import os

#Gestion un peu brouillonne des arguments
lines_training = readfile(sys.argv[1])
lines_test = readfile(sys.argv[2])

output_path = './output.txt'



if len(sys.argv)>3:
    use_spell_correct = bool(sys.argv[3])
else:
    use_spell_correct = False
    
if len(sys.argv)>4:
    java_path = sys.argv[4]
    os.environ['JAVAHOME'] = java_path

if len(sys.argv)>5:
    jar = sys.argv[5]
    model = sys.argv[6]
    
else:
    jar = None
    model = None


parser = ProbabilisticCYK(jar,model,use_spell_correct)
parser.fit(lines_training)

predicted_parsing = []

for line in lines_test:
    predicted_parsing.append(parser.parse_line(line))
        
if output_path==None: 
    print('**********List of normalized tweets**********')
    lines = '\n'.join(predicted_parsing)
    print(lines)
else:
    lines = '\n'.join(predicted_parsing)
    print(lines)
    writefile(output_path,predicted_parsing)



