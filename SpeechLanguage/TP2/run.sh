#!/bin/bash
while read line; do
  python system/parser.py -s "$line" -d system/data/sequoia-corpus+fct.mrg_strict
done < /dev/stdin
