#!/bin/bash

grep loss | grep -v "epoch\|test" | sed -e "s/\s\+/ /g" | cut -d" " -f5 | tr '\n' ' ' ; echo 
