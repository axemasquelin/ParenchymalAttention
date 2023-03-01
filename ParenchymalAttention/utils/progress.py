# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
import time
import sys
import os
# ---------------------------------------------------------------------------- #

class ProgressBar():
    def __init__(self, model: str, method: str, maxfold: int, maxrep: int, maskratio:float,bar_length: 50, chr = '=') -> None:
        self.batch_id = 1
        self.modelname = model
        self.methodname = method
        self.maxfold = maxfold
        self.maxrep = maxrep
        self.chr = chr
        self.maskratio = maskratio
        self.barlength = bar_length
        
    def _update(self, rep:int, fold:int):
        self.rep = rep
        self.fold = fold

    def visual(self, epoch:int, batch_id:int, max_batches:int, max_epoch:int):
        """
        Definition:
        Inputs: 1) value        - int showing the number of repetition left
                2) endvalue     - int maximum number of repetition
                3) bar_length   - int shows how long the bar is
                4) chr          - str character to fill bar
        """
        
        percent = float(epoch) / max_epoch
        arrow = self.chr * int(round(percent * self.barlength)-1) + '>'
        spaces = ' ' * (self.barlength - len(arrow))
        sys.stdout.write("\r Maskratio: {0}x | Fold {1}/{2} | Rep: {3}/{4} | Batch: {5}/{6} [{7}] {8}%".format(
                                                                                self.maskratio,
                                                                                self.fold+1, 
                                                                                self.maxfold,
                                                                                self.rep+1,
                                                                                self.maxrep,
                                                                                batch_id+1,
                                                                                max_batches,
                                                                                arrow + spaces,
                                                                                int(round(percent*100))))
        
        

