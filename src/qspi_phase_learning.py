import csv
import itertools as it
import pandas as pd
import numpy as np
import sklearn.decomposition
from tqdm import tqdm
import re
import string
import random
import matplotlib.pyplot as plt 
import scipy.integrate as integrate
import numpy.lib.stride_tricks as tri
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torch_data
from torch.autograd import Variable
import scipy.optimize
from functools import partial
from numpy import linalg as LA

device = torch.device('cuda')

#############################################################################################
# function for calculating f and g coefficients given phases
# phases: the qsp phases
def calc_qsp_coeff(phases):
  #print("phases", phases)
  degree = len(phases) - 1
  # print("degree = ", degree)
  fcoeff = torch.zeros(2 * degree + 1, dtype = torch.double)
  gcoeff = torch.zeros(2 * degree + 1, dtype = torch.double)
    
  if degree == 0:
    fcoeff[0] = torch.cos(phases[0])
    gcoeff[0] = torch.sin(phases[0])
    # fcoeff[0] = torch.cos(torch.tensor(phases[0]))
    # gcoeff[0] = torch.sin(torch.tensor(phases[0]))
  else:
    phases_reduce = phases[:-1]
    # print("phases_reduce = ", phases_reduce)
    fcoeff_reduce, gcoeff_reduce = calc_qsp_coeff(phases_reduce)
    # now compute the new coefficients from the old one
    for idx in range(-degree, degree + 1):
        # print("idx = ", idx)
        if idx >= degree - 1:
            # print(idx+degree-2)
            if idx - 1 >= 0:
                # note the index in the RHS has to be idx + degree - 2, instead of idx + degree - 1 because there is also one negative index coefficient less.
                # fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                # gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                fcoeff[idx + degree] = torch.cos(phases[-1]) * fcoeff_reduce[idx + degree - 2]
                gcoeff[idx + degree] = torch.sin(phases[-1]) * fcoeff_reduce[idx + degree - 2]
        elif idx <= -(degree - 1):
            if idx + 1 <= 0:
                # fcoeff[idx + degree] = -torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                # gcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                fcoeff[idx + degree] = -torch.sin(phases[-1]) * gcoeff_reduce[idx + degree]
                gcoeff[idx + degree] = torch.cos(phases[-1]) * gcoeff_reduce[idx + degree]
        elif abs(idx) <= degree - 2:
            # fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
            #                     - torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
            # gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
            #                     + torch.cos(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
            fcoeff[idx + degree] = torch.cos(phases[-1]) * fcoeff_reduce[idx + degree - 2] \
                                - torch.sin(phases[-1]) * gcoeff_reduce[idx + degree]
            gcoeff[idx + degree] = torch.sin(phases[-1]) * fcoeff_reduce[idx + degree - 2] \
                                + torch.cos(phases[-1]) * gcoeff_reduce[idx + degree]
        else:
            print("something wrong with indexing. check.")
  
  # for test
  # print("fcoeff", fcoeff)
  # print("gcoeff", gcoeff)
    
  return fcoeff, gcoeff


def calc_qsp_coeff_tensor(phases):
  #print("phases", phases)
  degree = len(phases) - 1
  # print("degree = ", degree)
  fcoeff = torch.zeros(2 * degree + 1, dtype = torch.double)
  gcoeff = torch.zeros(2 * degree + 1, dtype = torch.double)
    
  if degree == 0:
    # fcoeff[0] = torch.cos(phases[0])
    # gcoeff[0] = torch.sin(phases[0])
    fcoeff[0] = torch.cos(torch.tensor(phases[0]))
    gcoeff[0] = torch.sin(torch.tensor(phases[0]))
  else:
    phases_reduce = phases[:-1]
    # print("phases_reduce = ", phases_reduce)
    fcoeff_reduce, gcoeff_reduce = calc_qsp_coeff_tensor(phases_reduce)
    # now compute the new coefficients from the old one
    for idx in range(-degree, degree + 1):
        # print("idx = ", idx)
        if idx >= degree - 1:
            # print(idx + degree - 2)
            if idx - 1 >= 0:
                # note the index in the RHS has to be idx+degree-2, instead of idx+degree-1 because there is also one negative index coefficient less.
                fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                # fcoeff[idx + degree] = torch.cos(phases[-1]) * fcoeff_reduce[idx + degree - 2]
                # gcoeff[idx + degree] = torch.sin(phases[-1]) * fcoeff_reduce[idx + degree - 2]
        elif idx <= -(degree - 1):
            if idx + 1 <= 0:
                fcoeff[idx + degree] = -torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                gcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                # fcoeff[idx + degree] = -torch.sin(phases[-1]) * gcoeff_reduce[idx + degree]
                # gcoeff[idx + degree] = torch.cos(phases[-1]) * gcoeff_reduce[idx + degree]
        elif abs(idx) <= degree - 2:
            fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
                                - torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
            gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
                                + torch.cos(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
            # fcoeff[idx + degree] = torch.cos(phases[-1]) * fcoeff_reduce[idx + degree - 2] \
            #                     - torch.sin(phases[-1]) * gcoeff_reduce[idx + degree]
            # gcoeff[idx + degree] = torch.sin(phases[-1]) * fcoeff_reduce[idx + degree - 2] \
            #                     + torch.cos(phases[-1]) * gcoeff_reduce[idx + degree]
        else:
            print("something wrong with indexing. check.")
  
  # for test
  # print("fcoeff", fcoeff)
  # print("gcoeff", gcoeff)
    
  return fcoeff, gcoeff


def test_calc_qsp_coeff():

  test_thres = 1e-4
  phases = torch.tensor([torch.pi / 7, torch.pi / 8, -torch.pi / 9])
  fcoeff_correct = torch.tensor([0.1371, 0.0000, -0.0381, 0.0000, 0.7822], dtype = torch.float64)
  gcoeff_correct = torch.tensor([0.3767, 0.0000, 0.3808, -0.0000, -0.2847], dtype = torch.float64)
  fcoeff, gcoeff = calc_qsp_coeff(phases)
  if torch.mean(torch.abs(fcoeff - fcoeff_correct)) < test_thres \
    and torch.mean(torch.abs(gcoeff - gcoeff_correct)) < test_thres:
    print("passed test.")
  else:
    print("failed test.")
    
    
#############################################################################################


#############################################################################################
def mysinc(x):
    return torch.sinc(x / torch.pi)

def prob_exact_long(phases, beta, K):
    f, g = calc_qsp_coeff(phases)
    d = phases.shape[0] - 1
    prob = 0
    for n in range(-d, d + 1):
        for nq in range(-d, d + 1):
            for m in range(-d, d + 1):
                for mp in range(-d, d + 1):
                    A = (f[n + d] * f[nq + d] + g[n + d] * g[nq + d]) * (f[m + d] * f[mp + d] + g[m + d] * g[mp + d]) \
                        * torch.exp(torch.tensor(-1 / 4 * K ** 2 * (n - nq - m + mp) ** 2)) * torch.cos(torch.tensor(K * (n - m) * beta))
                    prob += A
    return prob

def prob_exact_long_coeff(f, g, beta, K):
    d = int((f.shape[0] - 1) / 2)
    prob = 0
    for n in range(-d, d + 1):
        for nq in range(-d, d + 1):
            for m in range(-d, d + 1):
                for mp in range(-d, d + 1):
                    A = (f[n + d] * f[nq + d] + g[n + d] * g[nq + d]) * (f[m + d] * f[mp + d] + g[m + d] * g[mp + d]) \
                       * torch.exp(torch.tensor(-1 / 4 * K ** 2 * (n - nq - m + mp) ** 2)) * torch.cos(torch.tensor(K * (n - m) * beta))
                    prob += A
    return prob

def prob_exact_yuanliu(phases, beta_th, K):
    # calculate the half-period of the qubit response function
    half_period = torch.pi / (2 * K)
    degree = len(phases) - 1
    
      
    # make torch tensors to store the coefficients of the QSP
    # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    
    # compute the QSP coefficients using phases
    # fcoeff, gcoeff = calc_qsp_coeff(phases).to(device)
    fcoeff, gcoeff = calc_qsp_coeff(phases)
    #print("fcoeff = ", fcoeff)
    #print("gcoeff = ", gcoeff)
    
    # compute p_err from the coefficients analytically
    p_err = 0.0
    for r in range(-2 * degree, 2 * degree + 1):
        # this is range of integration for [-pi / 2k, pi / 2k]
        # Hr = 2 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 2.0)) \
        #   - (4 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        Hr = torch.cos(torch.tensor(K * r * beta_th))
        
        # this is a restricted range of integration for [-pi/4k, pi/4k]
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi*r/4.0)) \
        #     - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K*r*beta_th))
        cr = 0.0
        for n in range(-degree, degree + 1):
            if abs(n + r) <= degree:
                for mp in range(-degree, degree + 1):
                    for nq in range(-degree, degree + 1):
                        cr += (fcoeff[n + degree] * fcoeff[nq + degree] + gcoeff[n + degree] * gcoeff[nq + degree]) \
                        * (fcoeff[n + r + degree] * fcoeff[mp + degree] + gcoeff[n + r + degree] * gcoeff[mp + degree]) \
                        * torch.exp(torch.tensor(-0.25 * K ** 2 * (mp - nq - r) ** 2))
#             else:
#                 print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
        p_err += cr * Hr
        # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return p_err

def prob_exact_qsp2_yuanliu(phases1, phases2, beta_th, K):
    # calculate the half-period of the qubit response function assuming two QSPs
    half_period = torch.pi / (2 * K)
    degree = len(phases1) - 1
    if degree != len(phases2) - 1:
        print("Two QSP sequence must have the same length.")
        exit()
    
      
    # make torch tensors to store the coefficients of the QSP
    # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    fcoeffp = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeffp = torch.zeros(2 * degree + 1, dtype = torch.float)
    
    # compute the QSP coefficients using phases
    # fcoeff, gcoeff = calc_qsp_coeff(phases).to(device)
    fcoeff, gcoeff = calc_qsp_coeff(phases1)
    fcoeffp, gcoeffp = calc_qsp_coeff(phases2)
    #print("fcoeff = ", fcoeff)
    #print("gcoeff = ", gcoeff)
    
    # compute p_err from the coefficients analytically
    p_err = 0.0
    for r in range(-2 * degree, 2 * degree + 1):
        # this is range of integration for [-pi / 2k, pi / 2k]
        # Hr = 2 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 2.0)) \
        #   - (4 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        Hr = torch.cos(torch.tensor(K * r * beta_th))
        
        # this is a restricted range of integration for [-pi / 4k, pi / 4k]
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 4.0)) \
        #     - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        cr = 0.0
        for n in range(-degree, degree + 1):
            if abs(n + r) <= degree:
                for mp in range(-degree, degree + 1):
                    for nq in range(-degree, degree + 1):
                        cr += (fcoeff[n + degree] * fcoeffp[nq + degree] + gcoeff[n + degree] * gcoeffp[nq + degree]) \
                        * (fcoeff[n + r + degree] * fcoeffp[mp + degree] + gcoeff[n + r + degree] * gcoeffp[mp + degree]) \
                        * torch.exp(torch.tensor(-0.25 * K ** 2 * (mp - nq - r) ** 2))
#             else:
#                 print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
        p_err += cr * Hr
        # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return p_err


def test_prob_exact():
    phases = torch.tensor([torch.pi / 7, torch.pi / 8, -torch.pi / 9])
    print(prob_exact_yuanliu(phases, torch.pi / 18, 1))
    print(prob_exact_long(phases, torch.pi / 18, 1))

#################################################################################################################

def loss_fn_exact(phases, beta_th, K, sigma, flag_callback = False):
    # calculate the half-period of the qubit response function
    half_period = torch.pi / (2 * K)
    degree = len(phases) - 1
    
      
    # make torch tensors to store the coefficients of the QSP
    # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    
    # compute the QSP coefficients using phases
    # fcoeff, gcoeff = calc_qsp_coeff(phases).to(device)
    fcoeff, gcoeff = calc_qsp_coeff_tensor(phases)
    #print("fcoeff = ", fcoeff)
    #print("gcoeff = ", gcoeff)
    
    # compute p_err from the coefficients analytically
    p_err = 0.0
    for r in range(-2 * degree, 2 * degree + 1):
        # this is range of integration for [-pi / 2k, pi / 2k]
        Hr = 2 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 2.0)) \
           - (4 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        
        # this is a restricted range of integration for [-pi / 4k, pi / 4k]
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi*r/4.0)) \
        #     - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K*r*beta_th))
        cr = 0.0
        for n in range(-degree, degree + 1):
            if abs(n + r) <= degree:
                for mp in range(-degree, degree + 1):
                    for nq in range(-degree, degree + 1):
                        cr += (fcoeff[n + degree] * fcoeff[nq + degree] + gcoeff[n + degree] * gcoeff[nq + degree]) \
                        * (fcoeff[n + r + degree] * fcoeff[mp + degree] + gcoeff[n + r + degree] * gcoeff[mp + degree]) \
                        * torch.exp(torch.tensor(-0.25 * K ** 2 * sigma ** 2 * (mp - nq - r) ** 2))
#             else:
#                 print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
        p_err += cr * Hr
        # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return p_err



# A better version directly pass the f and g coefficients
def loss_fn_exact_coeff(fcoeff, gcoeff, beta_th, K, sigma):
    # calculate the half-period of the qubit response function
    half_period = torch.pi / (2 * K)
    degree = int((len(fcoeff) - 1) / 2)
    
      
    
    # compute p_err from the coefficients analytically
    p_err = 0.0
    for r in range(-2 * degree, 2 * degree + 1):
        # this is range of integration for [-pi / 2k, pi / 2k]
        Hr = 2 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 2.0)) \
           - (4 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        
        # this is a restricted range of integration for [-pi / 4k, pi / 4k]
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi*r/4.0)) \
        #     - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K*r*beta_th))
        cr = 0.0
        for n in range(-degree, degree + 1):
            if abs(n + r) <= degree:
                for mp in range(-degree, degree + 1):
                    for nq in range(-degree, degree + 1):
                        cr += (fcoeff[n + degree] * fcoeff[nq + degree] + gcoeff[n + degree] * gcoeff[nq + degree]) \
                        * (fcoeff[n + r + degree] * fcoeff[mp + degree] + gcoeff[n + r + degree] * gcoeff[mp + degree]) \
                        * torch.exp(torch.tensor(-0.25 * K ** 2 * sigma ** 2 * (mp - nq - r) ** 2))
#             else:
#                 print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
        p_err += cr * Hr
        # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return p_err


def loss_fn_exact(phases, beta_th, K, sigma, flag_callback = False):
    # calculate the half-period of the qubit response function
    half_period = torch.pi / (2 * K)
    degree = len(phases) - 1
    
      
    # make torch tensors to store the coefficients of the QSP
    # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    
    # compute the QSP coefficients using phases
    # fcoeff, gcoeff = calc_qsp_coeff(phases).to(device)
    fcoeff, gcoeff = calc_qsp_coeff_tensor(phases)
    #print("fcoeff = ", fcoeff)
    #print("gcoeff = ", gcoeff)
    
    # compute p_err from the coefficients analytically
    p_err = 0.0
    for r in range(-2 * degree, 2 * degree + 1):
        # this is range of integration for [-pi / 2k, pi / 2k]
        Hr = 2 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 2.0)) \
           - (4 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        
        # this is a restricted range of integration for [-pi / 4k, pi / 4k]
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 4.0)) \
        #     - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        cr = 0.0
        for n in range(-degree, degree + 1):
            if abs(n + r) <= degree:
                for mp in range(-degree, degree + 1):
                    for nq in range(-degree, degree + 1):
                        cr += (fcoeff[n + degree] * fcoeff[nq + degree] + gcoeff[n + degree] * gcoeff[nq + degree]) \
                        * (fcoeff[n + r + degree] * fcoeff[mp + degree] + gcoeff[n + r + degree] * gcoeff[mp + degree]) \
                        * torch.exp(torch.tensor(-0.25 * K ** 2 * sigma ** 2 * (mp - nq - r) ** 2))
#             else:
#                 print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
        p_err += cr * Hr
        # print("r, cr, Hr = ", r, cr, Hr)
        
    if flag_callback:
        print(" => " + str(datetime.now()) + ": loss = ", p_err.item(), "  Phases = ", phases, " Ccoeff = ", signal_poly_coeff(phases, K, sigma))
        # print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "Ccoeff = ", ccoeff)
    # return p_err.to(device)
    return p_err

def signal_poly_coeff(phases, K, sigma):
    # calculate the half-period of the qubit response function
    half_period = torch.pi / (2 * K)
    degree = len(phases) - 1
    
      
    # make torch tensors to store the coefficients of the QSP
    # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    ccoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    
    # compute the QSP coefficients using phases
    # fcoeff, gcoeff = calc_qsp_coeff(phases).to(device)
    fcoeff, gcoeff = calc_qsp_coeff_tensor(phases)
    #print("fcoeff = ", fcoeff)
    #print("gcoeff = ", gcoeff)
    
    for s in range(-degree, degree + 1):
        cs = 0.0
        for t in range(-degree, degree + 1):
            for n in range(-degree, degree + 1):
                 for nq in range(-degree, degree + 1):
                    if abs(n + 2 * s) <= degree and abs(nq + 2 * t) <= degree:
                        cs += (fcoeff[n + degree] * fcoeff[nq + degree] + gcoeff[n + degree] * gcoeff[nq + degree]) \
                            * (fcoeff[n + 2 * s + degree] * fcoeff[nq + 2 * t + degree] + gcoeff[n + 2 * s + degree] * gcoeff[nq + 2 * t + degree]) \
                            * torch.exp(torch.tensor(-K ** 2 * sigma ** 2 * (t - s) ** 2))
#         else:
#             print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
        ccoeff[s + degree] = cs
    # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return ccoeff

def signal_poly_coeff_qsp2(phases, K):
    # calculate the half-period of the qubit response function
    half_period = torch.pi / (2 * K)
    degree = int(len(phases) / 2 - 1)
      
    # make torch tensors to store the coefficients of the QSP
    # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    fcoeffp = torch.zeros(2 * degree + 1, dtype = torch.float)
    gcoeffp = torch.zeros(2 * degree + 1, dtype = torch.float)
    ccoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    
    phases1 = phases[:(degree + 1)]
    phases2 = phases[(degree + 1):]
    
    # compute the QSP coefficients using phases
    # fcoeff, gcoeff = calc_qsp_coeff(phases).to(device)
    fcoeff, gcoeff = calc_qsp_coeff_tensor(phases1)
    fcoeffp, gcoeffp = calc_qsp_coeff_tensor(phases2)
    #print("fcoeff = ", fcoeff)
    #print("gcoeff = ", gcoeff)
    
    for s in range(-degree, degree + 1):
        cs = 0.0
        for t in range(-degree, degree + 1):
            for n in range(-degree, degree + 1):
                 for nq in range(-degree, degree + 1):
                    if abs(n + 2 * s) <= degree and abs(nq + 2 * t) <= degree:
                        cs += (fcoeff[n + degree] * fcoeffp[nq + degree] + gcoeff[n + degree] * gcoeffp[nq + degree]) \
                            * (fcoeff[n + 2 * s + degree] * fcoeffp[nq + 2 * t + degree] + gcoeff[n + 2 * s + degree] * gcoeffp[nq + 2 * t + degree]) \
                            * torch.exp(torch.tensor(-K ** 2 * (t + s) ** 2))
#         else:
#             print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
        ccoeff[s + degree] = cs
    # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return ccoeff

def loss_fn_signal_coeff(phases, target_coeff, K, flag_callback = False):
    
    ccoeff = signal_poly_coeff(phases, K)
    loss = LA.norm(ccoeff - target_coeff)
    loss = loss * loss
    
    if flag_callback:
        print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "  Phases = ", phases, "Ccoeff = ", ccoeff)
        # print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "Ccoeff = ", ccoeff)
    
    return loss

def loss_fn_signal_coeff_qsp2(phases, target_coeff, K, flag_callback = False):
    
    
    ccoeff = signal_poly_coeff_qsp2(phases, K)
    loss = LA.norm(ccoeff - target_coeff)
    loss = loss*loss
    
    degree = int(len(phases) / 2 - 1)
    phases1 = phases[:(degree + 1)]
    phases2 = phases[(degree + 1):]
    
    
    if flag_callback:
        print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "  Phases1 = ", phases1, "  Phases2 = ", phases2, "Ccoeff = ", ccoeff)
        # print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "Ccoeff = ", ccoeff)
    
    return loss

def loss_fn_exact_constraint(phases_mus, beta_th, K, kappa, delta_p, delta_s, fn_qb_res, loss_fn_exact, flag_callback = False):
    # calculate the half-period of the new loss function, with two additional constraints on the value of the qubit response function at the end of the passband and beginning of the stopband
    # K: the original displacement amount used to construct the QSP circuit
    # kappa: width of the rising/falling edge of the qubit response function
    # delta_p: tolerance of the passband, Prob in [1 - 2 * delta_p, 1]
    # delta_s: tolerance of the stopband, Prob in [0, 2 * delta_s]
    # fn_qb_res: the qubit response function
    
    half_period = torch.pi / (2 * K)

    phases = torch.tensor(phases_mus[:-2])
    # phases = torch.cat((phases, torch.flip(phases, [0])))
    phases = torch.cat((phases, phases.flip([0])))
    mus = phases_mus[-2:]
    degree = len(phases) - 1

    # compute the QSP coefficients using phases
    fcoeff, gcoeff = calc_qsp_coeff(phases)
#     # print("fcoeff = ", fcoeff)
#     # print("gcoeff = ", gcoeff)
    
    # beta_p: position of the end of the passband
    # beta_s: position of the start of the stopband
    beta_p = beta_th - 0.5 * kappa
    beta_s = beta_th + 0.5 * kappa
    
    # now compute the two new constraints at beta_p and beta_s
    grad0 = (fn_qb_res(fcoeff, gcoeff, beta_p, K) - (1 - 2 * delta_p))
    grad1 = (fn_qb_res(fcoeff, gcoeff, beta_s, K) - 2 * delta_s)
    qb_res0 = mus[0] * grad0
    qb_res1 = mus[1] * grad1
    
    
      
#     # make torch tensors to store the coefficients of the QSP
#     fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
#     gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
#     # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
#     # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float)
    
    
#     # compute p_err from the coefficients analytically
#     p_err = 0.0
#     for r in range(-2 * degree, 2 * degree + 1):
#         # this is range of integration for [-pi / 2k, pi / 2k]
#         # Hr = 2 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 2.0)) \
#         #    - (4 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        
#         # this is a restricted range of integration for [-pi/4k, pi/4k]
#         Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 4.0)) \
#             - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
#         cr = 0.0
#         for n in range(-degree, degree+1):
#             if abs(n+r) <= degree:
#                 for mp in range(-degree, degree + 1):
#                     for nq in range(-degree, degree + 1):
#                         cr += (fcoeff[n + degree] * fcoeff[nq + degree] + gcoeff[n + degree] * gcoeff[nq + degree]) \
#                         * (fcoeff[n + r + degree] * fcoeff[mp + degree] + gcoeff[n + r + degree] * gcoeff[mp + degree]) \
#                         * torch.exp(torch.tensor(-0.25 * K ** 2 * (mp - nq - r) ** 2))
# #             else:
# #                 print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
#         p_err += cr * Hr
#         # print("r, cr, Hr = ", r, cr, Hr)

    # compute the exact loss function without the constraints
    # p_err = loss_fn_exact(phases, beta_th, K)
    p_err = loss_fn_exact(fcoeff, gcoeff, beta_th, K)
        
    # now add the two constraints to the original p_err
    p_err += qb_res0 + qb_res1
        
    if flag_callback:
        print(" => " + str(datetime.now()) + ": p_err = ", p_err.item(), ";  Mus = ", mus, ";  Phases = ", phases)
        print("      Grad Mus = {}, {} \n".format(grad0.item(), grad1.item()))
    # return p_err.to(device)
    return p_err


def signal_poly_prob_grid_qsp(phases, K, sigma, num_grid):
    '''
    Calculate the probability of the qubit response function in the range of [-pi/2K, pi/2K]
    given a grid spacing of 
    '''
    degree = len(phases) - 1
    ccoeff = signal_poly_coeff(phases, K, sigma)
    beta_grid = torch.from_numpy(np.linspace(-torch.pi / (2 * K), torch.pi / (2 * K), num_grid))
    prob_grid = torch.zeros(num_grid, dtype=torch.cfloat)
    
    for s in range(-degree, degree + 1):
        prob_grid += ccoeff[s + degree] * torch.exp(1j * 2 * K * s * beta_grid)
    
    if torch.norm(prob_grid.imag) > 1e-5:
        print("prob has imaginary part with norm ", torch.norm(prob_grid.imag))
        exit()
    return beta_grid, prob_grid.real
#############################################################################################


#############################################################################################
########################### Trigonometric QSP ###############################################
def calc_triqsp_coeff(phases):
    #print("phases", phases)
    degree = int(len(phases) / 2 - 1)
    # print("degree = ", degree)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.complex64)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.complex64)

    phases_theta = phases[:(degree + 1)]
    phases_phi = phases[(degree + 1):]
    
    if degree == 0:
        fcoeff[0] = torch.cos(phases_theta[0]) * torch.exp(1j * phases_phi[0])
        gcoeff[0] = -torch.sin(phases_theta[0]) * torch.exp(-1j * phases_phi[0])
        # fcoeff[0] = torch.cos(torch.tensor(phases[0]))
        # gcoeff[0] = torch.sin(torch.tensor(phases[0]))
    else:
        phases_theta_reduce = torch.tensor(phases_theta[:-1])
        phases_phi_reduce = torch.tensor(phases_phi[:-1])
        phases_reduce = torch.cat((phases_theta_reduce, phases_phi_reduce), dim = 0)
        # print("phases_reduce = ", phases_reduce)
        fcoeff_reduce, gcoeff_reduce = calc_triqsp_coeff(phases_reduce)
        # now compute the new coefficients from the old one
        for idx in range(-degree, degree + 1):
            # print("idx = ", idx)
            if idx >= degree - 1:
                # print(idx + degree - 2)
                if idx - 1 >= 0:
                    # note the index in the RHS has to be idx+degree-2, instead of idx+degree-1 because there is also one negative index coefficient less.
                    # fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                    # gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                    fcoeff[idx + degree] = torch.cos(torch.tensor(phases_theta[-1])) * fcoeff_reduce[idx + degree - 2] * torch.exp(1j * torch.tensor(phases_phi[-1]))
                    gcoeff[idx + degree] = -torch.sin(torch.tensor(phases_theta[-1])) * fcoeff_reduce[idx + degree - 2] * torch.exp(-1j * torch.tensor(phases_phi[-1]))
            elif idx <= -(degree - 1):
                if idx+1 <= 0:
                    # fcoeff[idx + degree] = -torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                    # gcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                    fcoeff[idx + degree] = torch.sin(torch.tensor(phases_theta[-1])) * gcoeff_reduce[idx + degree] * torch.exp(1j * torch.tensor(phases_phi[-1]))
                    gcoeff[idx + degree] = torch.cos(torch.tensor(phases_theta[-1])) * gcoeff_reduce[idx + degree] * torch.exp(-1j * torch.tensor(phases_phi[-1]))
            elif abs(idx) <= degree - 2:
                # fcoeff[idx+degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
                #                     - torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                # gcoeff[idx+degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
                #                     + torch.cos(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                fcoeff[idx + degree] = (torch.cos(torch.tensor(phases_theta[-1])) * fcoeff_reduce[idx + degree - 2] \
                                    + torch.sin(torch.tensor(phases_theta[-1])) * gcoeff_reduce[idx + degree]) * torch.exp(1j * torch.tensor(phases_phi[-1]))
                gcoeff[idx + degree] = ( -torch.sin(torch.tensor(phases_theta[-1])) * fcoeff_reduce[idx + degree - 2] \
                                    + torch.cos(torch.tensor(phases_theta[-1])) * gcoeff_reduce[idx + degree]) * torch.exp(-1j * torch.tensor(phases_phi[-1]))
            else:
                print("something wrong with indexing. check.")
  
  # for test
  # print("fcoeff", fcoeff)
  # print("gcoeff", gcoeff)
    
    return fcoeff, gcoeff

def test_calc_triqsp_coeff():

  test_thres = 1e-4
  phases = torch.tensor([torch.pi / 7, torch.pi / 8, -torch.pi / 9, torch.pi / 4, torch.pi / 3, -torch.pi / 10])
  fcoeff_correct = torch.tensor([-0.0747 - 0.1150j, 0.0000 + 0.0000j, -0.0569 - 0.0561j, 0.0000 + 0.0000j, 0.0409 + 0.7811j])
  gcoeff_correct = torch.tensor([-0.0197 + 0.3762j, 0.0000 + 0.0000j, -0.3712 - 0.0479j, 0.0000 + 0.0000j, -0.1551 + 0.2388j])
  fcoeff, gcoeff = calc_triqsp_coeff(phases)
  if torch.mean(torch.abs(fcoeff - fcoeff_correct)) < test_thres \
    and torch.mean(torch.abs(gcoeff - gcoeff_correct)) < test_thres:
    print("passed test.")
  else:
    print("failed test.")

    
def signal_poly_coeff_triqsp(phases, K, sigma):
    # calculate the half-period of the qubit response function
    degree = int(len(phases) / 2 - 1)
    
      
    # make torch tensors to store the coefficients of the QSP
    # fcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    # gcoeff = torch.zeros(2 * degree + 1, dtype = torch.float).to(device)
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.complex64)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.complex64)
    ccoeff = torch.zeros(2 * degree + 1, dtype = torch.complex64)
    
    # compute the QSP coefficients using phases
    # fcoeff, gcoeff = calc_qsp_coeff(phases).to(device)
    fcoeff, gcoeff = calc_triqsp_coeff(phases)
    # print("fcoeff = ", fcoeff)
    # print("gcoeff = ", gcoeff)
    
    for s in range(-degree, degree + 1):
        cs = 0.0
        for t in range(-degree, degree + 1):
            for n in range(-degree, degree + 1):
                 for nq in range(-degree, degree + 1):
                    if abs(n + 2 * s) <= degree and abs(nq + 2 * t) <= degree:
                        cs += (fcoeff[n + degree] * torch.conj(fcoeff[nq + degree]) + gcoeff[n + degree] * torch.conj(gcoeff[nq + degree])) \
                            * (torch.conj(fcoeff[n + 2 * s + degree]) * fcoeff[nq + 2 * t + degree] + torch.conj(gcoeff[n + 2 * s + degree]) * gcoeff[nq + 2 * t + degree]) \
                            * torch.exp(torch.tensor(-K ** 2 * sigma ** 2 * (t - s) ** 2))
                        # print("cs = ", cs)
                    # else:
                    #     print("m, mp = ", n + 2 * s, nq + 2 * t, ". index out of range for m, mp. Skip. ")
        ccoeff[s+degree] = cs
    # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return ccoeff

def loss_fn_signal_coeff_triqsp(phases, target_coeff, K, sigma, flag_callback = False):
    
    ccoeff = signal_poly_coeff_triqsp(phases, K, sigma)
    # loss = LA.norm(ccoeff - target_coeff)
    loss = LA.norm(ccoeff.real - target_coeff)
#     loss = loss * loss
    
    if flag_callback:
        print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "  Phases = ", phases, "Ccoeff = ", ccoeff)
        # print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "Ccoeff = ", ccoeff)
    
    return loss


# The original loss function as defined in the paper
def loss_fn_exact_triqsp(phases, beta_th, K, sigma, flag_callback = False):
    # calculate the half-period of the qubit response function
    degree = int(len(phases) / 2 - 1)
    
      
    # make torch tensors to store the coefficients of the QSP
    fcoeff = torch.zeros(2 * degree + 1, dtype = torch.complex64)
    gcoeff = torch.zeros(2 * degree + 1, dtype = torch.complex64)
    
    # compute the QSP coefficients using phases
    fcoeff, gcoeff = calc_triqsp_coeff(phases)
    #print("fcoeff = ", fcoeff)
    #print("gcoeff = ", gcoeff)
    
    # compute p_err from the coefficients analytically
    p_err = 0.0
    for r in range(-2 * degree, 2 * degree + 1):
        # this is range of integration for [-pi / 2k, pi / 2k]
        if r == 0:
            Hr = 1 - 2 * K * beta_th / torch.pi
        else:
            Hr = 2 * ( -1j - 1j * torch.exp(torch.tensor(1j * torch.pi * r / 2)) + 2 * 1j * torch.exp(torch.tensor(1j * K * r * beta_th)) + K * r * beta_th) / (torch.pi * r)
        
        # this is a restricted range of integration for [-pi / 4k, pi / 4k]
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 4.0)) \
        #     - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        cr = 0.0
        for n in range(-degree, degree + 1):
            if abs(n + r) <= degree:
                for mp in range(-degree, degree + 1):
                    for nq in range(-degree, degree + 1):
                        cr += (fcoeff[n+degree] * torch.conj(fcoeff[nq + degree]) + gcoeff[n + degree] * torch.conj(gcoeff[nq + degree])) \
                        * (torch.conj(fcoeff[n + r + degree]) * fcoeff[mp + degree] + torch.conj(gcoeff[n + r + degree]) * gcoeff[mp + degree]) \
                        * torch.exp(torch.tensor(-0.25 * K ** 2 * sigma ** 2 * (mp - nq - r) ** 2))
#            else:
#                 print("r, m = ", r, n+r, ". index out of range for m. Skip. ")
        p_err += cr * Hr
        # print("r, cr, Hr = ", r, cr, Hr)
                            
    if flag_callback:
        print(" => " + str(datetime.now()) + ": loss = ", p_err.real.item(), "  Phases = ", phases, " Ccoeff = ", signal_poly_coeff_triqsp(phases, K, sigma))
        # print(" => " + str(datetime.now()) + ": loss = ", loss.item(), "Ccoeff = ", ccoeff)
        
    # return p_err.to(device)
    # print("p_err", p_err)
    if torch.abs(p_err.imag) > 1e-5:
        print("p_err has imaginary part, ", p_err.imag)
        exit()
    return p_err.real

def signal_poly_prob_grid(phases, K, sigma, num_grid):
    '''
    Calculate the probability of the qubit response function in the range of [-pi / 2K, pi / 2K]
    given a grid spacing of pi / (K * num_grid)
    '''
    degree = int(len(phases) / 2 - 1)
    ccoeff = signal_poly_coeff_triqsp(phases, K, sigma)
    beta_grid = torch.from_numpy(np.linspace(-torch.pi / (2 * K), torch.pi / (2 * K), num_grid))
    prob_grid = torch.zeros(num_grid, dtype = torch.cfloat)
    
    for s in range(-degree, degree + 1):
        prob_grid += ccoeff[s + degree] * torch.exp(1j * 2 * K * s * beta_grid)
    
    if torch.norm(prob_grid.imag) > 1e-5:
        print("prob has imaginary part with norm ", torch.norm(prob_grid.imag))
        exit()
    return beta_grid, prob_grid.real
#############################################################################################
########################### Trigonometric QSP ###############################################

np.random.seed()

# now try to directly optimize for the coeffcient

thresh = 1e-5
tol = 1e-8
# torch.manual_seed(0)

K = 1 / 2048
sigma = 1
degree_list = [9]
num_trials = 1
eta = 0.25
beta_th = eta * np.pi / K
d_best_results = []
d_best_phases = []
d_best_losses = []

opt_method = 'Nelder-Mead'
options = {'fatol': thresh, 'xatol': thresh}

for degree in degree_list:
    print("Degree", degree)
    best_results = []
    best_phases = []
    best_losses = []
    for trial in range(num_trials):
        print("Trial", trial + 1)
        # Try the protocol with QSP
        phases0 = torch.rand(degree + 1) * 2 * torch.pi
        
        # This is to use the original loss function directly
        new_callback = partial(loss_fn_exact, beta_th = beta_th, K = K, sigma = sigma, flag_callback = True)
        res = scipy.optimize.minimize(loss_fn_exact, phases0, (beta_th, K, sigma),
                                      method = opt_method, tol = thresh, callback = new_callback, options = options)
        best_results.append(res)
        best_phases.append(res.x)
        best_losses.append(res.fun)
    best_losses = np.array(best_losses)
    d_best_results.append(best_results)
    d_best_phases.append(best_phases)
    d_best_losses.append(best_losses)

for i, phases in enumerate(d_best_phases):
    print("Degree", degree_list[i])
    best_index = np.argmin(d_best_losses[i])
    best_loss = d_best_losses[i][best_index]
    best_phases = phases[best_index]
    print("Best loss for degree " + str(degree_list[i]) + ":", best_loss)
    print("Best phases for degree " + str(degree_list[i]) + ":", best_phases)
