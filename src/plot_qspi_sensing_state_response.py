import csv
import itertools as it
import pandas as pd
import numpy as np
import sklearn.decomposition
from tqdm import tqdm
import re
import string
import random
import matplotlib
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
            # print(idx + degree - 2)
            if idx - 1 >= 0:
                # note the index in the RHS has to be idx + degree - 2, instead of idx + degree - 1 because there is also one negative index coefficient less.
                # fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                # gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                fcoeff[idx + degree] = torch.cos(phases[-1]) * fcoeff_reduce[idx + degree - 2]
                gcoeff[idx + degree] = torch.sin(phases[-1]) * fcoeff_reduce[idx + degree - 2]
        elif idx <= -(degree - 1):
            if idx+1 <= 0:
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
  # print("phases", phases)
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
    for idx in range(-degree,degree + 1):
        # print("idx = ", idx)
        if idx >= degree - 1:
            # print(idx + degree - 2)
            if idx - 1 >= 0:
                # note the index in the RHS has to be idx + degree - 2, instead of idx + degree - 1 because there is also one negative index coefficient less.
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
        Hr = torch.cos(torch.tensor(K*r*beta_th))
        
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
        Hr = torch.cos(torch.tensor(K*r*beta_th))
        
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
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r/4.0)) \
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
                        cs += (fcoeff[n+degree] * fcoeffp[nq + degree] + gcoeff[n + degree] * gcoeffp[nq + degree]) \
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
    loss = loss * loss
    
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
    # print("fcoeff = ", fcoeff)
    # print("gcoeff = ", gcoeff)
    
    # beta_p: position of the end of the passband
    # beta_s: position of the start of the stopband
    beta_p = beta_th - 0.5 * kappa
    beta_s = beta_th + 0.5 * kappa
    
    # now compute the two new constraints at beta_p and beta_s
    grad0 = (fn_qb_res(fcoeff, gcoeff, beta_p, K) - (1 - 2 * delta_p) )
    grad1 = (fn_qb_res(fcoeff, gcoeff, beta_s, K) - 2 * delta_s )
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
        
#         # this is a restricted range of integration for [-pi / 4k, pi / 4k]
#         Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 4.0)) \
#             - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
#         cr = 0.0
#         for n in range(-degree, degree + 1):
#             if abs(n + r) <= degree:
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
    Calculate the probability of the qubit response function in the range of [-pi / 2K, pi / 2K]
    given a grid spacing of 
    '''
    degree = len(phases) - 1
    ccoeff = signal_poly_coeff(phases, K, sigma)
    beta_grid = torch.from_numpy(np.linspace(-torch.pi / (2 * K), torch.pi / (2 * K), num_grid))
    prob_grid = torch.zeros(num_grid, dtype = torch.cfloat)
    
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
                    # note the index in the RHS has to be idx + degree - 2, instead of idx + degree - 1 because there is also one negative index coefficient less.
                    # fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                    # gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2]
                    fcoeff[idx + degree] = torch.cos(torch.tensor(phases_theta[-1])) * fcoeff_reduce[idx + degree - 2] * torch.exp(1j * torch.tensor(phases_phi[-1]))
                    gcoeff[idx + degree] = -torch.sin(torch.tensor(phases_theta[-1])) * fcoeff_reduce[idx + degree - 2] * torch.exp(-1j * torch.tensor(phases_phi[-1]))
            elif idx <= -(degree - 1):
                if idx + 1 <= 0:
                    # fcoeff[idx + degree] = -torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                    # gcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                    fcoeff[idx + degree] = torch.sin(torch.tensor(phases_theta[-1])) * gcoeff_reduce[idx + degree] * torch.exp(1j * torch.tensor(phases_phi[-1]))
                    gcoeff[idx + degree] = torch.cos(torch.tensor(phases_theta[-1])) * gcoeff_reduce[idx + degree] * torch.exp(-1j * torch.tensor(phases_phi[-1]))
            elif abs(idx) <= degree - 2:
                # fcoeff[idx + degree] = torch.cos(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
                #                     - torch.sin(torch.tensor(phases[-1])) * gcoeff_reduce[idx + degree]
                # gcoeff[idx + degree] = torch.sin(torch.tensor(phases[-1])) * fcoeff_reduce[idx + degree - 2] \
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
        ccoeff[s + degree] = cs
    # print("r, cr, Hr = ", r, cr, Hr)
        
    # return p_err.to(device)
    return ccoeff

def loss_fn_signal_coeff_triqsp(phases, target_coeff, K, sigma, flag_callback = False):
    
    ccoeff = signal_poly_coeff_triqsp(phases, K, sigma)
    # loss = LA.norm(ccoeff - target_coeff)
    loss = LA.norm(ccoeff.real - target_coeff)
#     loss = loss*loss
    
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
            Hr = 2 * ( -1j - 1j * torch.exp(torch.tensor(1j * torch.pi * r / 2)) + 2 * 1j * torch.exp(torch.tensor(1j * K * r * beta_th)) + K * r * beta_th ) / (torch.pi * r)
        
        # this is a restricted range of integration for [-pi / 4k, pi / 4k]
        # Hr = 4 * K * beta_th / torch.pi + mysinc(torch.tensor(torch.pi * r / 4.0)) \
        #     - (8 * K * beta_th / torch.pi) * mysinc(torch.tensor(K * r * beta_th))
        cr = 0.0
        for n in range(-degree, degree + 1):
            if abs(n + r) <= degree:
                for mp in range(-degree, degree + 1):
                    for nq in range(-degree, degree + 1):
                        cr += (fcoeff[n + degree] * torch.conj(fcoeff[nq + degree]) + gcoeff[n + degree] * torch.conj(gcoeff[nq + degree])) \
                        * (torch.conj(fcoeff[n + r + degree]) * fcoeff[mp + degree] + torch.conj(gcoeff[n + r + degree]) * gcoeff[mp + degree]) \
                        * torch.exp(torch.tensor(-0.25 * K ** 2 * sigma ** 2 * (mp - nq - r) ** 2))
#            else:
#                 print("r, m = ", r, n + r, ". index out of range for m. Skip. ")
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
    given a grid spacing of 
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

def signal_poly_prob_grid_qsp_partial(phases, K, sigma, num_grid, prop, beginning = True):
    '''
    Calculate the probability of the qubit response function in a proportion prop of the range of [-pi / 2K, pi / 2K], where this portion starts at -pi / 2K if beginning = True and ends at pi / 2K otherwise
    given a grid spacing of 
    '''
    degree = len(phases) - 1
    ccoeff = signal_poly_coeff(phases, K, sigma)
    if beginning:
        beta_grid = torch.from_numpy(np.linspace(-torch.pi / (2 * K), prop * torch.pi / (2 * K), num_grid))
    else:
        beta_grid = torch.from_numpy(np.linspace(-prop * torch.pi / (2 * K), torch.pi / (2 * K), num_grid))
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

#specify parameters for plotting
num_grid = 1000
K = 1 / 2048
sigma = 1
eta = 0.25
beta_th = eta * np.pi / K

# store all of the optimal QSPI sensing-state phases that we found for plotting qubit response functions
phases1 = torch.tensor([3.92699082, 1.18831717])
phases2 = torch.tensor([3.14158545, 5.49778714, 2.14366189])
phases3 = torch.tensor([3.95413797, 4.1344972, 0.05429256, 1.38243345])
phases4 = torch.tensor([1.36652403, 3.4147937, 3.53677557, 5.36928697, 4.22445961])
phases5 = torch.tensor([2.35619562, 2.16251256, 3.14159578, 3.53940905, 3.14159019, 0.74547365])
phases6 = torch.tensor([3.47221108, 6.39322977, 2.03140306, 3.83763273, 3.34231622, 3.12090727, 2.11557662])
phases7 = torch.tensor([4.44918921, 0.38949998, 2.922373, 0.58311111, 2.22541229, 3.34946588, 6.94736085, 1.55650721])
phases8 = torch.tensor([4.40925344, 0.64010868, 6.12261423, 5.5679031, 3.86168694, 2.89580752, 3.41575066, 5.90281062, 1.47019342])
phases9 = torch.tensor([2.28907563e+00, 5.32992720e-07, 5.29348575e+00, 3.14159173e+00, 5.76908827e+00, 6.28318418e+00, 1.07601037e+00, 8.43953820e-07, 3.63649790e+00, 8.83930272e-01])
phases10 = torch.tensor([0.51225058, 0.75417921, 3.04459114, 0.66257205, 0.81209149, 2.41909505, 3.07485433, 2.86795647, 2.18691807, 0.36477988, 0.85439159])
phases11 = torch.tensor([0.53318057, 0.38179323, 5.90292125, 3.61143081, 0.08996479, 5.11962931, 5.83420082, 5.81609809, 6.24367924, 0.67687692, 5.78068303, 0.65789585])
phases12 = torch.tensor([3.58487941, 3.38203082, 0.11354016, 1.08176341, 3.47791685, 2.98483514, 3.77445565, 2.84576103, 5.7709177, 2.64519602, 2.16698018, 3.49721315, 0.24832305])
phases13 = torch.tensor([3.5604268, 3.53539003, 6.24496916, 0.94611558, 3.65493259, 2.87248805, 3.4106715, 3.56083945, 5.31024663, 3.35290876, 2.31445542, 2.65849403, 3.04401507, 1.40109045])
phases15 = torch.tensor([3.96027657, 3.3228269, 0.58410433, 0.67308454, 3.25323775, 2.61830191, 2.86563178, 3.8736934, 5.22492424, 2.92001249, 3.09553139, 3.16709671, 3.5727264, 0.77114436, 3.38987993, 1.57901798])
phases17 = torch.tensor([4.01204606, 3.36302421, 0.61104152, 0.62579379, 3.06581316, 2.84454956, 2.33640641, 4.01871977, 5.71537902, 2.61609113, 3.50250199, 3.64457045, 3.78121014, 0.79085748, 2.97885651, 3.48682301, 3.08376274, 1.24845439])

# specify what range of values to plot over (i.e., proportion of the total range) and whether to start from the beginning or the end when defining this range
prop = 0.44
beginning = False

# increase line thickness for zoomed insets
linewidth = 3

# compute qubit response functions
beta_grid1, prob_grid1 = signal_poly_prob_grid_qsp_partial(phases1, K, sigma, num_grid, prop, beginning)
beta_grid2, prob_grid2 = signal_poly_prob_grid_qsp_partial(phases2, K, sigma, num_grid, prop, beginning)
beta_grid3, prob_grid3 = signal_poly_prob_grid_qsp_partial(phases3, K, sigma, num_grid, prop, beginning)
beta_grid4, prob_grid4 = signal_poly_prob_grid_qsp_partial(phases4, K, sigma, num_grid, prop, beginning)
beta_grid5, prob_grid5 = signal_poly_prob_grid_qsp_partial(phases5, K, sigma, num_grid, prop, beginning)
beta_grid6, prob_grid6 = signal_poly_prob_grid_qsp_partial(phases6, K, sigma, num_grid, prop, beginning)
beta_grid7, prob_grid7 = signal_poly_prob_grid_qsp_partial(phases7, K, sigma, num_grid, prop, beginning)
beta_grid8, prob_grid8 = signal_poly_prob_grid_qsp_partial(phases8, K, sigma, num_grid, prop, beginning)
beta_grid9, prob_grid9 = signal_poly_prob_grid_qsp_partial(phases9, K, sigma, num_grid, prop, beginning)
beta_grid10, prob_grid10 = signal_poly_prob_grid_qsp_partial(phases10, K, sigma, num_grid, prop, beginning)
beta_grid11, prob_grid11 = signal_poly_prob_grid_qsp_partial(phases11, K, sigma, num_grid, prop, beginning)
beta_grid12, prob_grid12 = signal_poly_prob_grid_qsp_partial(phases12, K, sigma, num_grid, prop, beginning)
beta_grid13, prob_grid13 = signal_poly_prob_grid_qsp_partial(phases13, K, sigma, num_grid, prop, beginning)
beta_grid15, prob_grid15 = signal_poly_prob_grid_qsp_partial(phases15, K, sigma, num_grid, prop, beginning)
beta_grid17, prob_grid17 = signal_poly_prob_grid_qsp_partial(phases17, K, sigma, num_grid, prop, beginning)

# define limited range over one period in beta
beta_grid1 /= torch.pi / (2 * K)
beta_grid2 /= torch.pi / (2 * K)
beta_grid3 /= torch.pi / (2 * K)
beta_grid4 /= torch.pi / (2 * K)
beta_grid5 /= torch.pi / (2 * K)
beta_grid6 /= torch.pi / (2 * K)
beta_grid7 /= torch.pi / (2 * K)
beta_grid8 /= torch.pi / (2 * K)
beta_grid9 /= torch.pi / (2 * K)
beta_grid10 /= torch.pi / (2 * K)
beta_grid11 /= torch.pi / (2 * K)
beta_grid12 /= torch.pi / (2 * K)
beta_grid13 /= torch.pi / (2 * K)
beta_grid15 /= torch.pi / (2 * K)
beta_grid17 /= torch.pi / (2 * K)
beta_th /= torch.pi / (2 * K)

# define figure dimensions, and create the figure
width = 6.4
height = 4.8
plt.figure(figsize = (width, height))

# add a vertical line showing the threshold value of beta, as well as horizontal reference lines as necessary
plt.axvline(x = beta_th, color = 'r', linestyle = '--', linewidth = linewidth)
# plt.axhline(y = 1, color = 'black', linestyle = '--')
# plt.axhline(y = 0, color = 'black', linestyle = '--')
# plt.axvline(x = beta_th + 0.5 * kappa, color = 'b')
# plt.axvline(x = beta_th - 0.5 * kappa, color = 'b')
# plt.axhline(y = 1 - 2 * delta_p, color = 'r')
# plt.axhline(y = 2*delta_s, color = 'r')

# plot only the qubit response functions for desired degrees
plt.plot(beta_grid1, prob_grid1, label = 'd = 1', linewidth = linewidth)
# plt.plot(beta_grid2, prob_grid2, label = 'd = 2', linewidth = linewidth)
# plt.plot(beta_grid3, prob_grid3, label = 'd = 3', linewidth = linewidth)
# plt.plot(beta_grid4, prob_grid4, label = 'd = 4', linewidth = linewidth) # looks to be off in the threshold
plt.plot(beta_grid5, prob_grid5, label = 'd = 5', linewidth = linewidth)
# plt.plot(beta_grid6, prob_grid6, label = 'd = 6', linewidth = linewidth)
# plt.plot(beta_grid7, prob_grid7, label = 'd = 7', linewidth = linewidth)
# plt.plot(beta_grid8, prob_grid8, label = 'd = 8', linewidth = linewidth)
plt.plot(beta_grid9, prob_grid9, label = 'd = 9', linewidth = linewidth)
# plt.plot(beta_grid10, prob_grid10, label = 'd = 10', linewidth = linewidth)
# plt.plot(beta_grid11, prob_grid11, label = 'd = 11', linewidth = linewidth)
# plt.plot(beta_grid12, prob_grid12, label = 'd = 12', linewidth = linewidth)
plt.plot(beta_grid13, prob_grid13, label = 'd = 13', linewidth = linewidth)
# plt.plot(beta_grid15, prob_grid15, label = 'd = 15', linewidth = linewidth)
plt.plot(beta_grid17, prob_grid17, label = 'd = 17', linewidth = linewidth)

# set display parameters for plot
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
#matplotlib.rcParams.update({'font.size': 18})
params = {'axes.labelsize': 36,'axes.titlesize':36, 'legend.fontsize': 36, 'xtick.labelsize': 36, 'ytick.labelsize': 36}
matplotlib.rcParams.update(params)

# set title and labels on plot
# plt.title('Qubit Response, k = ' + r'$2^{-11}$')
plt.xlabel(r'$\beta$'+ ' (in units of ' + r'$\frac{\pi}{2k}) $')
plt.ylabel(r'$\mathbb{P}(M = \downarrow) $')

# adjust limits of plot as necessary given parameters from earlier
if beginning:
    plt.xlim(0, 0.44)
    plt.ylim(0.95, 1)
else:
    plt.xlim(0.56, 1)
    plt.ylim(0, 0.05)

# include legend
plt.legend()

# get figure and save plot with appropriate file name before showing plot
fig = plt.gcf()
if beginning:
    save_title = "20230904_qubit_response_linewidth_3_beginning_true"
else:
    save_title = "20230904_qubit_response_linewidth_3_beginning_false"
fig.savefig(save_title, dpi = 300, bbox_inches = 'tight')
plt.show()
