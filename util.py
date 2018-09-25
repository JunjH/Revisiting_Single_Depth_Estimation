#original script: https://github.com/fangchangma/sparse-to-dense/blob/master/utils.lua
	
import torch
import math
import numpy as np

def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    nanMask = getNanMask(target)
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement


def evaluateError(output, target):
    errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
              'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    _output, _target, nanMask, nValidElement = setNanToZero(output, target)

    if (nValidElement.data.cpu().numpy() > 0):
        diffMatrix = torch.abs(_output - _target)

        errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement

        errors['MAE'] = torch.sum(diffMatrix) / nValidElement

        realMatrix = torch.div(diffMatrix, _target)
        realMatrix[nanMask] = 0
        errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

        LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
        LG10Matrix[nanMask] = 0
        errors['LG10'] = torch.sum(LG10Matrix) / nValidElement

        yOverZ = torch.div(_output, _target)
        zOverY = torch.div(_target, _output)

        maxRatio = maxOfTwo(yOverZ, zOverY)

        errors['DELTA1'] = torch.sum(
            torch.le(maxRatio, 1.25).float()) / nValidElement
        errors['DELTA2'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
        errors['DELTA3'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

        errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
        errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
        errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
        errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
        errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
        errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
        errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

    return errors


def addErrors(errorSum, errors, batchSize):
    errorSum['MSE']=errorSum['MSE'] + errors['MSE'] * batchSize
    errorSum['ABS_REL']=errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
    errorSum['LG10']=errorSum['LG10'] + errors['LG10'] * batchSize
    errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

    errorSum['DELTA1']=errorSum['DELTA1'] + errors['DELTA1'] * batchSize
    errorSum['DELTA2']=errorSum['DELTA2'] + errors['DELTA2'] * batchSize
    errorSum['DELTA3']=errorSum['DELTA3'] + errors['DELTA3'] * batchSize

    return errorSum


def averageErrors(errorSum, N):
    averageError={'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                    'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    averageError['MSE'] = errorSum['MSE'] / N
    averageError['ABS_REL'] = errorSum['ABS_REL'] / N
    averageError['LG10'] = errorSum['LG10'] / N
    averageError['MAE'] = errorSum['MAE'] / N

    averageError['DELTA1'] = errorSum['DELTA1'] / N
    averageError['DELTA2'] = errorSum['DELTA2'] / N
    averageError['DELTA3'] = errorSum['DELTA3'] / N

    return averageError





	
