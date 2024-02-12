from tqdm import tqdm

import torch
from tools import Accumulator
import config


def data_to_device(data):
    if isinstance(data, tuple):
        return tuple(d.to(config.device) for d in data)
    if isinstance(data, list):
        return list(d.to(config.device) for d in data)
    return data.to(config.device)


def train_epoch(model, calc_loss, data, opt, measurements=None, callbacks=None, batch_acc:int = 1):
    if callbacks is None:
        callbacks = []
    
    model.train(True)
    opt.zero_grad()  # Zero the gradients

    for i, (inputs, targets) in enumerate(data):
        # Move data to device
        inputs = data_to_device(inputs)
        targets = data_to_device(targets)

        predicted = model(inputs)  # Get prediction from the model

        if calc_loss is not None: # Hebbian pre-training does not require loss
            loss = calc_loss(predicted, targets) / batch_acc # Calculate the loss
            loss.backward()

        if (i+1) % batch_acc == 0 or i+1 == len(data):
            opt.step()
            opt.zero_grad()

        [cb(model, loss, opt) for cb in callbacks]

        if measurements is not None:
            # No need for backpropagation when measuring performance on train data
            with torch.no_grad():
                [m(predicted, targets) for m in measurements]
    

def validate_epoch(model, data, measurements):
    model.eval()
    with torch.no_grad():
        for inputs, targets in data:
            inputs = data_to_device(inputs)
            targets = data_to_device(targets)
            
            predicted = model(inputs)
            [m(predicted, targets) for m in measurements]


def train(model, data, opt, calc_loss=None, val_data=None, epochs=1,
          train_measurements=None,
          val_measurements=None,
          epoch_callbacks=None,
          train_batch_callbacks=None,
          learning_rate_schedule=None,
          recover_checkpoint=None,
          batch_acc:int = 1):

    if train_measurements is None:
        train_measurements = []

    if val_measurements is None:
        val_measurements = []

    train_meas = [Accumulator(f'train_{m}', m) for m in train_measurements]
    if calc_loss is not None:
        train_meas.insert(0, Accumulator('train_loss', calc_loss))

    val_meas = []
    if val_data is not None:
        val_meas = [Accumulator(f'val_{m}', m) for m in val_measurements]
        if calc_loss is not None:
            val_meas.insert(0, Accumulator('val_loss', calc_loss))

    start_epoch = 0

    if recover_checkpoint is not None and recover_checkpoint.exists():
        print(f'initialising from checkpoint {recover_checkpoint}')
        checkpoint = torch.load(recover_checkpoint, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.device)
        
        if checkpoint['opt_state_dict'] is not None:
           opt.load_state_dict(checkpoint['opt_state_dict'])
        if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
            learning_rate_schedule.load_state_dict(checkpoint['lr_scheduler'])

        train_meas = checkpoint['train_meas']
        val_meas = checkpoint['val_meas']
        start_epoch = checkpoint['epoch']

        checkpoint = None # dereference
        
    else:
        print('starting new experiment')
        model = model.to(config.device)

    for e in tqdm(range(start_epoch, epochs)):

        train_epoch(model, calc_loss, data, opt, train_meas, callbacks=train_batch_callbacks, batch_acc=batch_acc)
        
        if learning_rate_schedule is not None:
            learning_rate_schedule.step()
        
        [m.end_epoch() for m in train_meas]
        
        if val_data is not None:
            validate_epoch(model, val_data, val_meas)
            [m.end_epoch() for m in val_meas]

        [ec(model, opt, e, train_meas, val_meas, learning_rate_schedule) for ec in epoch_callbacks]

    result = dict()
    for m in train_meas:
        result[m.name] = m.history
    for m in val_meas:
        result[m.name] = m.history
    return result, model.to('cpu')
