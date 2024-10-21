
import copy
import logging
from scipy import io
import numpy as np
import torch
from train_utils import SAM, disable_running_stats, enable_running_stats, save_checkpoint, validate

def matrix_weights_time():
    # Replace 'your_file.mat' with the path to your .mat file
    mat_contents = io.loadmat('quantization/4bit.mat')

    data = mat_contents["ww_mdn"]

    # Determine the column indices to select (odd indices starting from 9)
    start_index = 10  # MATLAB index 911 corresponds to Python index 9
    odd_indices = np.arange(start_index, data.shape[0], 2)  # Generates 11, 13, ...

    # Create the new matrix with only the selected columns
    new_matrix = data[odd_indices,:]
    return new_matrix #returns matrix with temporal values from 10 to 40 microsiemens (shape: 4x8). Value 0 is always the same

def update_drift_in_model(model, drift_values):

    for name, module in model.named_modules():
        # Check if the module has a 'drift' attribute
        if hasattr(module, 'drift_w') and 'shortcut' not in name:
            # Update the 'drift' value
            if drift_values is not None:
                module.drift_w = torch.tensor(drift_values,requires_grad=False).to('cuda')
            else:
                module.drift_w = None # no drift

def validate_drift(test_loader, model, device):
    archive = matrix_weights_time() 
    np.delete(archive, 1, axis=1) # remove the second column
    accs = []
    for timestep in range(7):
        logging.info(f"Validating with drift at timestep {timestep}")
        drift_values = archive[:, timestep] / 1e-5
        update_drift_in_model(model, drift_values)
        accs.append(validate(test_loader, model, device, print_freq=100))
    return sum(accs) / len(accs), accs

def train_with_drift(train_loader, val_loader, num_epochs, model, device, optimizer, criterion, scheduler, log, ckpt_path=None, label_smoothing=0.1):
        
        model.to(device)
        best_model = copy.deepcopy({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}) #initialize best model with the first model
        archive=matrix_weights_time()
        np.delete(archive, 1, axis=1) # remove the second column
        for epoch in range(num_epochs):
            model.train()
            log.train(model, optimizer, len_dataset=len(train_loader))

            for (inputs,targets) in train_loader:
                # sample random drift value
                drift = np.random.randint(archive.shape[1])
                update_drift_in_model(model, archive[:, drift] / 1e-5)
                #inputs = F.interpolate(inputs, size=180, mode='bicubic', align_corners=False)
                inputs, targets = inputs.to(device), targets.to(device)

                # first forward-backward step
                if isinstance(optimizer, SAM):
                    enable_running_stats(model)
                else:
                    optimizer.zero_grad()
                    
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()

                if not isinstance(optimizer, SAM):
                    optimizer.step()
                else:
                    optimizer.first_step(zero_grad=True)
                    # second forward-backward step
                    disable_running_stats(model)
                    criterion(model(inputs), targets).backward()
                    optimizer.second_step(zero_grad=True)

                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets
                    log(model, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])
                    scheduler.step()
            
            update_drift_in_model(model, None)
            model.eval()
            log.eval(model, optimizer, len_dataset=len(val_loader))

            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = (b.to(device) for b in batch)
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
                curr_loss=log.epoch_state["loss"] / log.epoch_state["steps"]
                if curr_loss < log.best_loss: 
                    best_model = copy.deepcopy({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()})


        log.flush(model, optimizer)
        model.load_state_dict(best_model['state_dict']) # load best model for inference 
        optimizer.load_state_dict(best_model['optimizer']) # load optim for further training 
        
        if ckpt_path is not None:
            save_checkpoint(model, optimizer, ckpt_path)
        
        top1=log.best_accuracy*100
        return top1, model, optimizer