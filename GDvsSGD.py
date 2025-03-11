import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy 

##############################################################################
#                 0) landscape of loss function for checking non-convexity        #
##############################################################################

def plot_loss_landscape(x, y, filename, w_min=-5.0, w_max=5.0, num_points=200):
    """
    x: (N,1) torch tensor of inputs
    y: (N,1) torch tensor of targets
    w_min, w_max: range over which to scan w
    num_points: number of discrete w-values in [w_min, w_max]
    """
    # We'll sample w-values over some range
    w_values = torch.linspace(w_min, w_max, steps=num_points)
    
    losses = []
    for w in w_values:
        # sin(w * x) => shape (N,1)
        pred = torch.sin(w * x)
        # L(w) = sum_i [sin(w*x_i) - y_i]^2
        loss = ((pred - y)**2).sum().item()
        losses.append(loss)
    
    # Convert w_values to numpy for plotting:
    w_values_np = w_values.numpy()
    losses_np = losses  # already a Python list of floats

    # Plot
    plt.figure()
    plt.plot(w_values_np, losses_np, label=r"$L(w) = \sum_i [\sin(w\,x_i) - y_i]^2$")
    plt.xlabel("w")
    plt.ylabel("Loss L(w)")
    plt.title("Loss Landscape vs. w")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=150)
    
##############################################################################
#                 1) Model classes: SinModel, MLP_Scalar, MLP_Vector + CosineLoss        #
##############################################################################
class CosineLoss(nn.Module):
    """
    Computes sum of (1 - cos_similarity) over all samples.
    If you prefer the mean, replace the sum(...) with mean(...).
    """
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # pred, target: shape [N, dim]
        # cos_val: shape [N], each entry is cos-sim between pred[i] and target[i]
        cos_val = F.cosine_similarity(pred, target, dim=1)  # in [-1,1]
        loss_per_sample = 1.0 - cos_val  # want to minimize 1 - cos => maximize cos

        if self.reduction == "sum":
            return loss_per_sample.sum()
        elif self.reduction == "mean":
            return loss_per_sample.mean()
        else:
            # fallback
            return loss_per_sample

class SinModel(nn.Module):
    """
    A simple parameterized function f_{w,1}(x) = sin(w * x).
    We'll store w as an nn.Parameter (initialized to 0).
    """
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return torch.sin(self.w * x)


class MLP_Scalar(nn.Module):
    """
    An MLP that outputs a single scalar, then takes sigmoid of it.
    Used for f_{w,2}(x) = Sigmoid(MLP(x)).
    By default, input_dim=4. No bias, just to keep things simple.
    """
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        
    def forward(self, x):
        # MLP -> single scalar -> apply sigmoid
        return torch.sigmoid(self.net(x))


class MLP_Vector(nn.Module):
    """
    An MLP that outputs a vector of the same dimension as x,
    then returns its L2 norm => f_{w,3}(x) = ||MLP(x)||.
    By default, input_dim=4. No bias.
    """
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )
        
    def forward(self, x):
        out_vec = self.net(x)              # shape [N, input_dim]
        return out_vec.norm(dim=1, keepdim=True)


def init_last_layer_zero(model):
    """
    Zero out the parameters in the last Linear layer (if it exists),
    so the MLP starts out producing a constant output (often 0).
    For tasks with Sigmoid, that yields ~0.5 at the beginning, etc.
    """
    if not hasattr(model, 'net'):
        return
    last_layer = model.net[-1]
    if isinstance(last_layer, nn.Linear):
        with torch.no_grad():
            last_layer.weight.fill_(0.0)
            if last_layer.bias is not None:
                last_layer.bias.fill_(0.0)

##############################################################################
#                     2) Data generation for each “task”                     #
##############################################################################

def generate_data_sin(N, seed=None, w_range=(-4.0*np.pi, 4.0*np.pi)):
    """
    Generate data for f_{w,1}(x) = sin(w*x), 
    with w randomly sampled from w_range.
    We still produce a 1D x here.
    """
    if seed is not None:
        np.random.seed(seed)
    w_true = np.random.uniform(*w_range)
    
    x_np = np.random.uniform(-2, 2, size=(N,))
    y_np = np.sin(w_true * x_np)
    
    x = torch.from_numpy(x_np).float().view(-1, 1)
    y = torch.from_numpy(y_np).float().view(-1, 1)
    return x, y


def generate_data_mlp_scalar(N, seed=None, input_dim=4, hidden_dim=16):
    """
    Generate data for f_{w,2}(x) = Sigmoid(MLP(x)), 
    where the ground-truth MLP is randomly initialized.
    Now the input dimension is 4 by default.
    """
    if seed is not None:
        np.random.seed(seed)
    # Create a random “ground truth” MLP
    gt_model = MLP_Scalar(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # Sample x of dimension input_dim
    x_np = np.random.uniform(-2.0, 2.0, size=(N, input_dim))
    x = torch.from_numpy(x_np).float()
    
    # Evaluate y = Sigmoid(gt_model(x)) (already built into MLP_Scalar.forward)
    with torch.no_grad():
        y = gt_model(x)
    return x, y


def generate_data_mlp_vector(N, seed=None, input_dim=4, hidden_dim=16):
    """
    Generate data for f_{w,3}(x) = ||MLP(x)||,
    where the ground-truth MLP is randomly initialized.
    Now the input dimension is 4 by default.
    """
    if seed is not None:
        np.random.seed(seed)
    # Create a random “ground truth” MLP
    gt_model = MLP_Vector(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # Sample x of dimension input_dim
    x_np = np.random.uniform(-2.0, 2.0, size=(N, input_dim))
    x = torch.from_numpy(x_np).float()
    
    # Evaluate y = ||gt_model(x)||
    with torch.no_grad():
        y = gt_model(x)
    return x, y

##############################################################################
#            3) train_for_one_epoch: does one epoch of GD or SGD            #
##############################################################################

def train_for_one_epoch(model, loss_fn, x, y, lr, method='GD', batch_size=None):
    """
    Performs exactly *one epoch* of training on (x, y).
    
    If method='GD', do 1 update with the entire dataset (batch_size = N).
    If method='SGD', split data into mini-batches and do multiple updates.

    Returns the final loss on the entire dataset at the end of the epoch.
    """
    model.train()
    
    if method == 'GD':
        # Single step on the entire dataset
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        pred = model(x)
        loss = loss_fn(pred, y)
        #print('GD')
        #print(x,y)
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
                #print(p)

        final_loss = loss.item()

    else:
        # method = 'SGD'
        N = x.size(0)
        if batch_size is None or batch_size > N:
            batch_size = N

        indices = torch.randperm(N)
        start = 0
        while start < N:
            end = start + batch_size
            mb_idx = indices[start:end]

            x_mb = x[mb_idx]
            y_mb = y[mb_idx]

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            pred = model(x_mb)
            loss = loss_fn(pred, y_mb)
            #print('SGD')
            #print(x,y)
            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
            
            start = end

    # Evaluate final loss on entire dataset
    with torch.no_grad():
        final_loss = loss_fn(model(x), y).item()

    return final_loss


##############################################################################
# 4) experiment_f: multiple epochs, average across repeats, store in dicts  #
##############################################################################

def experiment_f(
    data_gen_func,
    create_train_model_func,
    loss_fn,
    N=64,
    num_repeats=3,
    num_epochs=5,
    lr_list=(1e-3, 1e-2, 1e-1),
    batch_size_list=(1, 8, 64)
):
    """
    Runs multiple training epochs for each hyperparameter setting, repeated 
    multiple times for random data (and random ground-truth if relevant).

    - data_gen_func(N, seed) => (x, y)
    - create_train_model_func() => returns a *fresh* model (with init)
    - N: number of samples
    - num_repeats: how many times to repeat with different random seeds
    - num_epochs: how many epochs to train
    - lr_list: learning rates to test
    - batch_size_list: mini-batch sizes to test (for SGD)

    We'll store results in dictionaries with the structure:
      gd_losses = {lr: {epoch: avg_loss}}
      sgd_losses = {batch_size: {lr: {epoch: avg_loss}}}

    Returns:
       gd_losses, sgd_losses
    """

    # We will store *per-epoch* losses across repeats in a caching structure,
    # then average them at the end.

    # For GD:  gd_cache[lr][epoch_idx] -> list of losses over repeats
    gd_cache = {lr: [[] for _ in range(num_epochs)] for lr in lr_list}

    # For SGD: sgd_cache[bs][lr][epoch_idx] -> list of losses over repeats
    sgd_cache = {
        bs: {lr: [[] for _ in range(num_epochs)] for lr in lr_list}
        for bs in batch_size_list
    }

    for repeat_i in range(num_repeats):
        # Generate fresh data for each repeat
        x, y = data_gen_func(N, seed=repeat_i)
        init_model = create_train_model_func()
        if init_model.__class__.__name__ == "SinModel":
            filename = 'sin_'+str(repeat_i)+'.png'
            plot_loss_landscape(x,y,filename)
        # =========== For each LR, do GD for multiple epochs ===========
        for lr in lr_list:
            # Make a fresh model for each repeat
            model_gd = copy.deepcopy(init_model)#create_train_model_func()
            for epoch_idx in range(num_epochs):
                # Just before training, print what we are about to do:
                print(f"[Repeat {repeat_i+1}] GD, lr={lr}, epoch={epoch_idx+1}")
                #print('model_gd:', model_gd.w)
                final_loss = train_for_one_epoch(
                    model=model_gd,
                    loss_fn=loss_fn,
                    x=x,
                    y=y,
                    lr=lr,
                    method='GD',
                )
                #print('model_gd:', model_gd.w, final_loss)
                # Save the final loss for this epoch
                gd_cache[lr][epoch_idx].append(final_loss)

        # ========== For each BS and LR, do SGD for multiple epochs ==========
        for bs in batch_size_list:
            for lr in lr_list:
                # Fresh model for each repeat
                model_sgd = copy.deepcopy(init_model)#create_train_model_func()
                #print('model_sgd:', model_sgd.w)
                for epoch_idx in range(num_epochs):
                    print(f"[Repeat {repeat_i+1}] SGD, lr={lr}, bs={bs}, epoch={epoch_idx+1}")

                    final_loss = train_for_one_epoch(
                        model=model_sgd,
                        loss_fn=loss_fn,
                        x=x,
                        y=y,
                        lr=lr,
                        method='SGD',
                        batch_size=bs
                    )
                    #print('model_sgd:', model_sgd.w, final_loss)
                    # Save the final loss
                    sgd_cache[bs][lr][epoch_idx].append(final_loss)

    # Now we average across the repeats for each (lr, epoch) or (bs, lr, epoch).
    gd_losses = {lr: {} for lr in lr_list}
    for lr in lr_list:
        for epoch_idx in range(num_epochs):
            avg_loss = float(np.mean(gd_cache[lr][epoch_idx]))
            # 1-based epoch indexing if you want => epoch=epoch_idx+1
            gd_losses[lr][epoch_idx+1] = avg_loss

    sgd_losses = {bs: {lr: {} for lr in lr_list} for bs in batch_size_list}
    for bs in batch_size_list:
        for lr in lr_list:
            for epoch_idx in range(num_epochs):
                avg_loss = float(np.mean(sgd_cache[bs][lr][epoch_idx]))
                sgd_losses[bs][lr][epoch_idx+1] = avg_loss

    return gd_losses, sgd_losses


##############################################################################
#                   5) Example usage & demonstration code                    #
##############################################################################

if __name__ == "__main__":
    N = 64
    num_repeats = 16
    num_epochs = 64
    lr_list = [1e-0, 1e-2, 1e-4]
    batch_size_list = [1, 8, 64]
    # Example:  f_{w,1}(x) = sin(w*x)
    def create_train_sin():
        return SinModel()  # w=0 init
    sin_loss_fn = nn.MSELoss(reduction='sum')
    gd_sin, sgd_sin = experiment_f(
        data_gen_func=generate_data_sin,
        create_train_model_func=create_train_sin,
        loss_fn=sin_loss_fn,
        N=N,
        num_repeats=num_repeats,
        num_epochs=num_epochs,
        lr_list=lr_list,
        batch_size_list=batch_size_list
    )
    with open("gd_sin.pkl", "wb") as f:
        pickle.dump(gd_sin, f)
    with open("sgd_sin.pkl", "wb") as f:
        pickle.dump(sgd_sin, f)

    # Example 2: f_{w,2}(x) = sigmoid(MLP(x)), with input_dim=4
    def create_train_mlp_scalar():
        model = MLP_Scalar(input_dim=4, hidden_dim=8)
        init_last_layer_zero(model)
        return model
    bce_loss_fn = nn.BCELoss(reduction='sum')
    gd_scalar, sgd_scalar = experiment_f(
        data_gen_func=lambda N, seed: generate_data_mlp_scalar(
            N, seed=seed, input_dim=4, hidden_dim=8
        ),
        create_train_model_func=create_train_mlp_scalar,
        loss_fn=bce_loss_fn,
        N=N,
        num_repeats=num_repeats,
        num_epochs=num_epochs,
        lr_list=lr_list,
        batch_size_list=batch_size_list
    )
    with open("gd_scalar.pkl", "wb") as f:
        pickle.dump(gd_scalar, f)
    with open("sgd_scalar.pkl", "wb") as f:
        pickle.dump(sgd_scalar, f)

    # Example 3: f_{w,3}(x) = ||MLP(x)||, with input_dim=4
    def create_train_mlp_vector():
        model = MLP_Vector(input_dim=4, hidden_dim=8)
        init_last_layer_zero(model)
        return model
    cosine_loss_fn = nn.MSELoss(reduction='sum') #CosineLoss(reduction='sum')
    gd_vector, sgd_vector = experiment_f(
        data_gen_func=lambda N, seed: generate_data_mlp_vector(
            N, seed=seed, input_dim=4, hidden_dim=8
        ),
        create_train_model_func=create_train_mlp_vector,
        loss_fn=cosine_loss_fn,
        N=N,
        num_repeats=num_repeats,
        num_epochs=num_epochs,
        lr_list=lr_list,
        batch_size_list=batch_size_list
    )
    with open("gd_vector.pkl", "wb") as f:
        pickle.dump(gd_vector, f)
    with open("sgd_vector.pkl", "wb") as f:
        pickle.dump(sgd_vector, f)
