import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
from functools import partial
import math
import os
import shutil
import sys

# from torchdiffeq import odeint_adjoint

sys.path.append(r'../')
from TorchDiffEqPack.odesolver_mem import odesolve_adjoint_sym12, odesolve_adjoint 
# from models.resnet import ResNet18
from models.sqnxt import SqNxt_23_1x

import warnings
import torch
import torch.nn as nn
from torchdiffeq._impl.odeint import SOLVERS, odeint
from torchdiffeq._impl.misc import _check_inputs, _flat_to_shape
from torchdiffeq._impl.misc import _mixed_norm


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)

            if event_fn is None:
                y = ans
                ctx.save_for_backward(t, y, *adjoint_params)
            else:
                event_t, y = ans
                ctx.save_for_backward(t, y, event_t, *adjoint_params)

        return ans

    @staticmethod
    def backward(ctx, *grad_y):
        with torch.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.
            event_mode = ctx.event_mode
            if event_mode:
                t, y, event_t, *adjoint_params = ctx.saved_tensors
                _t = t
                t = torch.cat([t[0].reshape(-1), event_t.reshape(-1)])
                grad_y = grad_y[1]
            else:
                t, y, *adjoint_params = ctx.saved_tensors
                grad_y = grad_y[0]

            adjoint_params = tuple(adjoint_params)

            ##################################
            #      Set up initial state      #
            ##################################

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
            aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
            def augmented_dynamics(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                y = y_aug[1]
                adj_y = y_aug[2]
                # ignore gradients wrt time and parameters

                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    y = y.detach().requires_grad_(True)

                    # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                    # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                    # wrt t here means we won't compute that if we don't need it.
                    func_eval = func(t if t_requires_grad else t_, y)

                    # Workaround for PyTorch bug #39784
                    _t = torch.as_strided(t, (), ())  # noqa
                    _y = torch.as_strided(y, (), ())  # noqa
                    _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + adjoint_params, -adj_y,
                        allow_unused=True, retain_graph=True
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)]

                return (vjp_t, func_eval, vjp_y, *vjp_params)

            ##################################
            #       Solve adjoint ODE        #
            ##################################

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time.
                aug_state = odeint(
                    augmented_dynamics, tuple(aug_state),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
                aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0] = aug_state[0]

            # Only compute gradient wrt initial time when in event handling mode.
            if event_mode and t_requires_grad:
                time_vjps = torch.cat([time_vjps[0].reshape(-1), torch.zeros_like(_t[1:])])

            adj_y = aug_state[2]
            adj_params = aug_state[3:]

        return (None, None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)


def odeint_adjoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
                         "`options` has been passed then `adjoint_options` must be passed as well.")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    else:
        # Avoid in-place modifying a user-specified dict.
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    if len(adjoint_params) != oldlen_:
        # Some params were excluded.
        # Issue a warning if a user-specified norm is specified.
        if 'norm' in adjoint_options and callable(adjoint_options['norm']):
            warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                          "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")

    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    # Handle the adjoint norm function.
    state_norm = options["norm"]
    handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    ans = OdeintAdjointMethod.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    """In-place modifies the adjoint options to choose or wrap the norm function."""

    # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        # `adjoint_options` was not explicitly specified by the user. Use the default norm.
        adjoint_options["norm"] = default_adjoint_norm
    else:
        # `adjoint_options` was explicitly specified by the user...
        try:
            adjoint_norm = adjoint_options['norm']
        except KeyError:
            # ...but they did not specify the norm argument. Back to plan A: use the default norm.
            adjoint_options['norm'] = default_adjoint_norm
        else:
            # ...and they did specify the norm argument.
            if adjoint_norm == 'seminorm':
                # They told us they want to use seminorms. Slight modification to plan A: use the default norm,
                # but ignore the parameter state
                def adjoint_seminorm(tensor_tuple):
                    t, y, adj_y, *adj_params = tensor_tuple
                    # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
                    return max(t.abs(), state_norm(y), state_norm(adj_y))
                adjoint_options['norm'] = adjoint_seminorm
            else:
                # And they're using their own custom norm.
                if shapes is None:
                    # The state on the forward pass was a tensor, not a tuple. We don't need to do anything, they're
                    # already going to get given the full adjoint state as (t, y, adj_y, adj_params)
                    pass  # this branch included for clarity
                else:
                    # This is the bit that is tuple/tensor abstraction-breaking, because the odeint machinery
                    # doesn't know about the tupled nature of the forward state. We need to tell the user's adjoint
                    # norm about that ourselves.

                    def _adjoint_norm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        y = _flat_to_shape(y, (), shapes)
                        adj_y = _flat_to_shape(adj_y, (), shapes)
                        return adjoint_norm((t, *y, *adj_y, *adj_params))
                    adjoint_options['norm'] = _adjoint_norm

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)#nn.GroupNorm(planes//16, planes) #nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)#nn.GroupNorm(planes//16, planes)#nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)#nn.GroupNorm(self.expansion*planes//16, self.expansion*planes)#
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super(BasicBlock2, self).__init__()
        in_planes = dim
        planes = dim
        stride = 1
        self.nfe = 0
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) #nn.GroupNorm(planes//16, planes)#
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) #nn.GroupNorm(planes//16, planes)#

        self.shortcut = nn.Sequential()

    def forward(self,t, x):
        self.nfe += 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, ODEBlock_ = None):
        super(ResNet, self).__init__()
        self._planes = 64
        self.in_planes = self._planes
        self.ODEBlock = ODEBlock_
        self.dummy = torch.nn.Parameter(torch.ones(64,32,32))

        self.conv1 = nn.Conv2d(3, self._planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self._planes)
#         self.layer1_1 = self._make_layer(self._planes, 1, stride=1)
        self.layer1_1 = self._make_layer2(self._planes, num_blocks[0]-1, stride=1)
        self.layer1_2 = self._make_layer2(self._planes, num_blocks[0]-1, stride=1)

#         self.layer2_1 = self._make_layer(self._planes, 1, stride=2)
        self.layer2_1 = self._make_layer2(self._planes, num_blocks[1]-1, stride=1)
        self.layer2_2 = self._make_layer2(self._planes, num_blocks[1]-1, stride=1)

#         self.layer3_1 = self._make_layer(self._planes, 1, stride=2)
        self.layer3_1 = self._make_layer2(self._planes, num_blocks[2]-1, stride=1)
        self.layer3_2 = self._make_layer2(self._planes, num_blocks[2]-1, stride=1)

#         self.layer4_1 = self._make_layer(self._planes, 1, stride=2)
        self.layer4_1 = self._make_layer2(self._planes, num_blocks[3]-1, stride=1)
        self.layer4_2 = self._make_layer2(self._planes, num_blocks[3]-1, stride=1)
        self.linear = nn.Linear(self._planes * block.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, planes, num_blocks, stride):
        return self.ODEBlock(BasicBlock2(self.in_planes))
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         #layers.append(nn.BatchNorm2d(self.in_planes))
#         for stride in strides:
#             layers.append(self.ODEBlock(BasicBlock2(self.in_planes)))
#         return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
#         out, dummy = self.layer1_1(out, torch.broadcast_to(self.dummy, out.shape), name="layer1_1")
#         out, dummy = self.layer1_2(out, dummy, name="layer1_2")
#         out, dummy = self.layer2_1(out, dummy, name="layer2_1")
#         out, dummy = self.layer2_2(out, dummy, name="layer2_2")
#         out, dummy = self.layer3_1(out, dummy, name="layer3_1")
#         out, dummy = self.layer3_2(out, dummy, name="layer3_2")
#         out, dummy = self.layer4_1(out, dummy, name="layer4_1")
#         out, dummy = self.layer4_2(out, dummy, last_module=True, name="layer4_2")
        out = self.layer1_1(out)
        out = self.layer1_2(out)#, dummy, name="layer1_2")
        out = self.layer2_1(out)#, dummy, name="layer2_1")
        out = self.layer2_2(out)#, dummy, name="layer2_2")
        out = self.layer3_1(out)#, dummy, name="layer3_1")
        out = self.layer3_2(out)#, dummy, name="layer3_2")
        out = self.layer4_1(out)#, dummy, name="layer4_1")
        out = self.layer4_2(out)#, dummy, last_module=True, name="layer4_2")
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(ODEBlock):
      return ResNet(BasicBlock, [2,2,2,2], ODEBlock_ = ODEBlock)


# from torch_ode_cifar import odesolve
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 60:
        optim_factor = 2
    elif epoch > 30:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'sqnxt'], default='resnet')
parser.add_argument('--use_ode', type=str2bool, default=True, help='Use Neural ODE blocks or not')
parser.add_argument('--method', type=str, choices=['Euler', 'RK2', 'RK4', 'RK23', 'Sym12Async', 'RK12','Dopri5'], default='Sym12Async')
parser.add_argument('--num_epochs', type=int, default=90)
parser.add_argument('--start_epoch', type=int, default=0)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint_mem', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--h', type=float, default=None, help='Initial Stepsize')
parser.add_argument('--t0', type=float, default=1.0, help='Initial time')
parser.add_argument('--t1', type=float, default=0.0, help='End time')
parser.add_argument('--rtol', type=float, default=1e-1, help='Releative tolerance')
parser.add_argument('--atol', type=float, default=1e-2, help='Absolute tolerance')
parser.add_argument('--print_neval', type=bool, default=False, help='Print number of evaluation or not')
parser.add_argument('--neval_max', type=int, default=50000, help='Maximum number of evaluation in integration')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--use_wandb', action='store_true',
                        help='ranking within the nodes')
parser.set_defaults(use_wandb=False)
args = parser.parse_args()
if args.network == 'sqnxt':
    writer = SummaryWriter(
        'sqnxt/' + args.network + '_mem_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')
elif args.network == 'resnet':
    writer = SummaryWriter(
        'resnet/' + args.network + '_mem_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = int(args.start_epoch)
batch_size = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")
print("="*30)
print("is_use_cuda:", is_use_cuda)
print(args)
print("="*30)

if args.use_wandb:
    import wandb
    wandb.login()
    args.wnb = wandb.init(project="cs780_final", entity="tst008", tags=[sys.argv[0], args.method])

class ODEBlock(nn.Module):

    def __init__(self, odefunc, odesolve_func):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = args.method
        self.integration_time = torch.tensor([args.t0, args.t1], dtype=torch.float32)
        self.rtol = args.rtol
        self.atol = args.atol
        self.options = {}
        self.options.update({'step_size': args.h})

    def forward(self, x):
#         out, step = self.odesolve_func(self.odefunc, x, self.options)
#         print(x.shape)
#         print(self.integration_time, self.method.lower(), self.rtol, self.atol)
        out = odeint_adjoint(self.odefunc, x, self.integration_time.to(x), method=self.method.lower(), rtol=self.rtol, atol=self.atol, options=self.options, )
#         print(step)
        return out[-1] # last time point only

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        
class IdentityBlock(nn.Module):

    def __init__(self, odefunc):
        super(IdentityBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        print("Using identity (no ode)")

    def forward(self, x):
        return x

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', transform=transform_train, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', transform=transform_test, train=False, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=False)

block = None
if args.use_ode:
    if args.method == "Sym12Async":
        block = partial(ODEBlock, odesolve_func=odesolve_adjoint_sym12)
    else:
        block = partial(ODEBlock, odesolve_func=odesolve_adjoint)
else:
    block = IdentityBlock
if args.network == 'sqnxt':
    net = SqNxt_23_1x(10, block)
elif args.network == 'resnet':
    net = ResNet18(block)
print("Number of trainable params:", sum(p.numel() for p in net.parameters() if p.requires_grad))

net.apply(conv_init)
print(net)

if args.use_wandb:
    args.wnb.watch(net)
if is_use_cuda:
    net.cuda()  # to(device)
    net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print('Training Epoch: #%d, LR: %.4f' % (epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

        optimizer.step()
        writer.add_scalar('Train/Loss', loss.item(), epoch * 50000 + batch_size * (idx + 1))
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()

        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(train_dataset) // batch_size,
                            train_loss / (batch_size * (idx + 1)), correct / total))
        sys.stdout.flush()
    if args.use_wandb:
        args.wnb.log(
            {
                "epoch": epoch,
                'train_loss': train_loss / (batch_size * len(train_loader)), #TODO
                'train_acc': correct / total,
            }
        )
    writer.add_scalar('Train/Accuracy', correct / total, epoch)


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = net(inputs)
            
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        writer.add_scalar('Test/Loss', loss.item(), epoch * 50000 + test_loader.batch_size * (idx + 1))

        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size,
                            test_loss / (100 * (idx + 1)), correct / total))
        sys.stdout.flush()

    acc = correct / total
    if args.use_wandb:
        args.wnb.log(
            {
                "epoch": epoch,
                'test_loss': test_loss / (100 * len(test_loader)), 
                'test_acc': acc,
            }
        )
    writer.add_scalar('Test/Accuracy', acc, epoch)
    return acc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


best_acc = 0.0

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()

    _lr = lr_schedule(args.lr, _epoch)
    adjust_learning_rate(optimizer, _lr)

    train(_epoch)
    print()
    test_acc = test(_epoch)
    print()
    print()
    end_time = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': _epoch + 1,
        'state_dict': net.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=args.checkpoint + '_' + args.method + '_' + args.network)

print('Best Acc@1: %.4f' % (best_acc * 100))
writer.close()

if args.use_wandb:    
    args.wnb.finish()
