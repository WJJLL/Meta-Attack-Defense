import torch
import argparse
import sys
import os
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision import transforms

# ReColorAdv
sys.path.append(os.path.abspath('reColorAdv'))
sys.path.append(os.path.abspath('mister_ed'))
from reid import models
from torch.nn import functional as F
import os.path as osp
from reid import datasets
from reid.utils.data import transforms as T
from torchvision.transforms import Resize
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.evaluators import Evaluator
from torch.optim.optimizer import Optimizer, required
import random
import math
from reid.evaluators import extract_features
from sklearn.cluster import KMeans
from reid.utils.meters import AverageMeter
import torchvision
import numpy as np
import shutil
import adversarial_perturbations as ap
import perturbations as pt
import color_transformers as ct
import color_spaces as cs
from torch import nn
import collections
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.data import IterLoader
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)
random.seed(123)

CHECK = 1e-5
# CHECK = 1e-3
SAT_MIN = 0.5
MODE = "bilinear"


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets,perturbed_feature):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mat_similarity = targets.expand(n, n).eq(targets.expand(n, n).t()).float()

        # sorted_mat_distance, positive_indices = torch.sort(dist + (100000.0) * (1 - mat_similarity), dim=1,
        #                                                    descending=False)

        sorted_mat_distance, positive_indices = torch.sort(dist + (-100000.0) * (1 - mat_similarity), dim=1,
                                                           descending=True)

        hard_p = sorted_mat_distance[:, 0]
        hard_p_indice = positive_indices[:, 0]

        ##最远的一个negative_indices
        sorted_mat_distance, negative_indices = torch.sort(dist + (-100000.0) * mat_similarity, dim=1,
                                                           descending=True)
        hard_n = sorted_mat_distance[:, 0]
        hard_n_indice = negative_indices[:, 0]

        hard_p_feature=inputs[hard_p_indice,:]
        hard_n_feature = inputs[hard_n_indice, :]

        loss = 10 * F.triplet_margin_loss(perturbed_feature, hard_n_feature, hard_p_feature, 0.5)

        return loss


def get_data(sourceName, mteName, targetName, split_id, data_dir, height, width,
             batch_size, workers, combine):
    root = osp.join(data_dir, sourceName)
    rootMte = osp.join(data_dir, mteName)
    rootTgt = osp.join(data_dir, targetName)
    sourceSet = datasets.create(sourceName, root, num_val=0.1, split_id=split_id)
    mteSet = datasets.create(mteName, rootMte, num_val=0.1, split_id=split_id)
    tgtSet = datasets.create(targetName, rootTgt, num_val=0.1, split_id=split_id)
    num_classes = sourceSet.num_trainval_ids if combine else sourceSet.num_train_ids
    class_tgt = tgtSet.num_trainval_ids if combine else tgtSet.num_train_ids

    train_transformer = T.Compose([
        Resize((height, width)),
        T.ToTensor(),
    ])
    meta_train = DataLoader(
        Preprocessor(sourceSet.trainval, root=sourceSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers,  sampler=RandomIdentitySampler(sourceSet.trainval, 8), pin_memory=True)
    meta_test = DataLoader(
        Preprocessor(mteSet.trainval, root=mteSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers,  sampler=RandomIdentitySampler(mteSet.trainval, 8), pin_memory=True)

    return sourceSet, tgtSet, mteSet, num_classes, class_tgt, meta_train, meta_test


def keepGradUpdate(perturbation_model, optimizer, gradInfo):

    weight_decay = optimizer.param_groups[0]["weight_decay"]
    momentum = optimizer.param_groups[0]["momentum"]
    dampening = optimizer.param_groups[0]["dampening"]
    nesterov = optimizer.param_groups[0]["nesterov"]
    lr = optimizer.param_groups[0]["lr"]

    checkpoint = perturbation_model.state_dict()
    keys = list(checkpoint.keys())
    for i,(k,noiseData) in enumerate(perturbation_model.state_dict().items()):
        d_p=gradInfo[i]
        if optimizer.param_groups[0]["sign"]:
            d_p = d_p / (d_p.norm(1) + 1e-12)
        if weight_decay != 0:
            d_p.add_(weight_decay, noiseData)
        if momentum != 0:
            param_state = optimizer.state[noiseData]
            if "momentum_buffer" not in param_state:
                buf = param_state["momentum_buffer"] = torch.zeros_like(noiseData.data)
                buf = buf * momentum + d_p
            else:
                buf = param_state["momentum_buffer"]
                buf = buf * momentum + (1 - dampening) * d_p
            if nesterov:
                d_p = d_p + momentum * buf
            else:
                d_p = buf
            noiseData = noiseData - lr * d_p.sign()
        checkpoint[keys[i]] = noiseData

    perturbation_model.load_state_dict(checkpoint)
    return perturbation_model


class MI_SGD(Optimizer):
    def __init__(
            self,
            params,
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            max_eps=10 / 255,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign=False,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MI_SGD, self).__init__(params, defaults)
        self.sat = 0
        self.sat_prev = 0
        self.max_eps = max_eps

    def __setstate__(self, state):
        super(MI_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["sign"]:
                    d_p = d_p / (d_p.norm(1) + 1e-12)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if group["sign"]:
                    p.data.add_(-group["lr"], d_p.sign())
                else:
                    p.data.add_(-group["lr"], d_p)
        return loss

def trainMeta(meta_train_loader, meta_test_loader,net, epoch, normalize, perturbation):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mean = torch.Tensor(normalize.mean).view(1, 3, 1, 1).cuda()
    std = torch.Tensor(normalize.std).view(1, 3, 1, 1).cuda()

    net.eval()
    end = time.time()
    perturbation.zero_grad()
    optimizer.zero_grad()

    for i, (input, _, pids, _) in enumerate(meta_train_loader):
        metaTest, _, mtepids, _ = meta_test_loader.next()

        data_time.update(time.time() - end)
        model.zero_grad()
        input = input.cuda()
        metaTest = metaTest.cuda()

        # one step update
        with torch.no_grad():
            norm_output = (input - mean) / std
            feature = net(norm_output)[0]

        current_noise=perturbation
        perturted_input = current_noise(input)
        perturted_input_clamp = torch.clamp(perturted_input, 0, 1)
        perturted_input_norm = (perturted_input_clamp - mean) / std
        perturbed_feature = net(perturted_input_norm)[0]

        optimizer.zero_grad()
        loss= TripletLoss()(feature, pids.cuda(), perturbed_feature)

        # maml one step
        noise=perturbation.parameters()
        grad = torch.autograd.grad(loss, noise, create_graph=True)
        noiseOneStep = keepGradUpdate(perturbation, optimizer, grad)
        perturbation_new=noiseOneStep

        #maml test
        with torch.no_grad():
            normMte = (metaTest - mean) / std
            mteFeat = net(normMte)[0]

        perMteInput = perturbation_new(metaTest)
        perMteInput = torch.clamp(perMteInput, 0, 1)
        normPerMteInput = (perMteInput - mean) / std
        normMteFeat = net(normPerMteInput)[0]

        mteloss=TripletLoss()(mteFeat, mtepids.cuda(), normMteFeat)

        finalLoss=loss+mteloss
        finalLoss.backward()

        losses.update(loss.item())
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                ">> Train: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})"
                    .format(
                    epoch + 1,
                    i,
                    len(meta_train),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses
                )
            )
    print(f"Train {epoch}: Loss: {losses.avg}")
    perturbation.state_dict().requires_grad = False
    return losses.avg, perturbation

def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def test(dataset, net, perturbation, args, evaluator, epoch, dataName):
    print(">> Evaluating network on test datasets...")
    net = net.cuda()
    net.eval()
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    mean = torch.Tensor(normalize.mean).view(1, 3, 1, 1).cuda()
    std = torch.Tensor(normalize.std).view(1, 3, 1, 1).cuda()
    query_trans = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor()
    ])
    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor()
    ])
    query_loader = DataLoader(
        Preprocessor(dataset.query, dataName, root=dataset.images_dir, transform=query_trans),
        batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, dataName, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True
    )

    qFeats, gFeats, testQImage,noiseQIamge, qnames, gnames = [], [], [], [], [],[]
    qCams, gCams = [], []

    import random
    with torch.no_grad():
        for batch_index, (inputs, qname, _, qCam) in enumerate(query_loader):
            inputs = inputs.cuda()
            perturted_input = perturbation(inputs)
            successful_diffs = ((perturted_input - inputs) * 3 + 0.5).clamp(0, 1)
            if len(testQImage) < 8:
                testQImage.append(perturted_input[0, ...])
                noiseQIamge.append(successful_diffs[0, ...])
            perturted_input = torch.clamp(perturted_input, 0, 1)
            ###normalization
            norm_perturted_input = (perturted_input - mean) / std
            perturbed_feature = net(norm_perturted_input)[0]
            qFeats.append(perturbed_feature)
            qnames.extend(qname)
            qCams.append(qCam.cuda())
        qFeats = torch.cat(qFeats, 0)
        for (inputs, gname, _, gCam) in gallery_loader:
            inputs = inputs.cuda()
            inputs = (inputs - mean) / std
            gFeats.append(net(inputs)[0])
            gnames.extend(gname)
            gCams.append(gCam.cuda())
        gFeats = torch.cat(gFeats, 0)
        gCams, qCams = torch.cat(gCams).view(1, -1), torch.cat(qCams).view(-1, 1)
    distMat = calDist(qFeats, gFeats)
    # evaluate on test datasets
    evaluator.evaMat(distMat, dataset.query, dataset.gallery)
    return testQImage, noiseQIamge


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a ResNet-50 trained on Imagenet '
                    'against ReColorAdv'
    )

    parser.add_argument('--data', type=str, required=True,
                        help='path to reid dataset')
    parser.add_argument('--noise_path', type=str,
                        default='.', help='path to reid dataset')
    parser.add_argument('-s', '--source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-t', '--target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-m', '--mte', type=str, default='personx',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=64, required=True,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')

    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--resumeTgt', type=str, default='', metavar='PATH')
    parser.add_argument('--noise_resume', type=str, default='', metavar='PATH')

    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--noise_type', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument("--lp", default=8, type=int, help="max eps")
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument("--number", default=1, type=int, help="number")

    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    use_gpu = torch.cuda.is_available()
    args = parser.parse_args()

    sourceSet, tgtSet, mteSet, num_classes, class_tgt, meta_train, meta_test = \
        get_data(args.source, args.mte, args.target,
                 args.split, args.data, args.height,
                 args.width, args.batch_size, 8, args.combine_trainval)

    model = models.create(args.arch, pretrained=True, num_classes=num_classes)
    modelTest = models.create(args.arch, pretrained=True, num_classes=class_tgt)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        try:
            model.load_state_dict(checkpoint)
        except:
            allNames = list(checkpoint.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkpoint[name]
            model.load_state_dict(checkpoint, strict=False)
        checkTgt = torch.load(args.resumeTgt)
        if 'state_dict' in checkTgt.keys():
            checkTgt = checkTgt['state_dict']
        try:
            modelTest.load_state_dict(checkTgt)
        except:
            allNames = list(checkTgt.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkTgt[name]
            modelTest.load_state_dict(checkTgt, strict=False)

    model.eval()
    modelTest.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        modelTest = modelTest.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        modelTest = nn.DataParallel(modelTest)


    evaluator = Evaluator(modelTest, args.print_freq)
    evaSrc = Evaluator(model, args.print_freq)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    MAX_EPS = args.lp / 255.0

    ##define  advColor_noise
    recoloradv_threat = ap.ThreatModel(pt.ReColorAdv, {
        'xform_class': ct.FullSpatial,
        'cspace': cs.RGBColorSpace(),  # controls the color space used
        'lp_style': 'inf',
        'lp_bound': [args.lp/255, args.lp/255, args.lp/255],  # [epsilon_1, epsilon_2, epsilon_3]
        'xform_params': {
            'resolution_x': 25,  # R_1
            'resolution_y': 25,  # R_2
            'resolution_z': 25,  # R_3
        },
        'use_smooth_loss': False,
    })
    additive_threat = ap.ThreatModel(ap.DeltaAddition, {
        'lp_style': 'inf',
        'lp_bound': args.lp/255,
    })
    combined_threat = ap.ThreatModel(
        ap.SequentialPerturbation,
        [recoloradv_threat, additive_threat],
        ap.PerturbationParameters(norm_weights=[1.0, 0.0]),
    )

    if args.noise_type == 0:
        noise = combined_threat()
    elif args.noise_type == 1:
        noise = recoloradv_threat()
    else:
        noise = additive_threat()

    optimizer = MI_SGD(
        [
            {"params": noise.parameters(), "lr": MAX_EPS / 10, "momentum": 1, "sign": True}
        ],
        max_eps=MAX_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))

    import time

    meta_test = IterLoader(meta_test, length=200)

    for epoch in range(args.epoch):
        scheduler.step()
        begin_time = time.time()
        ####get feature of specfic layer ::source_layers[x][1][1]
        loss, noise_model = trainMeta(
            meta_train,meta_test, model, epoch,normalize, noise
        )
        meta_test.new_epoch()

        if epoch % 5 == 4:
            if not os.path.exists(args.noise_resume):
                os.makedirs(args.noise_resume)
            PATH = args.noise_resume + '/best_perturbation' + str(epoch) + '.pth'
            torch.save(noise_model.state_dict(), PATH)

            print('On Target...\n')
            testQImage, noiseIamge = test(tgtSet, modelTest, noise_model, args, evaluator, epoch,
                                          args.target)
            print('On Source...\n')
            testQ = test(sourceSet, model, noise_model, args, evaSrc, epoch, args.source)





