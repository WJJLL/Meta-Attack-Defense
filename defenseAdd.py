import torch
import argparse
import sys
import os
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torch.autograd import Variable
from torchvision import transforms
# from ssim import pytorch_ssim
from reid.dist_metric import DistanceMetric
from reid.utils.data import IterLoader
# sys.path.append(os.path.abspath('ssim'))
sys.path.append(os.path.abspath('mister_ed'))
sys.path.append(os.path.abspath('reColorAdv'))
# ReColorAdv
from reid import models
from torch.nn import functional as F
import os.path as osp
from reid import datasets
from reid.utils.data import transforms as T
from torchvision.transforms import Resize
from reid.utils.data.preprocessor import Preprocessor
from reid.evaluators import Evaluator
from reid.loss import TripletLoss
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.evaluation_metrics import accuracy

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
import modelsMate


CHECK = 1e-5
# CHECK = 1e-3
SAT_MIN = 0.5
MODE = "bilinear"



def get_data(sourceName, mteName, targetName, split_id, data_dir, height, width,
             batch_size, workers, combine,num_instances=8):
    root = osp.join(data_dir, sourceName)
    rootMte = osp.join(data_dir, mteName)
    rootTgt = osp.join(data_dir, targetName)
    sourceSet = datasets.create(sourceName, root, num_val=0.1, split_id=split_id)
    mteSet = datasets.create(mteName, rootMte, num_val=0.1, split_id=split_id)
    tgtSet = datasets.create(targetName, rootTgt, num_val=0.1, split_id=split_id)
    num_classes = sourceSet.num_trainval_ids if combine else sourceSet.num_train_ids
    class_tgt = tgtSet.num_trainval_ids if combine else tgtSet.num_train_ids
    class_meta = mteSet.num_trainval_ids if combine else mteSet.num_train_ids

    class_mix=class_meta+num_classes

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        Resize((height, width)), T.ToTensor()
    ])
    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(sourceSet.trainval,
                     root=sourceSet.images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    defen_train_transformer = T.Compose([
        Resize((height, width)),
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])

    curSet = [(osp.join(sourceSet.images_dir, val[0]), val[1], val[2]) for val in sourceSet.trainval]

    for val in mteSet.trainval:
        curSet.append((osp.join(mteSet.images_dir, val[0]), num_classes + int(val[1]), int(val[2])))

    add_train_loader = DataLoader(
        Preprocessor(curSet,  root=None,
                     transform=defen_train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(curSet, num_instances),
        pin_memory=True, drop_last=True)

    sc_test_loader = DataLoader(
        Preprocessor(list(set(sourceSet.query) | set(sourceSet.gallery)),
                     root=sourceSet.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return sourceSet, tgtSet, mteSet, num_classes, class_tgt, train_loader,add_train_loader,sc_test_loader,class_meta,class_mix


def rescale_check(check, sat, sat_change, sat_min):
    return sat_change < check and sat > sat_min


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

    def rescale(self, ):
        for group in self.param_groups:
            if not group["sign"]:
                continue
            for p in group["params"]:
                self.sat_prev = self.sat
                self.sat = (p.data.abs() >= self.max_eps).sum().item() / p.data.numel()
                sat_change = abs(self.sat - self.sat_prev)
                if rescale_check(CHECK, self.sat, sat_change, SAT_MIN):
                    print('rescaled')
                    p.data = p.data / 2

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


def train(train_loader, net, epoch, centroids, normalize, perturbation, optimizer):
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

    for i, (input, fname, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        model.zero_grad()
        input = input.cuda()
        with torch.no_grad():
            norm_output = (input - mean) / std
            feature = net(norm_output)[0]
            scores = F.normalize(feature.t(), p=2, dim=0)

            realLab = scores.max(0, keepdim=True)[1]
            _, ranks = torch.sort(scores, dim=0, descending=True)

        perturted_input = perturbation(input)

        perturted_input_clamp = torch.clamp(perturted_input, 0, 1)
        perturted_input_norm = (perturted_input_clamp - mean) / std

        perturbed_feature = net(perturted_input_norm)[0]

        fakePred = perturbed_feature

        oneHotReal = torch.zeros(scores.t().shape).cuda()
        oneHotReal.scatter_(1, realLab.view(-1, 1), float(1))
        label_loss = F.relu(
            (fakePred * oneHotReal).sum(1).mean() - (fakePred * (1 - oneHotReal)).max(1)[0].mean()
        )

        smoothAP = Loss(perturbed_feature, i)
        loss = smoothAP + label_loss

        optimizer.zero_grad()
        torch.autograd.backward(loss)
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
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses
                )
            )
    print(f"Train {epoch}: Loss: {losses.avg}")
    return losses.avg, perturbation


def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def test(dataset, net, perturbation, args, evaluator, epoch, name, saveRank=False):
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
        Preprocessor(dataset.query, name, root=dataset.images_dir, transform=query_trans),
        batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, name, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True
    )

    qFeats, gFeats, testQImage, noiseQIamge, qnames, gnames = [], [], [], [], [], []
    qCams, gCams = [], []

    import random
    with torch.no_grad():

        for batch_index, (inputs, qname, _, qCam) in enumerate(query_loader):
            inputs = inputs.cuda()
            norm_input = (inputs - mean) / std
            ##save origin_image
            perturted_input = perturbation(inputs)
            successful_diffs = ((perturted_input - inputs) * 3 + 0.5).clamp(0, 1)
            if len(testQImage) < 8:
                testQImage.append(perturted_input[0, ...])
                noiseQIamge.append(successful_diffs[0, ...])
            perturted_input = torch.clamp(perturted_input, 0, 1)
            ###normalization
            norm_perturted_input = (perturted_input - mean) / std
            ##save perturted_image
            perturbed_feature = net(norm_perturted_input)[0]
            qFeats.append(perturbed_feature)
            qnames.extend(qname)
            qCams.append(qCam.cuda())
        qFeats = torch.cat(qFeats, 0)
        for (inputs, gname, _, gCam) in gallery_loader:
            ###normalize####
            inputs = inputs.cuda()
            inputs = (inputs - mean) / std
            gFeats.append(net(inputs)[0])
            gnames.extend(gname)
            gCams.append(gCam.cuda())
        gFeats = torch.cat(gFeats, 0)
        gCams, qCams = torch.cat(gCams).view(1, -1), torch.cat(qCams).view(-1, 1)
    distMat = calDist(qFeats, gFeats)

    if saveRank:
        import cv2
        _, ind = torch.sort(distMat, 1)
        ind = ind[0, :8]
        # import ipdb
        # ipdb.set_trace()
        results = [osp.join(gallery_loader.dataset.root, gnames[val]) for val in ind]
        allNames = [osp.join(query_loader.dataset.root, qnames[0])] + results
        isCorrMaske = [1 if int(qnames[0].split('/')[-1].split('_')[0]) == int(val.split('/')[-1].split('_')[0])
                       else 0 for val in results]
        isCorrMaske = [0] + isCorrMaske
        # imshow
        ranklist = []
        for ii, (i_name, mask) in enumerate(zip(allNames, isCorrMaske)):
            img = cv2.imread(i_name)
            img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_LINEAR)
            if ii != 0:
                img = cv2.rectangle(img, (0, 0), (64, 128),
                                    (0, 255, 0) if mask == 1 else (0, 0, 255), 2)
            ranklist.append(img)
            if ii == 0:
                ranklist.append(np.zeros((128, 10, 3)))
        ranklist = np.concatenate(ranklist, 1)
        cv2.imwrite(f'{name}{epoch}.jpg', ranklist)
    # evaluate on test datasets
    s = evaluator.evaMat(distMat, dataset.query, dataset.gallery)
    return testQImage, noiseQIamge, s




def create_attack_exp(inputs,attack_obj,proportion_attacked=0.2):
    inputs,fname,pid,cam=inputs
    inputs=inputs.cuda()
    pid=pid.cuda()
    num_elements=inputs.shape[0]
    selected_idxs = sorted(random.sample(list(range(num_elements)),int(proportion_attacked * num_elements)))
    selected_idxs = inputs.new(selected_idxs).long()
    if selected_idxs.numel() == 0:
        return (None, None, None)
    adv_inputs = Variable(inputs.index_select(0, selected_idxs))
    adv_examples =attack_obj(adv_inputs)
    pre_adv_labels = pid.index_select(0, selected_idxs)
    return(adv_examples,pre_adv_labels,selected_idxs,adv_inputs)

def save_noise(noise, epoch, image):
    if not os.path.exists(args.noise_path):
        os.makedirs(args.noise_path)
    filename = os.path.join(args.noise_path, "noise_%d" % epoch)
    afterName = os.path.join(args.noise_path, "after_%d" % epoch)
    torchvision.utils.save_image(noise, filename + ".png", normalize=True)
    torchvision.utils.save_image(image, afterName + ".png", normalize=True)

class Trainer(object):
    def __init__(self, model, criterions, print_freq=1):
        super(Trainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq

    def train(self, epoch, data_loader, optimizer, pertutation,args, regularize_adv_scale=None):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        for i, inputs in enumerate(data_loader):

            data_time.update(time.time() - end)

            img, _, pid, _ = inputs

            adv_inputs_total, adv_labels, adv_labels_total, coupled_inputs = [], [], [], []

            adv_data = create_attack_exp(inputs, pertutation,args.p)
            adv_inputs, adv_labels, adv_idxs, og_adv_inputs = adv_data
            adv_inputs_total.append(adv_inputs)
            adv_labels_total.append(adv_labels)
            coupled_inputs.append(og_adv_inputs)

            inputs = torch.cat([img.cuda()] + [_.data for _ in adv_inputs_total], dim=0)
            labels = torch.cat([pid.cuda()] + [_.data for _ in adv_labels_total], dim=0)
            adv_inputs = torch.cat(coupled_inputs, dim=0)
            adv_examples = torch.cat(adv_inputs_total, dim=0)

            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            loss, prec1 = self._forward(inputs, labels, epoch,i)
            losses.update(loss.item(), pid.size(0))
            precisions.update(prec1, pid.size(0))
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, epoch,i=0):
        outputs = self.model(inputs)
        # new added by wc
        # x1 triplet loss
        if isinstance(outputs[0], list):
            loss_tri = sum([self.criterions[0](val, targets, epoch)[0] for val in outputs[0]]) / len(outputs[0])
            prec_tri = sum([self.criterions[0](val, targets, epoch)[1] for val in outputs[0]]) / len(outputs[0])
            # x2 global feature cross entropy loss
            import ipdb
            ipdb.set_trace()
            loss_global = sum([self.criterions[1](val, targets) for val in outputs[1]])
            prec_global, = accuracy(outputs[1][0].data, targets.data)
        else:
            loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
            # x2 global feature cross entropy loss

            loss_global = self.criterions[1](outputs[1], targets)
            prec_global, = accuracy(outputs[1].data, targets.data)
        prec_global = prec_global[0]

        return loss_tri + loss_global, prec_global


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
    parser.add_argument('--print_freq', type=int, default=50
                        )
    parser.add_argument("--max-eps", default=8, type=int, help="max eps")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--pre_dir', type=str, default='modelsMate', help='path to be attacked model', metavar='PATH')
    parser.add_argument('--targetmodel', type=str, default='lsro', choices=modelsMate.get_names())

    # optimizer
    parser.add_argument('--lr', type=float, default=0.0003,
                        help="learning rate of all parameters")
    parser.add_argument('--p',type=float,default=0.2)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume_reid', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=85)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--noise_type', type=int, default=0)
    args = parser.parse_args()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    sourceSet, tgtSet, mteSet, num_classes, class_tgt, train_loader, add_train_loader, sc_test_loader, class_meta ,class_mix= \
        get_data(args.source, args.mte, args.target,
                 args.split, args.data, args.height,
                 args.width, args.batch_size, 8, args.combine_trainval)

    model = models.create(args.arch, pretrained=True, num_classes=class_mix)
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

    model.eval()
    modelTest.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        modelTest = modelTest.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        modelTest = nn.DataParallel(modelTest)

    centroids=None


    evaluator = Evaluator(modelTest, args.print_freq)
    evaSrc = Evaluator(model, args.print_freq)

    ##define  advColor_noise
    recoloradv_threat = ap.ThreatModel(pt.ReColorAdv, {
        'xform_class': ct.FullSpatial,
        'cspace': cs.RGBColorSpace(),  # controls the color space used
        'lp_style': 'inf',
        'lp_bound': [8 / 255, 8 / 255, 8 / 255],  # [epsilon_1, epsilon_2, epsilon_3]
        'xform_params': {
            'resolution_x': 25,  # R_1
            'resolution_y': 25,  # R_2
            'resolution_z': 25,  # R_3
        },
        'use_smooth_loss': False,
    })
    additive_threat = ap.ThreatModel(ap.DeltaAddition, {
        'lp_style': 'inf',
        'lp_bound': 8 / 255,
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

    ####criterion
    criterion = [
        TripletLoss(0.5, args.num_instances, False).cuda(),
        nn.CrossEntropyLoss().cuda()
    ]
    # multi lr
    base_param_ids = set(map(id, model.base.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [
        {'params': model.base.parameters(), 'lr_mult': 1.0},
        {'params': new_params, 'lr_mult': 3.0}]
    # Optimizer
    optimizer_defense = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)


    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 40 else \
            args.lr * (0.1 ** ((epoch - 40) / 40.0))
        for g in optimizer_defense.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)


    MAX_EPS = args.max_eps / 255.0
    optimizer_att = MI_SGD(
        [
            {"params": noise.parameters(), "lr": MAX_EPS / 10, "momentum": 1, "sign": True}
        ],
        max_eps=MAX_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_att, gamma=math.exp(-0.01))

    ###defense_trainer
    trainer = Trainer(model, criterion, args.print_freq)
    # metric = DistanceMetric(algorithm=args.dist_metric)
    # Loss = SmoothAP(0.01, 64, 8, 2048)

    import time

    # PATH = './checkpoint/best_perturbation_duke.pth'
    checkpoint=torch.load(args.noise_path)
    noise.load_state_dict(checkpoint)

    start_epoch = best_top1 = 0
    if args.resume_reid:
        checkpoint = load_checkpoint(args.resume_reid)
        model.load_state_dict(checkpoint['state_dict'])
        best_top1 = checkpoint['best_top1']
        start_epoch = checkpoint['epoch'] - 1
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    # Loss = SmoothAP(0.01, args.batch_size, 8, 2048)
    defense_reid = []

    # _, _, rank_score = test(sourceSet, model, noise, args, evaSrc, 0, args.source)

    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, add_train_loader, optimizer_defense, noise,args)
        # add_train_loader.new_epoch()
        if epoch < args.start_save:
            continue
        if epoch % 5 == 0 or epoch == args.epochs-1:
            ##eval defense result
            print("eval on current attack ")
            _, _, rank_score = test(sourceSet, model, noise, args, evaSrc, epoch, args.source)
            ###eval raw result
            print("eval on new model ")
            s = evaSrc.evaluate(sc_test_loader, sourceSet.query, sourceSet.gallery)
            ####top-1选择最优的map#####
            # map = rank_score.map
            # epoch_score.append(map)
            # top1 = rank_score.market1501[0]
            top1 = rank_score
            defense_reid.append(rank_score)
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, epoch, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))