import torch
import argparse
import sys
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
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
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.data import IterLoader
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
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
from reid.evaluation_metrics import accuracy

CHECK = 1e-5
SAT_MIN = 0.5
MODE = "bilinear"


def get_data(sourceName, mteName, split_id, data_dir, height, width,
             batch_size, workers, combine,num_instances=8):
    root = osp.join(data_dir, sourceName)
    rootMte = osp.join(data_dir, mteName)
    sourceSet = datasets.create(sourceName, root, num_val=0.1, split_id=split_id)
    mteSet = datasets.create(mteName, rootMte, num_val=0.1, split_id=split_id)

    num_classes = sourceSet.num_trainval_ids if combine else sourceSet.num_train_ids
    class_meta = mteSet.num_trainval_ids if combine else mteSet.num_train_ids

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    defen_train_transformer = T.Compose([
        Resize((height, width)),
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])
    meta_train_loader = DataLoader(
        Preprocessor(sourceSet.trainval,  root=sourceSet.images_dir,
                     transform=defen_train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(sourceSet.trainval, num_instances),
        pin_memory=True, drop_last=True)

    meta_test_loader=DataLoader(
        Preprocessor(mteSet.trainval, root=mteSet.images_dir,
                     transform=defen_train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(mteSet.trainval, num_instances),
        pin_memory=True, drop_last=True)

    sc_test_loader = DataLoader(
        Preprocessor(list(set(sourceSet.query) | set(sourceSet.gallery)),
                     root=sourceSet.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return sourceSet, mteSet, num_classes, meta_train_loader, meta_test_loader,sc_test_loader,class_meta


def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m

def test(dataset, net, perturbation, args, evaluator, epoch, name,saveRank=False):
    print(">> Evaluating network on test datasets...")
    net = net.cuda()
    net.eval()
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    mean = torch.Tensor(normalize.mean).view(1, 3, 1, 1).cuda()
    std = torch.Tensor(normalize.std).view(1, 3, 1, 1).cuda()
    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor()
    ])
    query_loader = DataLoader(
        Preprocessor(dataset.query, name, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, name, root=dataset.images_dir, transform=test_transformer),
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
            ###normalize####
            inputs = inputs.cuda()
            inputs = (inputs - mean) / std
            gFeats.append(net(inputs)[0])
            gnames.extend(gname)
            gCams.append(gCam.cuda())
        gFeats = torch.cat(gFeats, 0)
        gCams, qCams = torch.cat(gCams).view(1, -1), torch.cat(qCams).view(-1, 1)
    distMat = calDist(qFeats, gFeats)
    # evaluate on test datasets
    s= evaluator.evaMat(distMat, dataset.query, dataset.gallery)
    return testQImage, noiseQIamge,s



class Trainer(object):
    def __init__(self, model, criterions, print_freq=1):
        super(Trainer, self).__init__()
        self.model = model
        self.criterions = criterions
        self.print_freq = print_freq
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, epoch,mt_train_loader, mt_test_loader, optimizer,noise_model,args):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        for i ,(inputs) in enumerate(mt_train_loader):
            meta_input = mt_test_loader.next()
            img, _, pid, _ = inputs
            metaTest, _, meta_pid, _ = meta_input

            adv_inputs_total, adv_labels, adv_labels_total, coupled_inputs = [], [], [], []
            adv_inputs_total_meta, adv_labels_meta, adv_labels_total_meta, coupled_inputs_meta = [], [], [], []

            ###generate perturbed images during meta train##
            adv_data = create_attack_exp(inputs, noise_model)
            adv_inputs, adv_labels, adv_idxs, og_adv_inputs = adv_data
            adv_inputs_total.append(adv_inputs)
            adv_labels_total.append(adv_labels)
            coupled_inputs.append(og_adv_inputs)

            inputs = torch.cat([img.cuda()] + [_.data for _ in adv_inputs_total], dim=0)
            labels = torch.cat([pid.cuda()] + [_.data for _ in adv_labels_total], dim=0)

            inputs, pid = Variable(inputs), Variable(labels)
            finall_input = inputs.cuda()
            targets = pid.cuda()

            ###generate perturbed images during meta test##
            adv_data = create_attack_exp(meta_input, noise_model)
            adv_inputs, adv_labels, adv_idxs, og_adv_inputs = adv_data
            adv_inputs_total_meta.append(adv_inputs)
            adv_labels_total_meta.append(adv_labels)
            coupled_inputs_meta.append(og_adv_inputs)

            meta_input = torch.cat([metaTest.cuda()] + [_.data for _ in adv_inputs_total_meta], dim=0)
            meta_pid = torch.cat([meta_pid.cuda()] + [_.data for _ in adv_labels_total_meta], dim=0)

            meta_input = meta_input.cuda()
            meta_pid = meta_pid.cuda()

            data_time.update(time.time() - end)

            ###meta train####
            # cur_model=self.model
            output = self.model(finall_input)
            loss, prec1 = self._memory(output, targets, epoch)
            self.model.zero_grad()

            grads = torch.autograd.grad(loss, (self.model.module.params()), create_graph=True)
            lr = optimizer.param_groups[0]["lr"]
            lr_base = optimizer.param_groups[1]["lr"]

            ###meta test###
            newMeta = models.create('resMeta', num_classes=class_meta)
            newMeta.copyModel(self.model.module)
            newMeta.update_params(lr_inner=lr, lr_base=lr_base, source_params=grads[:-2], solver='adam')
            del grads
            newMeta = nn.DataParallel(newMeta).to(self.device)

            meta_out = newMeta(meta_input)
            metaloss, prec2 = self._memory(meta_out, meta_pid, epoch)

            ######
            loss_finall = metaloss + loss

            optimizer.zero_grad()
            loss_finall.backward()
            optimizer.step()

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(mt_train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ ,_,_= inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _memory(self, outputs, targets, epoch):
        # outputs = self.model(inputs)
        # new added by wc
        # x1 triplet loss
        if isinstance(outputs[0], list):
            loss_tri = sum([self.criterions[0](val, targets, epoch)[0] for val in outputs[0]]) / len(outputs[0])
            prec_tri = sum([self.criterions[0](val, targets, epoch)[1] for val in outputs[0]]) / len(outputs[0])
            # x2 global feature cross entropy loss
            loss_global = sum([self.criterions[1](val, targets) for val in outputs[1]])
            prec_global, = accuracy(outputs[1][0].data, targets.data)
        else:

            loss_tri, prec_tri = self.criterions[0](outputs[0], targets, epoch)
            # x2 global feature cross entropy loss
            loss_global = self.criterions[1](outputs[1], targets)
            prec_global, = accuracy(outputs[1].data, targets.data)
        prec_global = prec_global[0]
        return loss_tri,prec_global

        # return loss_tri + loss_global, prec_global


def create_attack_exp(data,attack_obj,proportion_attacked=0.5):
    imgs,fname,pid,cam=data
    imgs=imgs.cuda()
    pid=pid.cuda()
    num_elements=imgs.shape[0]
    selected_idxs = sorted(random.sample(list(range(num_elements)),int(proportion_attacked * num_elements)))
    selected_idxs = imgs.new(selected_idxs).long()
    if selected_idxs.numel() == 0:
        return (None, None, None)
    adv_inputs = Variable(imgs.index_select(0, selected_idxs))
    adv_examples =attack_obj(adv_inputs)
    pre_adv_labels = pid.index_select(0, selected_idxs)
    return(adv_examples,pre_adv_labels,selected_idxs,adv_inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a ResNet-50 trained on Imagenet '
                    'against ReColorAdv'
    )

    parser.add_argument('--data', type=str, required=True,
                        help='path to reid dataset')
    parser.add_argument('--noise_path1', type=str,
                        default='.', help='path to reid dataset')
    parser.add_argument('-s', '--source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-m', '--mte', type=str, default='personx',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=64, required=True,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
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
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument("--max-eps", default=8, type=int, help="max eps")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--noise_type', type=int, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
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
    args = parser.parse_args()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    sourceSet,mteSet, num_classes,meta_train_loader, meta_test_loader,sc_test_loader,class_meta = \
        get_data(args.source, args.mte,
                 args.split, args.data, args.height,
                 args.width, args.batch_size, 8, args.combine_trainval)

    model = models.create('resMeta', num_classes=num_classes)
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

    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

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

    if args.noise_type==0:
        noise = combined_threat()

    elif args.noise_type==1:
        noise = recoloradv_threat()

    elif args.noise_type == 2:
        noise = additive_threat()

    ####criterion
    criterion = [
        TripletLoss(0.5, args.num_instances, False).cuda(),
        nn.CrossEntropyLoss().cuda()
    ]
    # multi lr
    base_param_ids = set(map(id, model.module.base.params()))
    new_params = [p for p in model.module.params() if id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.params(), 'lr_mult': 1.0},
        {'params': new_params, 'lr_mult': 3.0}]
    # Optimizer
    optimizer_defense = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 40 else \
            args.lr * (0.1 ** ((epoch - 40) / 40.0))
        for g in optimizer_defense.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    ###defense_trainer
    trainer = Trainer(model, criterion, args.print_freq)
    import time
    checkpoint1 = torch.load(args.noise_path1)
    noise.load_state_dict(checkpoint1)

    start_epoch = best_top1 = 0
    if args.resume_reid:
        checkpoint = load_checkpoint(args.resume_reid)
        model.load_state_dict(checkpoint['state_dict'])
        best_top1 = checkpoint['best_top1']
        start_epoch = checkpoint['epoch'] - 1
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    meta_test_loader=IterLoader(meta_test_loader,length=200)
    import random

    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, meta_train_loader, meta_test_loader, optimizer_defense, noise, args)
        meta_test_loader.new_epoch()
        if epoch < args.start_save:
            continue
        if (epoch+1) % 5 == 0:
            print("eval on current attack ")
            _, _, rank_score = test(sourceSet, model, noise, args, evaSrc, epoch, args.source)
            ###eval raw result
            print("eval on new model ")
            s = evaSrc.evaluate(sc_test_loader, sourceSet.query, sourceSet.gallery)
            top1 = rank_score.map
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, epoch , fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, top1, best_top1, ' *' if is_best else ''))