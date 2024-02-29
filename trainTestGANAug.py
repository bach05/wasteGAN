import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision
from torch.autograd import Variable

from PIL import Image
import cv2

import time
import os
from tqdm import tqdm
import json

from utils.dataset import SemSegDataset, AugSemSegDataset
from utils.util import save_tensorboard_images, get_images_with_mask

import segmentation_models_pytorch as smp
from utils.metrics import pixel_accuracy, mIoU, get_lr, mIoUMeter, intersectionAndUnionGPU, ConfMatrix
from torch.utils.tensorboard import SummaryWriter
import argparse
import albumentations as A
from utils.util import generate_distinguishable_colors
from torchvision.utils import draw_segmentation_masks

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, writer, device, n_classes, patch=False, args=None):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    aug_p = args.p

    max_miou = 0
    min_loss = 10000
    overfit_flag = False
    overfit_epoch = -1

    fit_time = time.time()
    for e in (range(epochs)):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        miou_meter_train = mIoUMeter()
        miou_meter_val = mIoUMeter()

        conf_mat_2 = ConfMatrix(2)
        conf_mat_n = ConfMatrix(n_classes)

        # training loop
        model.train()
        for i, data in enumerate((train_loader)):
            # training phase
            image_tiles, mask_tiles = data

            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask, n_classes=n_classes)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

            pred_mask = torch.argmax(output, dim=1)
            miou_meter_train.update(pred_mask, mask, n_classes)

            # compute metrics by confusion matrix
            conf_mat_n.update(pred_mask.flatten(), mask.flatten())
            conf_mat_2.update((pred_mask>1).long().flatten(), (mask>1).long().flatten())


        #SAVE IMAGES
        #print("Output: ", output.shape)
        color = generate_distinguishable_colors(args.n_cls)
        masked_images = get_images_with_mask(images=image, masks_logits=output, color=color)
        save_tensorboard_images(images=masked_images, label=f"[aug={aug_p}][TRAIN OUT]", logger=writer, iters = e)
        mask = mask.unsqueeze(1)
        mask = torch.cat([1-mask, mask], dim=1)
        #print("mask: ", mask.shape)
        #print("mask: ", torch.unique)
        masked_images = get_images_with_mask(images=image, masks_logits=mask.float())
        save_tensorboard_images(images=masked_images, label=f"[aug={aug_p}][TRAIN GT]", logger=writer, iters = e, color=color)

 #      else:
        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0

        val_conf_mat_2 = ConfMatrix(2)
        val_conf_mat_n = ConfMatrix(n_classes)

        # validation loop
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # reshape to 9 patches from single image, delete batch size
                image_tiles, mask_tiles = data

                if patch:
                    bs, n_tiles, c, h, w = image_tiles.size()

                    image_tiles = image_tiles.view(-1, c, h, w)
                    mask_tiles = mask_tiles.view(-1, h, w)

                image = image_tiles.to(device)
                mask = mask_tiles.to(device)
                output = model(image)

                # evaluation metrics
                val_iou_score += mIoU(output, mask, n_classes=n_classes)
                test_accuracy += pixel_accuracy(output, mask)
                # loss
                loss = criterion(output, mask)
                test_loss += loss.item()

                pred_mask = torch.argmax(output, dim=1)
                miou_meter_val.update(pred_mask, mask, n_classes)

                # compute metrics by confusion matrix
                val_conf_mat_n.update(pred_mask.flatten(), mask.flatten())
                val_conf_mat_2.update((pred_mask > 1).long().flatten(), (mask > 1).long().flatten())

        # calculation mean error for each batch
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(val_loader))

        # print(f"[{e+1}/{epochs}] \n"
        #       f"TRAIN Loss {running_loss / len(train_loader):.3f} "
        #       f"mIoU {miou_meter_train.get_miou()[1]:.3f}, "
        #       f"mIoU_cm {conf_mat_n.get_metrics()[0]:.3f}, "
        #       f"mIoU_cm2 {conf_mat_2.get_metrics()[0]:.3f} \n"
        #       f"VALI Loss {test_loss / len(val_loader):.3f}, "
        #       f"mIoU {miou_meter_val.get_miou()[1]:.3f}, "
        #       f"mIoU_cm {val_conf_mat_n.get_metrics()[0]:.3f}, "
        #       f"mIoU_cm2 {val_conf_mat_2.get_metrics()[0]:.3f}, "
        #       )

        #np_cm = val_conf_mat_n.get_matrix()
        #class_losses = np_cm.diag() / np_cm.sum(1)
        # print('VAL Pred/GT pixel count ratios:')
        # print('bg: {:3f}'.format(class_losses[0]))
        # print('rigid_plastic: {:3f}'.format(class_losses[1]))
        # print('cardboard: {:3f}'.format(class_losses[2]))
        # print('metal: {}'.format(class_losses[3]))
        # print('soft_plastic: {:3f}'.format(class_losses[4]))

        writer.add_scalars(f"[aug={aug_p}] Loss / Epoch", {'Train': running_loss / len(train_loader), 'Val': test_loss / len(val_loader) }, e)

        current_iou = val_conf_mat_n.get_metrics()[0]

        if current_iou > max_miou:
            max_miou = current_iou
            best_model_state = model.state_dict()
            #print(f'Update best model with mIoU {max_miou:.3f}...')
            torch.save(model.state_dict(), os.path.join(args.prefix, 'test_results', args.train_id, args.model_name+'_best-mIoU.pth'))

        if not overfit_flag:
            val_loss = (test_loss / len(val_loader))
            if val_loss < min_loss:
                min_loss = val_loss
                not_improve = 0 #ADD TO RESET THE COUNT
            else:
                not_improve += 1
                #print(f"VAL loss not improved for {not_improve} epochs.")
                if not_improve == 5:
                    overfit_flag = True
                    overfit_epoch = e
                    #print(f"!!! OVERFITTING WARNING: VAL loss not improved for {not_improve} epochs.")
                    torch.save(model.state_dict(), os.path.join(args.prefix, 'test_results', args.train_id, args.model_name+'_earlystop.pth'))

        # iou
        val_iou.append(val_iou_score / len(val_loader))
        train_iou.append(iou_score / len(train_loader))
        train_acc.append(accuracy / len(train_loader))
        val_acc.append(test_accuracy / len(val_loader))

        writer.add_scalars(f"[aug={aug_p}] Accuracy / Epoch", {'Train': accuracy / len(train_loader), 'Val': test_accuracy / len(val_loader) }, e)
        writer.add_scalars(f"[aug={aug_p}] mIoU / Epoch", {'Train': iou_score / len(train_loader), 'Val': val_iou_score / len(val_loader)}, e)

        #SAVE IMAGES
        #print("Output: ", output.shape)
        masked_images = get_images_with_mask(images=image, masks_logits=output, color=color)
        save_tensorboard_images(images=masked_images, label=f"[aug={aug_p}][VAL OUT]", logger=writer, iters = e)

        mask = mask.unsqueeze(1)
        mask = torch.cat([1-mask, mask], dim=1)
        #print("mask: ", mask.shape)
        #print("mask: ", torch.unique)
        masked_images = get_images_with_mask(images=image, masks_logits=mask.float())
        save_tensorboard_images(images=masked_images, label=f"[aug={aug_p}][VAL GT]", logger=writer, iters = e, color=color)
        

    history = {
                'train_loss': train_losses, 'val_loss': test_losses,
                'train_miou': train_iou, 'val_miou': val_iou,
                'train_acc': train_acc, 'val_acc': val_acc,
                'train_miou_all': miou_meter_train.get_miou()[1], 'val_miou_all': miou_meter_val.get_miou()[1],
                'train_miou_class': list(miou_meter_train.get_miou()[0]), 'val_miou_class': list(miou_meter_val.get_miou()[0]),
                'lrs': lrs,
                'overfit_epoch': overfit_epoch
               }

    # summary =  {'train_loss': np.mean(np.array(train_losses)), 'val_loss':  np.mean(np.array(test_losses)),
    #            'train_miou':  np.mean(np.array(train_iou)), 'val_miou':  np.mean(np.array(val_iou)),
    #            'train_acc':  np.mean(np.array(train_acc)), 'val_acc':  np.mean(np.array(val_acc))
    #             }

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history, best_model_state


def test(model, test_loader, criterion, device, n_classes, patch=False, args=None):

    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_iou_score = 0
    miou_meter = mIoUMeter()

    confusion_matrix_n = ConfMatrix(n_classes)
    confusion_matrix_2 = ConfMatrix(2)

    class_count_preds = np.zeros(n_classes)
    class_count_gts = np.zeros(n_classes)

    # validation loop
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            # reshape to 9 patches from single image, delete batch size
            image_tiles, mask_tiles = data

            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()
                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            output = model(image)
            # evaluation metrics
            test_iou_score += mIoU(output, mask, n_classes=n_classes)
            test_accuracy += pixel_accuracy(output, mask)
            # loss
            loss = criterion(output, mask)
            test_loss += loss.item()

            pred_mask = torch.argmax(output, dim=1)
            miou_meter.update(pred_mask, mask, n_classes)

            # compute metrics by confusion matrix
            confusion_matrix_n.update(pred_mask.flatten(), mask.flatten())
            confusion_matrix_2.update((pred_mask > 1).long().flatten(), (mask > 1).long().flatten())

            # pred_1hot = torch.nn.functional.one_hot(pred_mask, num_classes=n_classes).permute(0, 3, 1, 2).to(
            #     torch.float)
            # mask_1hot = torch.nn.functional.one_hot(mask, num_classes=n_classes).permute(0, 3, 1, 2).to(torch.float)
            #
            # # COUNT CLASS PREDICTIONS
            # B, K, H, W = pred_1hot.shape
            # class_count_pred = pred_1hot.view(B, K, H * W).sum(2).sum(0) / (H * W * B)
            # class_count_gt = mask_1hot.view(B, K, H * W).sum(2).sum(0) / (H * W * B)
            #
            # class_count_preds += class_count_pred.cpu().numpy()
            # class_count_gts += class_count_gt.cpu().numpy()

            ############## Visualization
            # B, C, H, W = image.shape
            # for j in range(B):
            #
            #     plt.subplot(2, 1, 1)
            #     plt.imshow(pred_mask[j].cpu())
            #
            #     # print("Pred: ", pred_mask[j].unique() )
            #
            #     plt.subplot(2, 1, 2)
            #     plt.imshow(mask[j].cpu())
            #     #
            #     # print("GT: ", mask[j].unique())
            #     #
            #     plt.show()

    # print(f"****** TEST  *************************************\n",
    #       f"mIoU     {miou_meter.get_miou()[1]:.3f}, \n"
    #       f"mIoU_cm  {confusion_matrix_n.get_metrics()[0]:.3f}, \n"
    #       f"mIoU_cm2 {confusion_matrix_2.get_metrics()[0]:.3f}, \n"
    #       f"IoU per class    {list(miou_meter.get_miou()[0])}, \n"
    #       f"IoU_cm per class {list(confusion_matrix_n.get_metrics()[1])}, "
    #       )

    np_cm = confusion_matrix_n.get_matrix()
    class_losses = np_cm.diag() / np_cm.sum(1)
    # print('TEST Pred/GT pixel count ratios:')
    # print('bg: {:3f}'.format(class_losses[0]))
    # print('rigid_plastic: {:3f}'.format(class_losses[1]))
    # print('cardboard: {:3f}'.format(class_losses[2]))
    # print('metal: {}'.format(class_losses[3]))
    # print('soft_plastic: {:3f}'.format(class_losses[4]))

    confusion_matrix_n.plot_confusion_matrix(args.cls_names, args.cm_save_path)

    # calculation mean error for each batch
    test_losses = test_loss / len(test_loader)

    # iou
    test_iou = test_iou_score / len(test_loader)
    test_acc = test_accuracy / len(test_loader)

    history = {
                'test_loss': test_losses,
                'test_miou': test_iou,
                'test_acc': test_acc,
                'test_miou_all': miou_meter.get_miou()[1],
                'test_miou_class': list(miou_meter.get_miou()[0]),
                'test_miou_CM': confusion_matrix_n.get_metrics()[0],
                'test_acc_CM': confusion_matrix_n.get_metrics()[2],
                'test_class_ious_CM': list(confusion_matrix_n.get_metrics()[1]),
                'class_counts_gt': list(class_count_gts / len(test_loader)),
                'class_counts_pred': list(class_count_preds / len(test_loader)),
    }

    return history

def main():

    init_time = time.time()

    # Create the parser
    parser = argparse.ArgumentParser()
    
    #Train Params
    parser.add_argument('--batch', type=int, default=16) #16
    parser.add_argument('--real_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2.5e-3) #1e-4 2.5e-3
    parser.add_argument('--wd', type=float, default=5e-4) #1e-5 5e-4

    parser.add_argument('--n_cls', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4) #4


    #Data Loading
    parser.add_argument('--model_step', type=str, default="step_159999")
    parser.add_argument('--train_id', type=str, default="maskGAN_dualD_Drgb") #which dataset to load
    parser.add_argument('--aug_root', type=str, default="AugDatasets")
    parser.add_argument('--split_root', type=str, default="zerowaste-v1")
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--use_imcLoss', action="store_true") #which dataset to load
    parser.add_argument('--use_sharpFeat', action="store_true") #which dataset to load
    parser.add_argument('--use_adaCustom', action="store_true") #which dataset to load
    parser.add_argument('--use_advCustom', action="store_true") #which dataset to load
    parser.add_argument('--use_symLogAct', action="store_true") #which dataset to load
    parser.add_argument('--symLogAct_a', type=int, default=2)
    parser.add_argument('--use_genXL', action="store_true") #which dataset to load
    parser.add_argument('--use_clsBal', action="store_true") #which dataset to load
    
    parser.add_argument('--trainFromScratch', action="store_true") #which dataset to load

    args = parser.parse_args()

    if args.use_genXL: 
        args.train_id += "_genXL"
    if args.use_symLogAct: 
        args.train_id += "_symLogA{}".format(args.symLogAct_a)        
    if args.use_adaCustom: 
        args.train_id += "_adaC"
    if args.use_advCustom: 
        args.train_id += "_advC"
    if args.use_imcLoss: 
        args.train_id += "_imcL"
    if args.use_sharpFeat: 
        args.train_id += "_s&fL"
    if args.use_clsBal:
        args.train_id += "_clsB"

    args.cls_names = ['bg', 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic']

    print("###### ARGUMENTS #######")
    print(args)
    print("########################")

    aug_folder_100 = os.path.join(args.prefix, args.aug_root, args.train_id+"_rs{}".format(args.real_size), args.model_step) 

    args.train_id += "_AugTest_rs{}".format(args.real_size)

    print(">>>>> TRAIN ID: ", args.train_id)

    if args.trainFromScratch:
        print("Testing by training each model from scratches")
        args.train_id += "_scratch"
    else:
        print("Testing by gradually fine tuning the model")
        args.train_id += "_incremental"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(args.prefix, "./runs", args.train_id))

    #CREATE LOG FOLDER
    log_folder = os.path.join(args.prefix, 'test_results', args.train_id)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    root = args.split_root
    train_img_folder = os.path.join(args.prefix, root, "train", "data")
    train_mask_folder = os.path.join(args.prefix, root, "train", "sem_seg")

    val_img_folder = os.path.join(args.prefix, root, "val", "data")
    val_mask_folder = os.path.join(args.prefix, root, "val", "sem_seg")

    test_img_folder = os.path.join(args.prefix, root, "test", "data")
    test_mask_folder = os.path.join(args.prefix, root, "test", "sem_seg")
    
    n_classes = args.n_cls

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    #backbones = ['resnet50', 'resnet101', 'timm-mobilenetv3_large_100', 'efficientnet-b6', 'tu-mobilevitv2_100']
    #heads = [smp.Unet, smp.FPN, smp.PAN, smp.DeepLabV3Plus, smp.MAnet]

    backbones = ['mit_b5', 'resnext101_32x8d', 'timm-mobilenetv3_large_100', 'tu-mobilevitv2_100']
    heads = [smp.FPN, smp.MAnet, smp.Unet, smp.DeepLabV3Plus]

    # backbones = ['resnet101', 'timm-mobilenetv3_large_100', 'tu-mobilevitv2_100']
    # heads = [smp.Unet, smp.DeepLabV3Plus]

    #TRY DIFFERENT MODELS
    for b in backbones:
        for h in heads:

            last_state_dict = None

            #BUILD THE MODEL
            try:
                # Your code that raises the exception
                model = h(b, encoder_weights="imagenet", classes=n_classes, activation=None)
            except ValueError as e:
                print(str(e))
                continue

            model.to(device)

            args.model_name = model.__class__.__name__ + "_{}".format(b)
            print("################### Training&Test {} ##########################".format(args.model_name))
            print("################### Dataset: {} ##########################".format(aug_folder_100))

            #AUGMENTATION RATIOS
            #ratios = [0.0, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
            ratios = [0.0, 1.0, 2.5, 5.0, 10.0, 25.0]

            #INIT LOG DICTIONARY
            log_dict = dict()
            for p in ratios:
                log_dict[p] = {
                    'train':{},
                    'test_last':{},
                    'test_earlystop':{}
                }

            log_file_name = args.model_name+".json"

            args.log_dict = log_dict
            
            #TRY DIFFERENT AUG RATES
            for p in ratios:

                if args.trainFromScratch:
                    # BUILD THE MODEL
                    model = h(b, encoder_weights="imagenet", classes=n_classes, activation=None)
                    model.to(device)
                elif last_state_dict is not None:
                    model.load_state_dict(last_state_dict)

                args.p = p
                print("+++++++++++++ p = {}".format(p))
                start_time = time.time()

                train_set = AugSemSegDataset(train_img_folder, train_mask_folder, aug_folder=aug_folder_100, aug_p=p, transform=None, real_size=args.real_size)
                val_set = SemSegDataset(val_img_folder, val_mask_folder, transform=None)
                test_set = SemSegDataset(test_img_folder, test_mask_folder, transform=None)
                batch_size = args.batch

                workers = args.workers
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
                val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
                test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)

                max_lr = args.lr
                epoch = args.epochs
                weight_decay = args.wd

                #criterion = nn.CrossEntropyLoss()

                weights = 1 - np.array([0.832770, 0.006781, 0.108312, 0.000917, 0.051221])
                weigths = torch.tensor(weights, device=device, dtype=torch.float)
                criterion = nn.CrossEntropyLoss(weight=weigths)

                optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
                sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader))

                print(f"TRAIN SET SIZE: {len(train_loader) * batch_size}")
                print(f"VAL SET SIZE: {len(val_loader) * batch_size}")
                print(f"TEST SET SIZE: {len(test_loader) * batch_size}")

                history, best_model_dict = fit(epochs=epoch,model= model,train_loader= train_loader,val_loader= val_loader,criterion= criterion,optimizer= optimizer, scheduler= sched,writer= writer,device= device,n_classes= n_classes, args= args)

                print("+++++++++++++ Training set size {}".format(len(train_set)))
                log_dict[args.p]['total_train_size'] = len(train_set)
                log_dict[args.p]['train'] = history

                last_state_dict = model.state_dict()
                model.load_state_dict(best_model_dict)

                args.cm_save_path = os.path.join(log_folder, args.model_name+"_CM.png")

                history = test(model=model, test_loader=test_loader, criterion=criterion, n_classes=n_classes, device=device, args=args)
                log_dict[args.p]['test_last'] = history
                #print("+++++++++++++ test 'LAST' mIOU {}".format(history['test_miou']))
                #print("+++++++++++++ test 'LAST' mIOU ALL {}".format(history['test_miou_all']))
                #print("+++++++++++++ test 'LAST' mIOU CLASS {}".format(history['test_miou_class']))
                print("+++++++++++++ test 'LAST' mIOU {:.3f}".format(history['test_miou_CM']))
                cls_mious = ' '.join(f'{num:.3f}' for num in history['test_class_ious_CM'])
                print("+++++++++++++ test 'LAST' mIOU per class {}".format(cls_mious))

                #print("+++++++++++++ CLASS COUNTS GT {}".format(history['class_counts_gt']))
                #print("+++++++++++++ CLASS COUNTS PRED {}".format(history['class_counts_pred']))

                es_model = os.path.join(args.prefix, 'test_results', args.train_id, args.model_name+'_earlystop.pth')
                if os.path.exists(es_model):
                    model.load_state_dict(torch.load(es_model))
                    model.to(device)
                    history = test(model=model, test_loader=test_loader, criterion=criterion, n_classes=n_classes, device=device, args=args)
                    log_dict[args.p]['test_earlystop'] = history
                    # print("+++++++++++++ test 'EARLY STOP' mIOU {}".format(history['test_miou']))
                    # print("+++++++++++++ test 'EARLY STOP' mIOU ALL {}".format(history['test_miou_all']))
                    # print("+++++++++++++ test 'EARLY STOP' mIOU CLASS {}".format(history['test_miou_class']))
                    print("+++++++++++++ test 'EARLY STOP' mIOU {:.3f}".format(history['test_miou_CM']))
                    cls_mious = ' '.join(f'{num:.3f}' for num in history['test_class_ious_CM'])
                    print("+++++++++++++ test 'EARLY STOP' mIOU per class {}".format(cls_mious))
                else:
                    log_dict[args.p]['test_earlystop'] = None
                    print("+++++++++++++ NO OVERFITTING in {} epochs".format(args.epochs))


                end_time = time.time()
                execution_time = (end_time - start_time) / 60
                print("+++++++++++++ Execution time:", execution_time, "minutes")

                print("########################################################## \n\n")

                with open(os.path.join(log_folder, log_file_name), 'w') as f:
                    json.dump(log_dict, f, indent=4)

            #log_file.close()
            model.cpu()
    
    end_time = time.time()
    total_execution_time = (end_time - init_time) / (60*60)
    print("######### END SCRIPT IN {} hours #########".format(total_execution_time))


if __name__ == "__main__":
    main()


