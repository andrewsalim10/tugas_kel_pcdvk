# V2: pakai torch.nn.parallel.DistributedDataParallel

from .basic_workflow_comvis import Basic_Workflow_CV, DfToDataset

import time

import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.models as models

import pandas as pd

import numpy as np

import sklearn
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

### ### ###
import os
import sys
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
#     # such as CUDA, MPS, MTIA, or XPU.
#     acc = torch.accelerator.current_accelerator()
#     backend = torch.distributed.get_default_backend_for_device(acc)
#     # initialize the process group
#     dist.init_process_group(backend, rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()
### ### ###

### ### ###
class Holdout_Workflow_CV2(Basic_Workflow_CV):

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
        # such as CUDA, MPS, MTIA, or XPU.
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def run_demo(self,#demo_fn,
                versi, model1, str_to_num,

                batch_size,
            #  n_splits,
                epochs,
                patience,
            #  class_names,
                path,
                name_model,
                name_data,
                save_path0,
                criterion_const,
                optim_const, optim_params,
                transform_train, transform_test,

                dataset_list:list=None,
                dataset_list2:list=None,

                splits="",
                use_train=True,
                use_pov2=False,
                pilihan='n',
                world_size=1):
        mp.spawn(self.run_workflow,
                args=(versi, model1, str_to_num,
                      batch_size,
                      epochs,
                      patience,
                      path,
                      name_model,
                      name_data,
                      save_path0,
                      criterion_const,
                      optim_const, optim_params,
                      transform_train, transform_test,
                      dataset_list,
                      dataset_list2,
                      splits,
                      use_train,
                      use_pov2,
                      pilihan,
                      world_size,),
                nprocs=world_size,
                join=True)

    def run_workflow(self,
                     rank,
                     versi, model1, str_to_num,

                     batch_size,
                    #  n_splits,
                     epochs,
                     patience,
                    #  class_names,
                     path,
                     name_model,
                     name_data,
                     save_path0,
                     criterion_const,
                     optim_const, optim_params,
                     transform_train, transform_test,

                     dataset_list:list=None,
                     dataset_list2:list=None,

                     splits="",
                     use_train=True,
                     use_pov2=False,
                     pilihan='n',
                     world_size=1
    ):
        ### mobilenetv2_v0.3 kfold 5 v0.3 30 epochs (target 95% akurasi)
        

        # dataset = pd.read_csv("kmufed_latih.csv")
        # X = dataset_const["file"]
        # y = dataset_const["labels"]
        # groups = dataset_const["resp"]

        # n_splits=5
        # group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
        # group_kfold.get_n_splits(X, y, groups)

        # class_names = {
        # "AN": 0,
        # "DI": 1,
        # "FE": 2,
        # "HA": 3,
        # "SA": 4,
        # "SU": 5
        # }
        # torch.manual_seed(42)

        # Set seed acak untuk GPU jika digunakan
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(42)

        # print(group_kfold)
        # for fi in range(1, n_splits+1):
        # fi =f+1
        # print(f"Fold {fi}:")
        # f"{name_data}{n_splits}.{epochs}_{name_model}.{versi}"

        # name_file = f"{name_data}{n_splits}.{epochs}{ep}_{name_model}.{versi}"
        name_file = f"{name_data}_{splits}{epochs}_{name_model}_{versi}"
        # f"{name_data}{n_splits}"
        # name_file_index = f"kdef_holdout{n_splits}"
        save_path = f"{save_path0}/{name_file}"
        try:
            os.makedirs(save_path)
        except FileExistsError:
            # pilihan = 'n'
            while True:
                print("Folder tersebut sudah ada, yakin tetap melanjutkan fungsi run_workflow()?(y/n)")
                print("status use_train: ", use_train)
                # pilihan = int( input("\t1. Iya\t2.Tidak\n") )
                # pilihan = input()
                if pilihan == 'y':
                    print("fungsi tetap dijalankan")
                    break
                elif pilihan == 'n':
                    # print('fungsi run_workflow() dihentikan')
                    break
                else:
                    print('pilih ulang')
            if pilihan == 'n':
                print('fungsi run_workflow() dihentikan')
                return
            # pass

        # path = "KDEF/KDEFv2_cropcolor"#"kmufed" #path
        if not dataset_list is None:
            valid_dataset = pd.read_csv(dataset_list[1]) # dataset_list[1]
            train_dataset = pd.read_csv(dataset_list[0]) # dataset_list[0]
            valid_data = DfToDataset(valid_dataset, path, transform_test)
            train_data = DfToDataset(train_dataset, path, transform_train)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
            test_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=0)

            test_dataset = pd.read_csv(dataset_list[2]) # dataset_list[2]
            test_data = DfToDataset(test_dataset, path, transform_test)
            test_loader2 = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
        if not dataset_list2 is None:
            train_loader = DataLoader(dataset_list2[0], batch_size=batch_size, shuffle=True, num_workers=8)
            test_loader = DataLoader(dataset_list2[1], batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader2 = DataLoader(dataset_list2[2], batch_size=batch_size, shuffle=True, num_workers=0)

        # kfold_dir = "kfold_data_nonperson"
        # train_index = pd.read_csv(f"{kfold_dir}/latih_{name_file_index}.csv")['idx_train'].values
        # test_index = pd.read_csv(f"{kfold_dir}/valid_{name_file_index}.csv")['idx_valid'].values

        # train_groups = pd.read_csv(f"{kfold_dir}/latih_{name_file_index}.csv")['groups_train'].values#groups[train_index]
        # test_groups = pd.read_csv(f"{kfold_dir}/valid_{name_file_index}.csv")['groups_valid'].values#groups[test_index]

        # train_groups_unique = list( set( train_groups ))
        # test_groups_unique = list( set( test_groups ))

        # if set(train_groups) == set(test_groups):
        #     print(f"\tFold {i+1} bocor")
        #     print()

        # is_break=False
        # for j in range( len(train_groups_unique) ):
        #     for k in range( len(test_groups_unique) ):
        #         if test_groups_unique[k] == train_groups_unique[j]:
        #             is_break=True
        #             break

        # if is_break:
        #     print()
        #     print(f"\t\tFold {fi} bocor")
        #     print("\t\tk:{}, j:{}".format(k, j))
        #     print()
        #     break
        # is_break = False
        # for j in range( len(train_index) ):
        #     for k in range( len(test_index) ):
        #         if test_index[k] == train_index[j]:
        #             is_break=True
        #             break

        # if is_break:
        #     print(f"\tFold {i} bocor")
        #     print("train_index:{}, test_index:{}".format(test_index, test_index))
        #     print()
        #     break

        # train_dataset, val_dataset = dataset_const.iloc[ train_index, :], dataset_const.iloc[ test_index, :]

        # path = "kmufed"
        # train_data = DfToDataset(train_dataset, path, transform_train)
        # valid_data = DfToDataset(val_dataset, path, transform_test)

        # print( next(iter(train_data)))
        # plt.imshow( next(iter(train_data))[0].view(224, 224, 3) )
        # break

        # batch_size = 8  # Sesuaikan dengan kebutuhan Anda
        # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        # test_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        ##Pemodelan##
        # torch.manual_seed(42)

        # Set seed acak untuk GPU jika digunakan
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(42)

        ###
        # model = MobileNetV2_CBAM(n_class=6, input_size=224) #last_channel=1280)
        # model = MobileNetV2_CBAM_hilmi(n_class=7) #last_channel=1280)
        # model = MobileNetV2_CBAM_hilmi_v2(n_class=6, input_size=224) #last_channel=1280 hchan32)


        # model = init_model
        # model = nn.DataParallel(model)

        # print('rank:',rank)
        # print('wsize:',world_size)
        self.setup(rank, world_size)
        model = DDP(model1.to(self.device), device_ids=[rank])
        # try:
        model = model.to(self.device)
        # except:
            # print("ada error di device")

        # model.load_state_dict(torch.load(f"../Hasil Eksperimen/mobilenetv2cbam/{name_file}.pth"))
        # model.load_state_dict(torch.load(f"../Hasil Eksperimen/mobilenetv2cbam/kmu_fed_kfold5.1.50_mobilenetv2cbam_v1.3.3SGD_lr0.003_momen0.9_wdecay0.01_20batch.pth"))


        # Define loss function and optimizer
        criterion = criterion_const

        # optimizer = optim.Adam(model.parameters(), 
        #                             lr= 0.0001,#0.01, #0.0001, #0.001,
        #                             betas=(0.95, 0.999),#(0.9, 0.999),
        #                             weight_decay=0)#0.0001,)
        # optimizer = optim.SGD(model.parameters(),
        #                       lr=0.002,#0.002,#0.001,
        #                       momentum=0.99,#0.99, 0.9,
        #                       weight_decay=0.0002)#0.00002,)#0.0002)#0.003)#0.001)
        # optimizer = optim.Adam(model.parameters(), )
        #                        lr=0.0001)
        optimizer = optim_const(model.parameters(), **optim_params)

        # optimizer = optim.Adam(model.parameters(), lr=0.0010034630022641613, weight_decay=0.00036801507694967837)
        # optimizer = optim.Adam(model.parameters(), lr=0.0011775027145180192, weight_decay=0.0006014906060577658,
        #                        )
        # optimizer = optim.Adam(model.parameters())
        # batch_size: 9
        # lr: 0.0010034630022641613
        # weight_decay: 0.00036801507694967837

        # epochs: 25
        # batch_size: 19
        # lr: 0.0011775027145180192
        # weight_decay: 0.0006014906060577658
        # optimizer: Adam

        # patience = 15
        counter = 0

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)# scheduler = scheduler_const

        train_losses = []
        train_accuracies = []
        train_precisions = list()
        train_f1s = list()
        train_recalls = list()
        val_losses = []
        val_accuracies = []
        val_precisions = list()
        val_f1s = list()
        val_recalls = list()
        roc_auc_list = []

        best_val_accuracy = 0.0
        best_val_precision = 0.0
        best_val_f1 = 0.0
        best_val_recall = 0.0
        best_val_loss = np.inf

        best_train_acc = 0.0
        best_train_precision = 0.0
        best_train_f1 = 0.0
        best_train_recall = 0.0
        best_train_loss = np.inf

        best_epoch = 0
        best_roc_auc = -1
        best_all_preds = 0
        best_all_targets = 0

        start_time = time.time()
        ## ## ## ## ##

        # epochs = 30
        # name_file = f"kmu_fed_kfold{n_splits}.{fi}.{epochs}_{name_model}"
        # save_path = f"../Hasil Eksperimen"
        if use_train:
            epc = 0+1
            for epoch in range(epc, epochs+1): #30 # loop over the dataset multiple times

                running_loss = 0.0
                total = 0
                true = 0
                val_loss = 0.0
                val_total = 0
                val_true = 0

                ## latih model
                model.train()
                # for i, data in enumerate(train_loader, 0):
                running_loss, train_preds, train_targets = self.train(train_loader, model, optimizer, criterion, str_to_num=str_to_num)
                ####
                # print("train_targets:", train_targets)
                # print()
                # print("train_preds:", train_preds)
                ####
                train_accuracy = accuracy_score(train_targets, train_preds)#true/total
                train_precision = precision_score(train_targets, train_preds, average='weighted', zero_division=0.0)
                train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0.0)
                train_recall = recall_score(train_targets, train_preds, average='weighted', zero_division=0.0)
                train_loss = running_loss/len(train_loader)

                train_accuracies.append(train_accuracy)
                train_precisions.append(train_precision)
                train_f1s.append(train_f1)
                train_recalls.append(train_recall)
                train_losses.append(train_loss)

                ## validasi model
                model.eval()  # Set model to evaluation mode
                with torch.no_grad(): # run ulang (sudah)
                    val_loss, valid_preds, valid_targets, roc_auc = self.validate(test_loader, model, criterion, str_to_num=str_to_num)

                    val_accuracy = accuracy_score(valid_targets, valid_preds)
                    val_precision = precision_score(valid_targets, valid_preds, average='weighted', zero_division=0.0)
                    val_f1 = f1_score(valid_targets, valid_preds, average='weighted', zero_division=0.0)
                    val_recall = recall_score(valid_targets, valid_preds, average='weighted', zero_division=0.0)
                    val_loss = val_loss/len(test_loader)

                    val_accuracies.append(val_accuracy)
                    val_precisions.append(val_precision)
                    val_f1s.append(val_f1)
                    val_recalls.append(val_recall)
                    val_losses.append(val_loss)
                    roc_auc_list.append(roc_auc)

                # update scheduler
                scheduler.step(val_losses[-1])

                # if val_loss < best_val_loss:
                if round(val_accuracy, 2) > round(best_val_accuracy, 2):
                    counter = 0

                    best_val_accuracy = val_accuracy
                    best_val_precision = val_precision
                    best_val_f1 = val_f1
                    best_val_recall = val_recall
                    best_val_loss = val_loss

                    best_train_acc = train_accuracy
                    best_train_precision = train_precision
                    best_train_f1 = train_f1
                    best_train_recall = train_recall
                    best_train_loss = train_loss

                    best_epoch = epoch
                    best_roc_auc = roc_auc

                    best_valid_preds = valid_preds
                    best_valid_targets = valid_targets

                    torch.save(model.module.state_dict(), f"{save_path}/{name_file}.pth")
                    
                else:
                    counter += 1
                    if counter >= patience:
                        print("\tEarly stopping triggered at epoch {}".format(epoch) )
                        break
                end_time = time.time()
                elapsed_time = end_time - start_time
            # Simpan data hasil dari epoch terbaik
            with open(f"{save_path}/report_validasi_{name_file}.txt", "w") as file:
                # file.write(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch} - Training Loss: {best_train_loss:.3f} Training Accuracy: {best_train_acc* 100:.3f}% Validation loss: {best_val_loss:.3f} Validation accuracy: {best_val_accuracy * 100:.3f}%\n')
                # # file.write(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
                # file.write(f"\t\tROC AUC validation: {best_roc_auc}")
                # # file.write(f"\t\tROC AUC test: {test_roc_auc}")
                # file.write(f'\tWaktu komputasi: {elapsed_time} detik\n')
                # # file.write(f'\tWaktu komputasi testing: {elapsed_time2} detik')
                # file.write("\n")
                # file.write(classification_report(best_valid_targets, best_valid_preds))
                file.write(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch}')
                file.write(f"\n\t\tTraining Loss: {best_train_loss:.3f}") 
                file.write(f"\n\t\t\tTraining Accuracy: {best_train_acc* 100:.3f}%")
                file.write(f"\n\t\t\tTraining Precision: {best_train_precision* 100:.3f}%")
                file.write(f"\n\t\t\tTraining F1: {best_train_f1* 100:.3f}%")
                file.write(f"\n\t\t\tTraining Recall: {best_train_recall* 100:.3f}%")

                file.write(f"\n\t\tValidation loss: {best_val_loss:.3f}")
                file.write(f"\n\t\t\tValidation Accuracy: {best_val_accuracy * 100:.3f}%")
                file.write(f"\n\t\t\tValidation Precision: {best_val_precision* 100:.3f}%")
                file.write(f"\n\t\t\tValidation F1: {best_val_f1* 100:.3f}%")
                file.write(f"\n\t\t\tValidation Recall: {best_val_recall* 100:.3f}%")

                # print(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
                file.write(f"\n\t\tROC AUC validation: {best_roc_auc}")
                # print(f"\t\tROC AUC test: {test_roc_auc}")
                # end_time = time.time()
                # Cetak waktu komputasi
                file.write(f'\n\tWaktu komputasi training: {elapsed_time} detik\n')
                # print(f'\tWaktu komputasi testing: {elapsed_time2} detik')
                file.write(classification_report(best_valid_targets, best_valid_preds, zero_division=0.0))

            dict_hasil_validasi = {"epoch": [epoch],
                        "best_epoch": [best_epoch],
                        "best_train_loss": [best_train_loss], 
                        "best_train_acc": [100*best_train_acc], 
                        "best_train_precision": [100*best_train_precision],
                        "best_train_f1": [100*best_train_f1],
                        "best_train_recall": [100*best_train_recall],

                        "best_val_loss": [best_val_loss], 
                        "best_val_accuracy": [100*best_val_accuracy],
                        "best_val_precision": [100*best_val_precision],
                        "best_val_f1": [100*best_val_f1],
                        "best_val_recall": [100*best_val_recall],

                        "best_roc_auc": [best_roc_auc],
                        "elapsed_time": [elapsed_time]}
            pd.DataFrame(dict_hasil_validasi).to_csv(f"{save_path}/hasil_validasi_{name_file}.csv",index=False)
            # Simpan epochs dan list akurasi dan loss
            dict_loss_acc_test={"epoch": [i for i in range(epc, epoch+1)],
                                "train_losses": train_losses,
                                "train_accuracies": train_accuracies,
                                "train_precisions": train_precisions,
                                "train_f1s": train_f1s,
                                "train_recalls": train_recalls,

                                "val_losses": val_losses,
                                "val_accuracies": val_accuracies,
                                "val_precisions": val_precisions,
                                "val_f1s": val_f1s,
                                "val_recalls": val_recalls,

                                "roc_auc": roc_auc_list}
            pd.DataFrame(dict_loss_acc_test).to_csv(f"{save_path}/epoch_akurasi_loss_{name_file}.csv", index=False)
            self.plot_show(f"valid_{name_file}", val_losses, train_losses, save_path)

            # pass
        else:
            model.load_state_dict(torch.load(f"{save_path}/{name_file}.pth"))
            running_loss = 0.0
            total = 0
            true = 0
            val_loss = 0.0
            val_total = 0
            val_true = 0

            ## latih model
            model.eval()
            with torch.no_grad():
                # for i, data in enumerate(train_loader, 0):
                running_loss, train_preds, train_targets, roc_auc_temp = self.validate(train_loader, model, criterion, str_to_num=str_to_num)
                ####
                # print("train_targets:", train_targets)
                # print()
                # print("train_preds:", train_preds)
                ####
                train_accuracy = accuracy_score(train_targets, train_preds)#true/total
                train_precision = precision_score(train_targets, train_preds, average='weighted', zero_division=0.0)
                train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0.0)
                train_recall = recall_score(train_targets, train_preds, average='weighted', zero_division=0.0)
                train_loss = running_loss/len(train_loader)

                train_accuracies.append(train_accuracy)
                train_precisions.append(train_precision)
                train_f1s.append(train_f1)
                train_recalls.append(train_recall)
                train_losses.append(train_loss)

            ## validasi model
            model.eval()  # Set model to evaluation mode
            with torch.no_grad(): # run ulang (sudah)
                val_loss, valid_preds, valid_targets, roc_auc = self.validate(test_loader, model, criterion, str_to_num=str_to_num)

                val_accuracy = accuracy_score(valid_targets, valid_preds)
                val_precision = precision_score(valid_targets, valid_preds, average='weighted', zero_division=0.0)
                val_f1 = f1_score(valid_targets, valid_preds, average='weighted', zero_division=0.0)
                val_recall = recall_score(valid_targets, valid_preds, average='weighted', zero_division=0.0)
                val_loss = val_loss/len(test_loader)

                val_accuracies.append(val_accuracy)
                val_precisions.append(val_precision)
                val_f1s.append(val_f1)
                val_recalls.append(val_recall)
                val_losses.append(val_loss)
                roc_auc_list.append(roc_auc)

            # update scheduler
            # scheduler.step(val_losses[-1])

            # if val_loss < best_val_loss:
            # if round(val_accuracy, 2) >= round(best_val_accuracy, 2):
            counter = 0

            best_val_accuracy = val_accuracy
            best_val_precision = val_precision
            best_val_f1 = val_f1
            best_val_recall = val_recall
            best_val_loss = val_loss

            best_train_acc = train_accuracy
            best_train_precision = train_precision
            best_train_f1 = train_f1
            best_train_recall = train_recall
            best_train_loss = train_loss

            best_epoch = "bukan train"
            epoch = "bukan train"
            best_roc_auc = roc_auc

            best_valid_preds = valid_preds
            best_valid_targets = valid_targets

            end_time = time.time()
            elapsed_time = end_time - start_time


            # torch.save(model.state_dict(), f"{save_path}/{name_file}.pth")
            # pass

        # roc_auc_direct = roc_auc_score(y_true, y_scores)
        # print(f'\tEpochs {epoch+1} - Train AVG Loss: {(sum(train_losses)/len(train_losses)):.3f} Train AVG Accuracy: {sum(train_accuracies* 100)/len(train_accuracies) :.3f}% Validation AVG loss: {sum(val_losses)/len(val_losses):.3f} Validation AVG accuracy: {(sum(val_accuracies * 100)/len(val_accuracies)):.3f}%')
        # print(f'\tEpochs {epoch+1}\n\tBest -> epoch: {best_epoch} - Training Loss: {best_train_loss:.3f} Training Accuracy: {best_train_acc* 100:.3f}% Validation loss: {best_val_loss:.3f} Validation accuracy: {best_val_accuracy * 100:.3f}%')
        # print(f"\t\tROC AUC Score (multi-class): {best_roc_auc}")
        # end_time = time.time()
        # elapsed_time = end_time - start_time

        # ## uji model
        # start_time2 = time.time()
        # model.load_state_dict(torch.load(f"{save_path}/{name_file}.pth") )
        # # test_dataset = pd.read_csv("kmufed_uji.csv")
        # # test_data = DfToDataset(test_dataset, path, transform_test)
        # # test_loader2 = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # test_accuracy = 0.0
        # test_precision = 0.0
        # test_f1 = 0.0
        # test_recall = 0.0
        # test_loss = 0.0
        # model.eval()  # Set model to evaluation mode
        # with torch.no_grad():
        #     test_loss, test_preds, test_targets, test_roc_auc = self.validate(test_loader2, model, criterion, str_to_num=str_to_num)

        #     test_accuracy = accuracy_score(valid_targets, valid_preds)
        #     test_precision = precision_score(valid_targets, valid_preds, average='weighted')
        #     test_f1 = f1_score(valid_targets, valid_preds, average='weighted')
        #     test_recall = recall_score(valid_targets, valid_preds, average='weighted')
        #     test_loss = val_loss/len(test_loader)

        #     # val_accuracies.append(val_accuracy)
        #     # val_precisions.append(val_precision)
        #     # val_f1s.append(val_f1)
        #     # val_recalls.append(val_recall)
        #     # val_losses.append(val_loss)
        #     # roc_auc_list.append(roc_auc)

        #     # test_accuracy = test_true / test_total
        #     # test_loss = test_loss/len(test_loader2)
        #     # val_accuracies.append(val_accuracy)
        #     # val_losses.append(val_loss)
        #     # roc_auc_list.append(roc_auc)
        # elapsed_time2 = time.time() - start_time2
        print(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch}')
        print(f"\t\tTraining Loss: {best_train_loss:.3f}") 
        print(f"\t\t\tTraining Accuracy: {best_train_acc* 100:.3f}%")
        print(f"\t\t\tTraining Precision: {best_train_precision* 100:.3f}%")
        print(f"\t\t\tTraining F1: {best_train_f1* 100:.3f}%")
        print(f"\t\t\tTraining Recall: {best_train_recall* 100:.3f}%")

        print(f"\t\tValidation loss: {best_val_loss:.3f}")
        print(f"\t\t\tValidation Accuracy: {best_val_accuracy * 100:.3f}%")
        print(f"\t\t\tValidation Precision: {best_val_precision* 100:.3f}%")
        print(f"\t\t\tValidation F1: {best_val_f1* 100:.3f}%")
        print(f"\t\t\tValidation Recall: {best_val_recall* 100:.3f}%")

        # print(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
        print(f"\t\tROC AUC validation: {best_roc_auc}")
        # print(f"\t\tROC AUC test: {test_roc_auc}")
        # end_time = time.time()
        # Cetak waktu komputasi
        print(f'\tWaktu komputasi training: {elapsed_time} detik')
        # print(f'\tWaktu komputasi testing: {elapsed_time2} detik')
        print(classification_report(best_valid_targets, best_valid_preds, zero_division=0.0))
        print(f'\tFinished Training: {name_file}')

        # # Simpan data hasil dari epoch terbaik
        # with open(f"{save_path}/report_validasi_{name_file}.txt", "w") as file:
        #     # file.write(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch} - Training Loss: {best_train_loss:.3f} Training Accuracy: {best_train_acc* 100:.3f}% Validation loss: {best_val_loss:.3f} Validation accuracy: {best_val_accuracy * 100:.3f}%\n')
        #     # # file.write(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
        #     # file.write(f"\t\tROC AUC validation: {best_roc_auc}")
        #     # # file.write(f"\t\tROC AUC test: {test_roc_auc}")
        #     # file.write(f'\tWaktu komputasi: {elapsed_time} detik\n')
        #     # # file.write(f'\tWaktu komputasi testing: {elapsed_time2} detik')
        #     # file.write("\n")
        #     # file.write(classification_report(best_valid_targets, best_valid_preds))
        #     file.write(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch}')
        #     file.write(f"\n\t\tTraining Loss: {best_train_loss:.3f}") 
        #     file.write(f"\n\t\t\tTraining Accuracy: {best_train_acc* 100:.3f}%")
        #     file.write(f"\n\t\t\tTraining Precision: {best_train_precision* 100:.3f}%")
        #     file.write(f"\n\t\t\tTraining F1: {best_train_f1* 100:.3f}%")
        #     file.write(f"\n\t\t\tTraining Recall: {best_train_recall* 100:.3f}%")

        #     file.write(f"\n\t\tValidation loss: {best_val_loss:.3f}")
        #     file.write(f"\n\t\t\tValidation Accuracy: {best_val_accuracy * 100:.3f}%")
        #     file.write(f"\n\t\t\tValidation Precision: {best_val_precision* 100:.3f}%")
        #     file.write(f"\n\t\t\tValidation F1: {best_val_f1* 100:.3f}%")
        #     file.write(f"\n\t\t\tValidation Recall: {best_val_recall* 100:.3f}%")

        #     # print(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
        #     file.write(f"\n\t\tROC AUC validation: {best_roc_auc}")
        #     # print(f"\t\tROC AUC test: {test_roc_auc}")
        #     # end_time = time.time()
        #     # Cetak waktu komputasi
        #     file.write(f'\n\tWaktu komputasi training: {elapsed_time} detik\n')
        #     # print(f'\tWaktu komputasi testing: {elapsed_time2} detik')
        #     file.write(classification_report(best_valid_targets, best_valid_preds, zero_division=0.0))

        # dict_hasil_validasi = {"epoch": [epoch],
        #             "best_epoch": [best_epoch],
        #             "best_train_loss": [best_train_loss], 
        #             "best_train_acc": [100*best_train_acc], 
        #             "best_train_precision": [100*best_train_precision],
        #             "best_train_f1": [100*best_train_f1],
        #             "best_train_recall": [100*best_train_recall],

        #             "best_val_loss": [best_val_loss], 
        #             "best_val_accuracy": [100*best_val_accuracy],
        #             "best_val_precision": [100*best_val_precision],
        #             "best_val_f1": [100*best_val_f1],
        #             "best_val_recall": [100*best_val_recall],

        #             "best_roc_auc": [best_roc_auc],
        #             "elapsed_time": [elapsed_time]}
        # pd.DataFrame(dict_hasil_validasi).to_csv(f"{save_path}/hasil_validasi_{name_file}.csv",index=False)

        # print()
        # print("idx_test.shape:", len(test_index))
        # print("test_groups.shape:", len(test_groups))
        # print("all_targets.shape:", len(all_targets))
        # print("all_preds.shape:", len(all_preds))
        # print()

        # Simpan data target dan hasil prediksi
        # dict_test = {"idx_test": test_index, "groups": test_groups, "y_test": all_targets, "y_predict": all_preds}

        # pd.DataFrame(dict_test).to_csv(f"{save_path}/tabel_{name_file}.csv",index=False)
        
        # self.plot_show(f"valid_{name_file}", val_losses, train_losses, save_path)

        self.cm_disp(f"valid_{name_file}", np.array(best_valid_targets), np.array(best_valid_preds), save_path)
        
        self.visualize_sample(f"valid_{name_file}", model, test_loader, batch_size, save_path, str_to_num=str_to_num, pov2=use_pov2)
        self.visualize_gradcam(f"valid_{name_file}", model.module, test_loader, batch_size, save_path, pov2=use_pov2)

        # Simpan epochs dan list akurasi dan loss
        # dict_loss_acc_test={"epoch": [i for i in range(epc, epoch+1)],
        #                     "train_losses": train_losses,
        #                     "train_accuracies": train_accuracies,
        #                     "train_precisions": train_precisions,
        #                     "train_f1s": train_f1s,
        #                     "train_recalls": train_recalls,

        #                     "val_losses": val_losses,
        #                     "val_accuracies": val_accuracies,
        #                     "val_precisions": val_precisions,
        #                     "val_f1s": val_f1s,
        #                     "val_recalls": val_recalls,

        #                     "roc_auc": roc_auc_list}
        # pd.DataFrame(dict_loss_acc_test).to_csv(f"{save_path}/epoch_akurasi_loss_{name_file}.csv", index=False)
        print()
        print()

        ## uji model
        start_time2 = time.time()
        model1.load_state_dict(torch.load(f"{save_path}/{name_file}.pth") )
        model = DDP(model1.to(self.device), device_ids=[rank])
        # try:
        model = model.to(self.device)

        # test_dataset = pd.read_csv("kmufed_uji.csv")
        # test_data = DfToDataset(test_dataset, path, transform_test)
        # test_loader2 = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        test_accuracy = 0.0
        test_precision = 0.0
        test_f1 = 0.0
        test_recall = 0.0
        test_loss = 0.0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            test_loss, test_preds, test_targets, test_roc_auc = self.validate(test_loader2, model, criterion, str_to_num=str_to_num)

            test_accuracy = accuracy_score(test_targets, test_preds)
            test_precision = precision_score(test_targets, test_preds, average='weighted', zero_division=0.0)
            test_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0.0)
            test_recall = recall_score(test_targets, test_preds, average='weighted', zero_division=0.0)
            test_loss = val_loss/len(test_loader)

            # val_accuracies.append(val_accuracy)
            # val_precisions.append(val_precision)
            # val_f1s.append(val_f1)
            # val_recalls.append(val_recall)
            # val_losses.append(val_loss)
            # roc_auc_list.append(roc_auc)

            # test_accuracy = test_true / test_total
            # test_loss = test_loss/len(test_loader2)
            # val_accuracies.append(val_accuracy)
            # val_losses.append(val_loss)
            # roc_auc_list.append(roc_auc)
        elapsed_time2 = time.time() - start_time2

        # print(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch}')
        # print(f"\t\tTraining Loss: {best_train_loss:.3f}") 
        # print(f"\t\t\tTraining Accuracy: {best_train_acc* 100:.3f}%")
        # print(f"\t\t\tTraining Precision: {best_train_precision* 100:.3f}%")
        # print(f"\t\t\tTraining F1: {best_train_f1* 100:.3f}%")
        # print(f"\t\t\tTraining Recall: {best_train_recall* 100:.3f}%")

        print(f"\t\tTest loss: {test_loss:.3f}")
        print(f"\t\t\tTest Accuracy: {test_accuracy * 100:.3f}%")
        print(f"\t\t\tTest Precision: {test_precision* 100:.3f}%")
        print(f"\t\t\tTest F1: {test_f1* 100:.3f}%")
        print(f"\t\t\tTest Recall: {test_recall* 100:.3f}%")

        # print(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
        # print(f"\t\tROC AUC validation: {best_roc_auc}")
        print(f"\t\tROC AUC test: {test_roc_auc}")
        # end_time = time.time()
        # Cetak waktu komputasi
        # print(f'\tWaktu komputasi training: {elapsed_time} detik')
        print(f'\tWaktu komputasi testing: {elapsed_time2} detik')
        print(classification_report(test_targets, test_preds, zero_division=0.0))
        print(f'\tFinished Testing: {name_file}')

        # Simpan data hasil dari epoch terbaik
        with open(f"{save_path}/report_uji_{name_file}.txt", "w") as file:
            # file.write(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch} - Training Loss: {best_train_loss:.3f} Training Accuracy: {best_train_acc* 100:.3f}% Validation loss: {best_val_loss:.3f} Validation accuracy: {best_val_accuracy * 100:.3f}%\n')
            # # file.write(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
            # file.write(f"\t\tROC AUC validation: {best_roc_auc}")
            # # file.write(f"\t\tROC AUC test: {test_roc_auc}")
            # file.write(f'\tWaktu komputasi: {elapsed_time} detik\n')
            # # file.write(f'\tWaktu komputasi testing: {elapsed_time2} detik')
            # file.write("\n")
            # file.write(classification_report(best_valid_targets, best_valid_preds))
            # file.write(f'\tEpochs {epoch}\n\t\tBest -> epoch: {best_epoch}')
            file.write(f"\n\t\tTest loss: {test_loss:.3f}")
            file.write(f"\n\t\t\tTest Accuracy: {test_accuracy * 100:.3f}%")
            file.write(f"\n\t\t\tTest Precision: {test_precision* 100:.3f}%")
            file.write(f"\n\t\t\tTest F1: {test_f1* 100:.3f}%")
            file.write(f"\n\t\t\tTest Recall: {test_recall* 100:.3f}%")

            # print(f"\t\tTest loss: {test_loss:.3f} test accuracy: {test_accuracy * 100:.3f}")
            # print(f"\t\tROC AUC validation: {best_roc_auc}")
            file.write(f"\n\t\tROC AUC test: {test_roc_auc}")
            # end_time = time.time()
            # Cetak waktu komputasi
            # print(f'\tWaktu komputasi training: {elapsed_time} detik')
            file.write(f'\n\tWaktu komputasi testing: {elapsed_time2} detik\n')
            file.write(classification_report(test_targets, test_preds, zero_division=0.0))

        dict_hasil_uji = {"epoch": [epoch],
                      "test_loss": [test_loss], 
                      "test_accuracy": [100*test_accuracy],
                      "test_precision": [100*test_precision],
                      "test_f1": [100*test_f1],
                      "test_recall": [100*test_recall],

                      "test_roc_auc": [test_roc_auc],
                      "elapsed_time": [elapsed_time2]}
        pd.DataFrame(dict_hasil_uji).to_csv(f"{save_path}/hasil_uji_{name_file}.csv",index=False)

        # print()
        # print("idx_test.shape:", len(test_index))
        # print("test_groups.shape:", len(test_groups))
        # print("all_targets.shape:", len(all_targets))
        # print("all_preds.shape:", len(all_preds))
        # print()

        # Simpan data target dan hasil prediksi
        # dict_test = {"idx_test": test_index, "groups": test_groups, "y_test": all_targets, "y_predict": all_preds}

        # pd.DataFrame(dict_test).to_csv(f"{save_path}/tabel_{name_file}.csv",index=False)
        
        # self.plot_show(f"valid_{name_file}", val_losses, train_losses, save_path)

        self.cm_disp(f"uji_{name_file}", np.array(test_targets), np.array(test_preds), save_path)
        
        self.visualize_sample(f"uji_{name_file}", model, test_loader2, batch_size, save_path, str_to_num=str_to_num, pov2=use_pov2)
        self.visualize_gradcam(f"uji_{name_file}", model.module, test_loader2, batch_size, save_path, pov2=use_pov2)

        self.cleanup()
        del train_loader
        del test_loader
        del model

        torch.cuda.empty_cache()
        # break
        ###
    ###
    ### ###