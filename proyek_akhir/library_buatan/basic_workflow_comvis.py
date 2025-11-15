# import
import torch
from torch import nn

import numpy as np

import pandas as pd

from PIL import Image #, ImageOps

import matplotlib.pyplot as plt

import cv2

import os

import sklearn
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    
    accuracy_score,
    precision_score,
    f1_score,
    recall_score)


from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image
)
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST

# code
class Pov2ImageDataset_V4(torch.utils.data.Dataset):
    def __init__(self, annotations_file90, annotations_fileDep, path_90="", path_dep="", transform=None, target_transform=None):
        self.img_labels90 = pd.read_csv(annotations_file90)
        self.img_labelsDep = pd.read_csv(annotations_fileDep)
        self.path_90 = path_90
        self.path_dep = path_dep

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels90)

    # def crop_center(self, img):
    #     h = min(img.shape[0], img.shape[1])
    #     w = h
        
    #     center = img.shape
    #     x = center[1]/2 - w/2
    #     y = center[0]/2 - h/2
        
    #     crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    #     crop_img = self.add_border(crop_img)

    #     return crop_img

    # def add_border(self, img):
    #     thick = 0.01
    #     top = int(thick * img.shape[0])
    #     bottom = top
    #     left = int(thick * img.shape[1])
    #     right = left
    #     value = [0, 0, 0]
        
    #     new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)
    #     return new_img

    def extrac_item(self,df,path): #, index):
        image, label = df[["file", "labels"]].iloc[0]
        if len(path)!= 0:
          temp = os.path.join(path, image) #self.path + "/" + image
          image = Image.open(temp).convert('L')
        #   image = ImageOps
        else:
          image = Image.open(image).convert('L')

        if self.transform is not None:
           image = self.transform(image)
        return image, label
    
    def get_90(self, idx):
        index = self.img_labels90["index"].iloc[idx]
        image90, lbl_90 = self.extrac_item(self.img_labels90[ self.img_labels90["index"]==index ],
                                           path=self.path_90)#.iloc[idx] #self.img_dir + "/" + self.img_labels90.iloc[idx, 0] #os.path.join(self.img_dir, self.img_labels90.iloc[idx, 0])

        return image90, lbl_90
    
    def get_dep(self, idx):
        index = self.img_labelsDep["index"].iloc[idx]
        imageDep, lbl_dep = self.extrac_item(self.img_labelsDep[ self.img_labelsDep["index"]==index ],
                                             path=self.path_dep)#.iloc[idx] #self.img_dir + "/" + self.img_labels90.iloc[idx, 0] #os.path.join(self.img_dir, self.img_labels90.iloc[idx, 0])

        return imageDep, lbl_dep
        
    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        index = self.img_labels90["index"].iloc[idx]
        image90, lbl_90 = self.extrac_item(self.img_labels90[ self.img_labels90["index"]==index ],
                                           self.path_90)#.iloc[idx] #self.img_dir + "/" + self.img_labels90.iloc[idx, 0] #os.path.join(self.img_dir, self.img_labels90.iloc[idx, 0])
        imageDep, lbl_dep = self.extrac_item(self.img_labelsDep[ self.img_labelsDep["index"]==index],
                                             self.path_dep)#.iloc[idx] #self.img_dir+"/" + self.img_labelsDep.iloc[idx, 0] #os.path.join(self.img_dir, self.img_labelsDep.iloc[idx, 0])
        # image = read_image(img_path, mode="GRAY")
        # image = Image.fromarray( torch.Tensor.numpy(image).squeeze() )
        
        # image90, imageDep = cv2.imread(img_90), cv2.imread(img_dep)
        # image90, imageDep = Image.open(img_90)
        # image90, imageDep = self.crop_center(image90), self.crop_center(imageDep)

        # h, w = image90.shape[:2]
        # h_new = imageDep.shape[0] #imageDep.shape[0]
        # aspect_ratio = h_new / h
        # w_new = imageDep.shape[1] #921 #int(w * aspect_ratio)
        # image90 = cv2.resize(image90, (w_new, h_new))

        # image90, imageDep = self.crop_center(image90), self.crop_center(imageDep)

        # image = cv2.hconcat( [image90, imageDep] )
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = Image.fromarray( image )

        # image90, imageDep = cv2.cvtColor(image90, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageDep, cv2.COLOR_BGR2GRAY)
        # image90, imageDep = Image.fromarray( image90 ), Image.fromarray( imageDep )
        # image90, imageDep = np.array(image90, dtype=np.float32) / 255.0, np.array(imageDep, dtype=np.float32) / 255.0

        try :
            assert imageDep.shape == image90.shape
        except:
            print("imageDep.shape=", imageDep.shape)
            print("image90.shape=", image90.shape)
        # # imageDep = imageDep.astype(np.float32) / 255.0 if imageDep.max() > 1 else imageDep.astype(np.float32)
        # # image90 = image90.astype(np.float32) / 255.0 if image90.max() > 1 else image90.astype(np.float32)
        # imageDep = imageDep.astype(np.uint8)
        # image90 = image90.astype(np.uint8)

        # image90, imageDep = Image.fromarray( image90 ), Image.fromarray( imageDep )
        # label = self.img_labels90.iloc[idx, 1]
        # if self.transform:
        #     image90 = self.transform(image90)
        #     imageDep = self.transform(imageDep)
        # if self.target_transform:
        #     label = self.target_transform(label)
        
        # print("90", image90.shape)
        # print("dep", imageDep.shape)
        
        # image = np.stack([image90, imageDep], axis=0)
        image = torch.stack([image90.squeeze(), imageDep.squeeze()], dim=0) # image = torch.stack([image90, imageDep], dim=0)
        # image = image.squeeze()
        
        # print("image", image.shape)
        # image = image.resize( (image.shape[1], image.shape[2], image.shape[0]) )
        # image = np.transpose(image, (1, 2, 0))
        # print("image1", image.shape)
        
        
        # image = Image.fromarray( image )
        
        # label = self.img_labels90.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, lbl_90


class DfToDataset(torch.utils.data.Dataset):
    def __init__(self, df, path="", transform=None, target_transform=None):
        self.df = df
        self.path=path
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.df[["file", "labels"]].iloc[index]
        if len(self.path)!= 0:
          temp = os.path.join(self.path, image) #self.path + "/" + image
          image = Image.open(temp)
        else:
          image = Image.open(image)

        if self.transform is not None:
           image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df)

class Basic_Workflow_CV:
    def __init__(self, target_names, device):
      self.target_names = target_names
      self.device = device
   
   ###
    def train(self, train_loader, model, optimizer, criterion, str_to_num=False):
        running_loss, total, true = 0, 0, 0
        all_preds = []
        all_targets = []

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if str_to_num:
                labels = torch.from_numpy( np.array([self.target_names[label] for label in labels]) )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)
            probs = torch.exp( nn.functional.log_softmax(outputs, dim=1) )
            # print("train output", probs )

            # print("labels:", labels)
            # print("labels.long():", labels.long())

            loss = criterion(outputs, labels) #.long())
            # print("train loss", loss)
            # train_losses.append(loss/100)
            loss.backward()
            optimizer.step()

            # print statistics for training set
            running_loss += loss.item()
            _ , predicted = outputs.max(1)
            total += labels.size(0)
            true += predicted.eq(labels).sum().item()
            
            all_preds.append(predicted.view(-1).cpu())
            all_targets.append(labels.view(-1).cpu())


            # train_accuracies.append(true / total * 100)

            # if i % 100 == 99:    # print every 100 mini-batches
            #     print('[%d, %5d] loss: %.3f accuracy: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 100, true / total * 100))
            #     running_loss = 0.0

        all_targets = torch.cat(all_targets).numpy() # shape (N,)
        all_preds = torch.cat(all_preds).numpy()
        return running_loss, all_preds, all_targets #total, true

    def validate(self, test_loader, model, criterion,str_to_num=False):
        val_loss, val_true, val_total = 0, 0, 0
        roc_auc = -1

        all_preds = []
        all_probs = []
        all_targets = []

        for inputs, labels in test_loader:
            #inputs, labels = data
            if str_to_num:
                labels = torch.from_numpy( np.array([self.target_names[label] for label in labels]) )

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)
            probs_log = nn.functional.log_softmax(outputs, dim=1)
            probs = torch.exp( probs_log )
            # print("validate output", probs)

            _ , predicted = outputs.max(1)

            loss = criterion(outputs, labels.long())
            # print("validate loss", loss)

            val_loss += loss.item()
            val_true += predicted.eq(labels.long()).sum().item()
            val_total += labels.size(0)

            all_preds.append(predicted.view(-1).cpu())
            all_probs.append(probs.cpu())
            all_targets.append(labels.view(-1).cpu())

        all_probs = torch.cat(all_probs).numpy()     # shape (N, C)
        all_targets = torch.cat(all_targets).numpy() # shape (N,)
        all_preds = torch.cat(all_preds).numpy()

        all_targets_bin = sklearn.preprocessing.label_binarize(all_targets, classes=list( set(all_targets) ))

        # print("all_probs.shpae: {}".format(all_probs.shape))
        # print("all_targets.shpae: {}".format(all_targets.shape))
        # print("all_preds.shpae: {}".format(all_preds.shape))
        # print("all_targets_bin.shpae: {}".format(all_targets_bin.shape))    

        # Hitung ROC AUC untuk multi-class
        # if use_roc:
        #   roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
        try:
            roc_auc = roc_auc_score(all_targets_bin, all_probs, multi_class='ovr')
        except:
            roc_auc = -1
        # print("ROC AUC Score (multi-class):", roc_auc)

        return val_loss, all_preds, all_targets, roc_auc

    def plot_show(self, name_file, val_losses, train_losses, save_path):
        plt.figure(figsize=(10, 6))  # Ukuran plot (opsional)

        # Plot loss validasi
        plt.plot(val_losses, marker='o', label='Valid')

        # Plot loss pelatihan
        plt.plot(train_losses, marker='o', label='Train')

        # Konfigurasi plot
        plt.title(f"Plot nilai loss dari {name_file}")
        plt.xlabel('Epoch atau Iterasi')
        plt.ylabel('Losses')
        plt.legend()  # Menampilkan label plot

        # Menampilkan plot
        plt.grid(True)  # Menampilkan grid (opsional)
        plt.savefig(f"{save_path}/plot_{name_file}.jpg")

    def cm_disp(self, name_file, y_test, predict, save_path):
        # "AN": 0,
        # "DI": 1,
        # "FE": 2,
        # "HA": 3,
        # "SA": 4,
        # "SU": 5
        # labels=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised'] #[0, 1, 2, 3, 4, 5]
        labels_key = [val for key, val in self.target_names.items()]
        labels_val = [key for key, val in self.target_names.items()]
        
        cm = confusion_matrix(y_test, predict, labels=labels_val)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels =labels_key)
        disp.plot()
        disp.ax_.set_title(name_file)

        # disp.plot()

        plt.savefig(f"{save_path}/cm_{name_file}.jpg", dpi=200)

    def visualize_sample(self, name_file, model, valid_loader, batch_size, save_path, str_to_num=False, pov2=False):
        # visualize_sample
        images, labels = next(iter(valid_loader))
        if str_to_num:
            labels = torch.from_numpy( np.array([self.target_names[label] for label in labels]) )
        
        batch_size = images.shape[0]
        if batch_size % 2 == 1:
            batch_size -= 1
        ###
        images.to(self.device)
        output = model(images)
        _, preds = torch.max(output, 1)
        preds = np.squeeze( preds.cpu().numpy() )
        ###

        fig = plt.figure(figsize=(batch_size, batch_size//4))
        fig.suptitle(f"vs_{name_file}")
        for idx in range(batch_size):
            ax = fig.add_subplot(2, batch_size//2, idx+1, xticks=[], yticks=[])
            
            if pov2:
                image = sum( images[idx].cpu().numpy() )
            else:
                image = images[idx].cpu().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = image * .5 + .5
            
            plt.imshow(image)
            title_labels = "{} ({})".format(preds[idx], labels[idx])
            ax.set_title(title_labels, color=('green' if preds[idx]==labels[idx].item() else 'red'))
        plt.savefig(f"{save_path}/vs_{name_file}.jpg", dpi=200)

    def visualize_gradcam(self, name_file, model, valid_loader, batch_size, save_path, pov2=False):
        images, labels = next(iter(valid_loader))
        col = images.shape[0]//2
        fig, axs = plt.subplots(3, col,
                                figsize=(images.shape[0], images.shape[0]//4) )
        fig.suptitle(f"vg_{name_file}")
        # plt.axis('off')

        methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM,
        'finercam': FinerCAM
        }

        target_layers = [model.features[-1]]
        targets = None

        for i in range(col):
            shape = images[i].shape
            img = images[i].reshape(1, shape[0], shape[1], shape[2])

            cam_algorithm = methods["gradcam"]
            with cam_algorithm(model=model,
                    target_layers=target_layers) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            # cam.batch_size = 32
                grayscale_cam = cam(input_tensor=img,#images[0],#input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True
                            )

                grayscale_cam = grayscale_cam[0, :]

                if pov2:
                    real_img = sum(images[i].cpu().numpy())
                    # real_img = real_img * .5 + .5
                    # real_img = (real_img- .5) / .5

                    # print()
                    # print(type(real_img))
                    # print()

                    real_img = ( real_img - min(real_img.reshape(-1)) ) / (max(real_img.reshape(-1))-min(real_img.reshape(-1)))
                    real_img = cv2.merge([real_img, real_img, real_img])
                    # use_rgb=True#False
                else:
                    real_img = np.transpose(images[i].squeeze().numpy(), (1,2,0))
                    real_img = real_img * .5 + .5
                    # use_rgb=True

                cam_image = show_cam_on_image(real_img, grayscale_cam, use_rgb=True)#use_rgb)
                # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            gb_model = GuidedBackpropReLUModel(model=model, device=self.device)
            gb = gb_model(img, target_category=None)

            if pov2:
                gb = np.transpose(gb, (2,0,1))
                gb = sum(gb)
                gb = cv2.merge([gb, gb, gb])

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)

            axs[0][i].imshow(cam_image)
            axs[1][i].imshow(gb)
            axs[2][i].imshow(cam_gb)

            if i == 0:
                axs[0][i].set_title('cam_image')
                axs[1][i].set_title('gb')
                axs[2][i].set_title('cam_gb')


            axs[0][i].axis('off')
            axs[1][i].axis('off')
            axs[2][i].axis('off')
        
        plt.savefig(f"{save_path}/vg_{name_file}.jpg", dpi=200)

            # os.makedirs(args.output_dir, exist_ok=True)

            # cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
            # gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
            # cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

            # cv2.imwrite(cam_output_path, cam_image)
            # cv2.imwrite(gb_output_path, gb)
            # cv2.imwrite(cam_gb_output_path, cam_gb)
    ###