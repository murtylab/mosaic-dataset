import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    alexnet,
    AlexNet,
    AlexNet_Weights,
    resnet50,
    ResNet,
    ResNet50_Weights,
    ResNet18_Weights,
    resnet18,
    mobilenet_v2,
    MobileNet_V2_Weights,
    squeezenet1_1,
    SqueezeNet1_1_Weights,
    swin_t,
    Swin_T_Weights,
)
from torchvision.models.squeezenet import Fire
from typing import Optional
import copy
from typing import Union
from ..constants import num_subjects, inference_dataset_name_mapping


"""
Input: (N, C_in, H_in, W_in)
Output: (N, C_out, H_out, W_out), where

H_out =[(H_in + 2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/stride[0]] + 1
W_out =[(W_in + 2×padding[1]−dilation[1]×(kernel_size[1]−1)−1)/stride[1]] + 1
"""

"""
TODO adjust num_outputs so we dont have such a huge range. consider changing the
 output to RSA for the glasser groups, so a min of 1 and max of 22
"""


class CNN(nn.Module):
    def __init__(
        self,
        num_outputs=91282,
        num_subjects=93,
        subject_embedding_dim=16,
        angle_embedding_dim=8,
    ):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, padding=1
        )

        # Adaptive Pooling to a fixed output size (e.g., 4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.fc1 = nn.Linear(1024 * 4 * 4, 256)  # Adjust based on adaptive output size

        # Embedding layers
        self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
        self.angle_fc = nn.Linear(1, angle_embedding_dim)

        # Combined fully connected layers
        self.fc_combined = nn.Linear(
            256 + subject_embedding_dim + angle_embedding_dim,
            max(512, num_outputs // 2),
        )
        self.norm1 = nn.LayerNorm(max(512, num_outputs // 2))
        self.fc_output = nn.Linear(max(512, num_outputs // 2), num_outputs)

        # Dropout
        dropout_rate = 0.5 if num_outputs > 1000 else 0.2
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, subject_id, viewing_angle):
        # Convolutional + Pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Embeddings
        subject_embed = self.subject_embedding(subject_id)
        angle_embed = F.relu(self.angle_fc(viewing_angle.unsqueeze(1)))

        # Combine features
        combined = torch.cat((x, subject_embed, angle_embed), dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.norm1(combined)
        combined = self.dropout(combined)

        # Output layer
        output = self.fc_output(combined)

        return output

    def count_parameters(self):
        """
        Counts how many trainable parameters are in our network. Networks are
        often defined partly by how many trainable parameters they contain, as this
        often is a good indicator of the networks complexity.

        Returns
        -------
        int
            number of trainable weights and biases in the model

        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RegressionAlexNet(nn.Module):
    def __init__(
        self,
        num_outputs: int = 91282,
        subjects: list = [],
        subject_embedding_dim: Optional[int] = 128,
        pretrained: bool = False,
        bottleneck_dim: int = 512,
        device: str = "cpu",
    ):
        """
        Modified AlexNet for regression tasks

        Args:
            num_outputs (int): Number of regression outputs
            pretrained (bool): Whether to use pretrained weights for the convolutional layers
        """
        super(RegressionAlexNet, self).__init__()
        self.subjects = subjects
        self.subjectID2idx = {
            subjectID: idx for idx, subjectID in enumerate(self.subjects)
        }
        self.device = device
        self.pretrained = pretrained
        self.subject_embedding_dim = subject_embedding_dim

        # Load the original AlexNet model
        original_alexnet = AlexNet()
        if self.pretrained:
            # Load pretrained weights if specified
            original_alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            self.features = original_alexnet.features
        else:
            # Modify features to add BatchNorm
            new_features = []
            for layer in original_alexnet.features:
                new_features.append(layer)
                if isinstance(layer, nn.Conv2d):
                    # Add BatchNorm after each Conv layer
                    new_features.append(nn.BatchNorm2d(layer.out_channels))

            # Replace features with new BatchNorm-enhanced version
            self.features = nn.Sequential(*new_features)
            # self.features = original_alexnet.features #uncomment to use orginal alexnet conv features for compatibility with some previously trained weights (without the batchnorm2d after conv)
        # Convolutional features from AlexNet
        self.avgpool = original_alexnet.avgpool
        self.conv_output_dim = 256 * 6 * 6  # AlexNet's conv output size

        # Embedding layers
        if self.subject_embedding_dim:
            self.subject_embedding = nn.Embedding(
                len(self.subjects), subject_embedding_dim
            )

            # Calculate dimensions for the merged features
            self.total_embedding_dim = subject_embedding_dim
        else:
            self.total_embedding_dim = 0

        # Modified regressor architecture to incorporate embeddings
        self.regressor_conv = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
        )

        self.regressor_combined = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(4096 + self.total_embedding_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(bottleneck_dim),
        )

        # self.regressor_final = nn.Linear(bottleneck_dim, num_outputs)
        self.regressor_final = nn.Sequential(
            nn.Linear(bottleneck_dim, num_outputs),
            # nn.Tanh(),  # Constrains outputs to [-1, 1]
            # Lambda(lambda x: 5 * x)  # Scales to [-5, 5]
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of all model components
        If pretrained is true, then the conv and batchnorm2d layers
        will not be initialized. The linear layers will though.
        """
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and not self.pretrained:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) and not self.pretrained:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze_alexnet_layers(self):
        """Freeze only the AlexNet feature layers"""
        # Freeze all parameters in the features (convolutional) part
        for param in self.features.parameters():
            param.requires_grad = False

        # Verify all other layers are still trainable
        # These will include your subject embedding, and regressor
        for name, param in self.named_parameters():
            if "features." not in name:
                param.requires_grad = True

    def forward(self, xin, subject_ids):
        # Process image through convolutional layers
        # self.print_layer("input", xin)
        x = self.features(xin)
        # for i, f in enumerate(self.features):
        #    if i == 0:
        #        x = f(xin)
        #    else:
        #        x = f(x)
        #    self.print_layer(f, x)
        x = self.avgpool(x)
        # self.print_layer("avgpool", x)
        x = torch.flatten(x, 1)
        x_regressor_conv = self.regressor_conv(x)
        # self.print_layer("regressor_conv", x_regressor_conv)

        # Get embeddings
        # subject_indices = torch.Tensor([self.subjectID2idx[subjectID] for subjectID in subject_ids]).to(self.device).long()
        if self.subject_embedding_dim:
            subject_emb = self.subject_embedding(subject_ids)
            # self.print_layer("subject_embedding", subject_emb)

            # Process through separate regressor components
            combined = torch.cat([x_regressor_conv, subject_emb], dim=1)
        else:
            combined = x_regressor_conv
        # self.print_layer("combined", combined)
        x = self.regressor_combined(combined)
        # self.print_layer("regressor_combined", x)
        output = self.regressor_final(x)
        # self.print_layer("output", output)

        return output

    def get_subjects(self):
        return self.subjects

    def print_layer(self, layer_name, feats):
        print(f"{'*'*10} layer {layer_name} {'*'*10}")
        print(f"shape: {feats.shape}")
        print(f"mean: {feats.mean()}")
        print(f"std: {feats.std()}")

    def get_subjectID2idx_mapping(self):
        return self.subjectID2idx

    def count_parameters(self):
        """
        Counts how many trainable parameters are in our network. Networks are
        often defined partly by how many trainable parameters they contain, as this
        often is a good indicator of the networks complexity.

        Returns
        -------
        int
            number of trainable weights and biases in the model

        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AlexNetCore(nn.Module):
    def __init__(self, pretrained: bool = False, add_batchnorm: bool = False):
        super(AlexNetCore, self).__init__()
        self.pretrained = pretrained
        self.add_batchnorm = add_batchnorm
        # Load the original AlexNet model
        original_alexnet = AlexNet()
        if self.pretrained:
            # Load pretrained weights if specified
            original_alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            self.features = original_alexnet.features
        else:
            # Modify features to add BatchNorm and randomize weights
            new_features = []
            for layer in original_alexnet.features:
                # If it's a Conv2d, randomize weights
                if isinstance(layer, nn.Conv2d):
                    # print(f'Randomizing layer: {layer}')
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                new_features.append(layer)
                if self.add_batchnorm and isinstance(layer, nn.Conv2d):
                    # Add BatchNorm after each Conv layer
                    bn = nn.BatchNorm2d(layer.out_channels)
                    nn.init.constant_(bn.weight, 1)
                    nn.init.constant_(bn.bias, 0)
                    new_features.append(bn)

            # Replace features with new BatchNorm-enhanced version
            self.features = nn.Sequential(*new_features)
        self.avgpool = original_alexnet.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class ResNet50Core(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(ResNet50Core, self).__init__()
        self.pretrained = pretrained

        if self.pretrained:
            original_resnet50 = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            original_resnet50 = resnet50(weights=None)
        # use all layers except last avgpool and fc
        new_features = [
            original_resnet50.conv1,
            original_resnet50.bn1,
            original_resnet50.relu,
            original_resnet50.maxpool,
            original_resnet50.layer1,
            original_resnet50.layer2,
            original_resnet50.layer3,
            original_resnet50.layer4,
        ]
        self.features = nn.Sequential(*new_features)

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet18Core(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(ResNet18Core, self).__init__()
        self.pretrained = pretrained

        if self.pretrained:
            original_resnet18 = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        else:
            original_resnet18 = resnet18(weights=None)
        # use all layers except last avgpool and fc
        new_features = [
            original_resnet18.conv1,
            original_resnet18.bn1,
            original_resnet18.relu,
            original_resnet18.maxpool,
            original_resnet18.layer1,
            original_resnet18.layer2,
            original_resnet18.layer3,
            original_resnet18.layer4,
        ]
        self.features = nn.Sequential(*new_features)

    def forward(self, x):
        x = self.features(x)
        return x


class SqueezeNet1_1Core(nn.Module):
    def __init__(self, pretrained: bool = False, add_batchnorm: bool = False):
        super(SqueezeNet1_1Core, self).__init__()
        self.pretrained = pretrained
        self.add_batchnorm = add_batchnorm

        if self.pretrained:
            original_model = squeezenet1_1(SqueezeNet1_1_Weights.IMAGENET1K_V1)
            self.features = original_model.features
        else:
            original_model = squeezenet1_1(weights=None)
            # Modify features to add BatchNorm
            new_features = []
            for layer in original_model.features:
                if isinstance(layer, nn.Conv2d):
                    # For top-level convolutions
                    new_features.append(
                        nn.Sequential(
                            layer,
                            (
                                nn.BatchNorm2d(layer.out_channels)
                                if self.add_batchnorm
                                else nn.Identity()
                            ),
                        )
                    )
                elif isinstance(layer, Fire):
                    # Create a modified copy of the Fire module
                    modified_fire = copy.deepcopy(layer)

                    if self.add_batchnorm:
                        # Add BatchNorm after squeeze
                        modified_fire.squeeze = nn.Sequential(
                            modified_fire.squeeze,
                            nn.BatchNorm2d(modified_fire.squeeze.out_channels),
                        )

                        # Add BatchNorm after expand1x1
                        modified_fire.expand1x1 = nn.Sequential(
                            modified_fire.expand1x1,
                            nn.BatchNorm2d(modified_fire.expand1x1.out_channels),
                        )

                        # Add BatchNorm after expand3x3
                        modified_fire.expand3x3 = nn.Sequential(
                            modified_fire.expand3x3,
                            nn.BatchNorm2d(modified_fire.expand3x3.out_channels),
                        )

                    new_features.append(modified_fire)
                else:
                    # Keep other layers as they are (like MaxPool, ReLU, etc.)
                    new_features.append(layer)

            # Replace features with new BatchNorm-enhanced version
            self.features = nn.Sequential(*new_features)

    def forward(self, x):
        x = self.features(x)
        return x


class SwinTCore(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(SwinTCore, self).__init__()
        self.pretrained = pretrained

        if self.pretrained:
            original_model = swin_t(Swin_T_Weights.IMAGENET1K_V1)
            self.features = original_model.features
            self.features.append(original_model.norm)
            self.features.append(original_model.permute)  # B H W C -> B C H W
        else:
            original_model = swin_t(weights=None)
            new_features = []
            for layer in original_model.features:
                new_features.append(layer)
            new_features.append(original_model.norm)
            new_features.append(original_model.permute)  # B H W C -> B C H W

            self.features = nn.Sequential(*new_features)

    def forward(self, x):
        x = self.features(x)
        return x


class C8NonSteerableCNN(torch.nn.Module):

    def __init__(self, n_feats=48):

        super(C8NonSteerableCNN, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, n_feats * 8, kernel_size=5, padding=1, bias=False),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_feats * 8, n_feats * 8, kernel_size=5, padding=2, bias=False
            ),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_feats * 8, n_feats * 8, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )

        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_feats * 8, n_feats * 8, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_feats * 8, n_feats * 8, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_feats * 8, n_feats * 8, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )

        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.block7 = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_feats * 8, n_feats * 8, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )
        self.block8 = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_feats * 8, n_feats * 8, kernel_size=3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(n_feats * 8),
            torch.nn.ReLU(inplace=True),
        )
        self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=1)

    def forward(self, input: torch.Tensor):

        x = self.block1(input)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool4(x)
        return x


class Encoder(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout

    def forward(self, x, data_key=None, detach_core=False, fake_relu=False, **kwargs):
        x = self.core(x)
        if detach_core:
            x = x.detach()
        if "sample" in kwargs:
            x = self.readout(x, sample=kwargs["sample"])
        else:
            x = self.readout(x)
        return x

    def count_parameters(self):
        """
        Count and print the number of trainable parameters in the core and readout components.

        Returns:
            total_params (int): Total number of trainable parameters in the model
        """
        # Count core parameters
        core_params = sum(p.numel() for p in self.core.parameters() if p.requires_grad)

        # Count readout parameters
        readout_params = sum(
            p.numel() for p in self.readout.parameters() if p.requires_grad
        )

        # Total parameters
        total_params = core_params + readout_params

        # Print the information
        print(f"Trainable parameters:")
        print(f"  Core:    {core_params:,}")
        print(f"  Readout: {readout_params:,}")
        print(f"  Total:   {total_params:,}")

        return total_params


class EncoderMultiHead(nn.Module):
    def __init__(
        self,
        core,
        readout_class,
        subjectID2idx,
        confidence_scores=None,
        **readout_kwargs,
    ):
        super().__init__()
        self.subjectID2idx = subjectID2idx
        self.idx2subjectID = {
            idx: subjectID for subjectID, idx in subjectID2idx.items()
        }
        self.core = core
        self.subject_readouts = nn.ModuleDict(
            {
                str(subjectID): readout_class(**readout_kwargs)
                for subjectID in self.subjectID2idx.keys()
            }
        )
        self.confidence_scores = confidence_scores or {
            subjectID: 1.0 for subjectID in self.subjectID2idx.keys()
        }
        self.training = False

    def forward_legacy(
        self,
        x,
        subjectID_value=None,
        data_key=None,
        detach_core=False,
        fake_relu=False,
        **kwargs,
    ):
        """
        implementation of Ben's original forward pass
        """
        # Get core features
        core_output = self.core(x)
        if subjectID_value is None:
            subjectID_value = list(self.subjectID2idx.values())
        # Register a backward hook for gradient scaling on core output
        if not detach_core and self.training:
            batch_confidence_scores = []
            for i in range(len(subjectID_value)):
                subjectID_str = self.idx2subjectID[subjectID_value[i]]
                confidence = self.confidence_scores.get(subjectID_str, 1.0)
                batch_confidence_scores.append(confidence)

            # Create tensor of confidence scores for the batch
            confidence_tensor = torch.tensor(
                batch_confidence_scores,
                device=core_output.device,
                dtype=core_output.dtype,
            )

            # Register hook to scale gradients at the boundary between core and readouts
            core_output.register_hook(
                lambda grad: grad * confidence_tensor.view(-1, 1, 1, 1)
            )

        if detach_core:
            core_output = core_output.detach()

        # Process each sample in the batch based on its subject ID
        batch_size = core_output.size(0)
        outputs = []
        for i in range(batch_size):
            # Get subject information for this sample
            # raise AssertionError(self.idx2subjectID, subjectID_value[i])
            subjectID_str = self.idx2subjectID[subjectID_value[i]]

            # Process this sample with its corresponding readout
            if "sample" in kwargs:
                sample_output = self.subject_readouts[subjectID_str](
                    core_output[i : i + 1], sample=kwargs["sample"]
                )
            else:
                sample_output = self.subject_readouts[subjectID_str](
                    core_output[i : i + 1]
                )

            outputs.append(sample_output)

        # Concatenate all outputs along the batch dimension
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        names_and_subjects: dict[str, Union[list, str]] = None
    ):

        assert names_and_subjects is not None, f"\033[33mPlease provide a names_and_subjects argument. It's a dictionary that maps dataset names (str) to subject_ids (list of integers or 'all').\nExample: {{'NSD': [1, 2], 'deeprecon': 'all'}}\033[0m"
        
        for name in names_and_subjects.keys():
            assert name in list(num_subjects.keys()), f"Dataset name {name} is not valid. Please choose from {list(num_subjects.keys())}."

        outputs = {}

        core_output = self.core(x)

        for dataset_name in names_and_subjects.keys():
            outputs[dataset_name] = {}

            if isinstance(names_and_subjects[dataset_name], str):
                assert names_and_subjects[dataset_name] == "all", f"Invalid value {names_and_subjects[dataset_name]} for dataset {dataset_name}. Must be 'all' or a list of subject IDs."
                names_and_subjects[dataset_name] = [i for i in range(1, num_subjects[dataset_name]+1)]
            else:
                pass

            for subject_id_int in names_and_subjects[dataset_name]:
                assert subject_id_int in range(1, num_subjects[dataset_name]+1), f"For {dataset_name}, subject_id {subject_id_int} is out of range. Must be between 1 and {num_subjects[dataset_name]}."

                subjectID_str = f"sub-{subject_id_int:02}_{inference_dataset_name_mapping[dataset_name]}"

                assert subjectID_str in self.subject_readouts, f"Subject ID {subjectID_str} not found in readouts: {list(self.subject_readouts.keys())}" 

                single_subject_output = self.subject_readouts[subjectID_str](
                    core_output
                )
                outputs[dataset_name][f"sub-{subject_id_int:02}"] = single_subject_output


        assert len(outputs) ==  len(names_and_subjects), f"Expected outputs from {len(names_and_subjects)} datasets in outputs, but got {len(outputs)}."
        return outputs

    def count_parameters(self):
        """
        Count and print the number of trainable parameters in the core and readout components.
        """
        # Count core parameters
        core_params = sum(p.numel() for p in self.core.parameters() if p.requires_grad)

        # Count readout parameters (across all subject-specific readouts)
        readout_params = sum(
            p.numel()
            for name, module in self.subject_readouts.items()
            for p in module.parameters()
            if p.requires_grad
        )

        # Total parameters
        total_params = core_params + readout_params

        # Print the information
        print(f"Trainable parameters:")
        print(f"  Core:    {core_params:,}")
        print(f"  Readout: {readout_params:,}")
        print(f"  Total:   {total_params:,}")

        return total_params
