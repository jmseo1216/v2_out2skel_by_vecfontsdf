"""VecFontSDF-style ResNet encoder + class-conditional segment decoder.

입력:
    image: [B, 1, 128, 128]
    codepoint: [B]  예: 65, 66, ...

출력:
    pred_exist_logits: [B, K]
    pred_segments: [B, K, 4]

각 segment:
    [x0, y0, x1, y1] in image coordinate [0, image_size]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """VecFontSDF 스타일의 3x3 convolution residual block.

    channel 수가 바뀌거나 stride가 1이 아닐 때는 projection shortcut을 사용한다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 원본 VecFontSDF 코드는 channel이 바뀔 때만 shortcut을 썼지만,
        # stride가 2인 경우 spatial size도 바뀌므로 projection shortcut을 쓰는 게 안전하다.
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """VecFontSDF 스타일 초기화."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.leaky_relu(y, negative_slope=0.2, inplace=True)

        y = self.conv2(y)
        y = self.bn2(y)

        y = y + identity
        y = F.relu(y, inplace=True)
        return y


class FontSkeletonModel(nn.Module):
    """VecFontSDF 구조를 skeleton segment 예측에 맞게 수정한 모델.

    VecFontSDF 원래 출력:
        [B, v_dim * p_dim, 6]
        각 primitive가 parabola parameter 6개를 예측

    여기서 필요한 출력:
        pred_exist_logits: [B, K]
        pred_segments: [B, K, 4]

    따라서 decoder output dimension은:
        K * 5

    각 segment마다:
        [exist_logit, x0, y0, x1, y1]
    """

    def __init__(
        self,
        num_segments: int = 48,
        image_size: int = 128,
        fc_channel: int = 1024,
        char_categories: int = 94,
        codepoint_min: int = 33,
        use_class_condition: bool = True,
    ) -> None:
        super().__init__()

        self.num_segments = num_segments
        self.image_size = image_size
        self.fc_channel = fc_channel
        self.char_categories = char_categories
        self.codepoint_min = codepoint_min
        self.use_class_condition = use_class_condition

        # VecFontSDF style encoder
        # image: [B,1,128,128]
        # layer0: [B,64,64,64]
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
        )

        # block1: [B,128,32,32]
        self.block1 = ResBlock(64, 128, kernel_size=(3, 3), stride=(2, 2))

        # block2: [B,256,16,16]
        self.block2 = ResBlock(128, 256, kernel_size=(3, 3), stride=(2, 2))

        # block3: [B,512,8,8]
        self.block3 = ResBlock(256, 512, kernel_size=(3, 3), stride=(2, 2))

        # block4: [B,512,4,4]
        self.block4 = ResBlock(512, 512, kernel_size=(3, 3), stride=(2, 2))

        # block5: [B,512,2,2]
        self.block5 = ResBlock(512, 512, kernel_size=(3, 3), stride=(2, 2))

        cond_dim = char_categories if use_class_condition else 0

        self.fc_layer1 = nn.Linear(512 + cond_dim, fc_channel)
        self.fc_layer2 = nn.Linear(fc_channel, num_segments * 5)

        self._init_weights()

    def _init_weights(self) -> None:
        """VecFontSDF 스타일 초기화."""
        for m in self.layer0.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)

        for fc in (self.fc_layer1, self.fc_layer2):
            nn.init.normal_(fc.weight, mean=0.0, std=0.02)
            nn.init.constant_(fc.bias, val=0.0)

    def codepoint_to_onehot(self, codepoint: torch.Tensor) -> torch.Tensor:
        """codepoint tensor를 one-hot character condition으로 변환한다.

        예:
            codepoint = 33 -> index 0
            codepoint = 65 -> index 32
            codepoint = 126 -> index 93
        """
        if codepoint.ndim == 0:
            codepoint = codepoint.unsqueeze(0)

        codepoint = codepoint.long()
        cls_idx = codepoint - self.codepoint_min

        if torch.any(cls_idx < 0) or torch.any(cls_idx >= self.char_categories):
            bad = codepoint[(cls_idx < 0) | (cls_idx >= self.char_categories)]
            raise ValueError(
                f"codepoint가 지원 범위를 벗어났습니다. "
                f"지원 범위: [{self.codepoint_min}, {self.codepoint_min + self.char_categories - 1}], "
                f"bad codepoints: {bad.detach().cpu().tolist()}"
            )

        return F.one_hot(cls_idx, num_classes=self.char_categories).float()

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """outline image를 512차원 feature로 encoding한다."""
        y = self.layer0(image)
        y = F.relu(y, inplace=True)

        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)

        y = F.adaptive_avg_pool2d(y, output_size=(1, 1))
        y = y.flatten(1)  # [B,512]
        return y

    def forward(
        self,
        image: torch.Tensor,
        codepoint: torch.Tensor | None = None,
        clss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            image:
                [B,1,H,W] outline image
            codepoint:
                [B] ASCII codepoint. 예: 65
            clss:
                [B,char_categories] one-hot vector.
                codepoint 대신 직접 one-hot을 넣고 싶을 때 사용.

        Returns:
            pred_exist_logits:
                [B,K]
            pred_segments:
                [B,K,4], image 좌표계 [0,image_size]
        """
        feat = self.encode_image(image)  # [B,512]

        if self.use_class_condition:
            if clss is None:
                if codepoint is None:
                    raise ValueError(
                        "use_class_condition=True이면 forward에 codepoint 또는 clss를 넣어야 합니다. "
                        "예: model(input_image, codepoint=batch['codepoint'])"
                    )
                clss = self.codepoint_to_onehot(codepoint.to(image.device))

            clss = clss.to(device=image.device, dtype=feat.dtype)
            feat = torch.cat((feat, clss), dim=1)  # [B,512+char_categories]

        y = self.fc_layer1(feat)
        y = F.relu(y, inplace=True)

        y = self.fc_layer2(y)
        y = y.view(image.shape[0], self.num_segments, 5)

        pred_exist_logits = y[..., 0]

        # coordinate raw output을 sigmoid로 [0,1]에 넣고,
        # image coordinate [0,image_size]로 변환한다.
        pred_segments = torch.sigmoid(y[..., 1:5]) * float(self.image_size)

        return pred_exist_logits, pred_segments