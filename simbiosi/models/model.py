from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from torch import nn
from torch.nn import functional as F

from simbiosi.models.layers import (
    BasicConv2d,
    Block8,
    Block17,
    Block35,
    Mixed_6a,
    Mixed_7a,
)


@register_model("inception_resnet")
class InceptionResnetV1(BaseFairseqModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # Instantiate the model
        model = cls(
            num_classes=task.num_classes,
        )
        return model

    def __init__(
        self,
        num_classes,
        dropout_prob=0.6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedder = InceptionResnetEmbedder(dropout_prob=dropout_prob)
        self.logits = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.embedder(x)
        x = self.logits(x)
        return x


@register_model_architecture("inception_resnet", "inception_resnet_v1")
def register_inception_resnet_v1_architecture(args):
    pass


class InceptionResnetEmbedder(nn.Module):
    def __init__(
        self,
        dropout_prob=0.6,
    ):
        super().__init__()

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        return x
        # if self.classify:
        #     x = self.logits(x)
        # else:
        #     x = F.normalize(x, p=2, dim=1)
        # return x
