# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from .ofa_proxyless import OFAProxylessNASNets
from .ofa_mbv3 import OFAMobileNetV3, OFAEEMobileNetV3
from .ofa_resnets import OFAResNets
from .ofa_resnets_he import OFAResNetsHE
from .ofa_mbv3_he import OFAMobileNetV3HE