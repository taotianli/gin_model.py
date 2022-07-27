"""The ``dgl.data`` package contains datasets hosted by DGL and also utilities
for downloading, processing, saving and loading data from external resources.
"""

from __future__ import absolute_import

from . import citation_graph as citegrh
from .citation_graph import CoraBinary, CitationGraphDataset
from .minigc import *
from .tree import SST, SSTDataset
from .utils import *
from .sbm import SBMMixture, SBMMixtureDataset
from .reddit import RedditDataset
from .ppi import PPIDataset, LegacyPPIDataset
from .tu import TUDataset, LegacyTUDataset
from .gnn_benchmark import AmazonCoBuy, CoraFull, Coauthor, AmazonCoBuyComputerDataset, \
    AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset, CoauthorCSDataset, CoraFullDataset
from .karate import KarateClub, KarateClubDataset
from .gindt import GINDataset
from .bitcoinotc import BitcoinOTC, BitcoinOTCDataset
from .gdelt import GDELT, GDELTDataset
from .icews18 import ICEWS18, ICEWS18Dataset
from .qm7b import QM7b, QM7bDataset
from .qm9 import QM9, QM9Dataset
from .qm9_edge import QM9Edge, QM9EdgeDataset
from .dgl_dataset import DGLDataset, DGLBuiltinDataset
from .citation_graph import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from .knowledge_graph import FB15k237Dataset, FB15kDataset, WN18Dataset
from .rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from .fraud import FraudDataset, FraudYelpDataset, FraudAmazonDataset
from .fakenews import FakeNewsDataset
from .csv_dataset import CSVDataset
from .adapter import *
from .synthetic import BAShapeDataset, BACommunityDataset, TreeCycleDataset, TreeGridDataset, BA2MotifDataset
from .wikics import WikiCSDataset

def register_data_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help=
        "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit"
    )


def load_data(args):
    if args.dataset == 'cora':
        return citegrh.load_cora()
    elif args.dataset == 'citeseer':
        return citegrh.load_citeseer()
    elif args.dataset == 'pubmed':
        return citegrh.load_pubmed()
    elif args.dataset is not None and args.dataset.startswith('reddit'):
        return RedditDataset(self_loop=('self-loop' in args.dataset))
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
