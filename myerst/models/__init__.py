from myerst.models.gcn import GCN, build_norm_adj, train_gcn

try:
    from myerst.models.stagate_lite import (GATLayer, STAGATELite, HeadOnEmbedding,
                                            train_stagate, adj_binary)
    from myerst.models.stagate_official_pt import STAGATEOfficialPT, train_stagate_official
    __all__ = ["GCN", "build_norm_adj", "train_gcn",
               "GATLayer", "STAGATELite", "HeadOnEmbedding", "train_stagate", "adj_binary",
               "STAGATEOfficialPT", "train_stagate_official"]
except ImportError:
    __all__ = ["GCN", "build_norm_adj", "train_gcn"]

