from typing import NamedTuple


class _MODULE_KEYS(NamedTuple):
    X_KEY: str = "x"
    Y_KEY: str = "y"
    # inference
    M_KEY: str = "m"
    B_KEY: str = "b"
    V_KEY: str = "v"
    QM_KEY: str = "qm"
    QB_KEY: str = "qb"
    QV_KEY: str = "qv"
    QLM_KEY: str = "qlm"
    QLB_KEY: str = "qlb"
    METABOLIC_LIBRARY_KEY: str = "metabolic_library"
    BACKGROUND_LIBRARY_KEY: str = "background_library"
    BATCH_INDEX_KEY: str = "batch_index"
    CONT_COVS_KEY: str = "cont_covs"
    CAT_COVS_KEY: str = "cat_covs"
    MET_SIZE_FACTOR_KEY: str = "met_size_factor"
    BACK_SIZE_FACTOR_KEY: str = "back_size_factor"
    # generative
    PX_KEY: str = "px"
    PM_KEY: str = "pm"
    PB_KEY: str = "pb"
    PV_KEY: str = "pv"
    PLM_KEY: str = "pl_met"
    PLB_KEY: str = "pl_back"
    PX_MET_KEY: str = "px_met"
    PX_BACK_KEY: str = "px_back"
    PG_KEY: str = "pg"
    # graph
    EIDX_KEY: str = "eidx"
    EWT_KEY: str = "ewt"
    ESGN_KEY: str = "esgn"
    # loss
    KL_M_KEY: str = "kl_m"
    KL_B_KEY: str = "kl_b"
    KL_V_KEY: str = "kl_v"
    KL_LM_KEY: str = "kl_lm"
    KL_LB_KEY: str = "kl_lb"
    ENZYME_ACTIVITY_KEY: str = "enzyme_activity"

class _METABOLIC_REGISTRY_KEYS(NamedTuple):
    MET_SIZE_FACTOR_KEY: str = "met_size_factor"
    BACK_SIZE_FACTOR_KEY: str = "back_size_factor"
    
class _GRAPH_REGISTRY_KEYS(NamedTuple):
    EIDX_KEY: str = "eidx"
    EWT_KEY: str = "ewt"
    ESGN_KEY: str = "esgn"

GRAPH_REGISTRY_KEYS = _GRAPH_REGISTRY_KEYS()
MODULE_KEYS = _MODULE_KEYS()
METABOLIC_REGISTRY_KEYS = _METABOLIC_REGISTRY_KEYS()
