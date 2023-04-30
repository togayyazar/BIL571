import warnings

import numpy as np
import pandas as pd
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
import dowhy
import statsmodels.api

warnings.filterwarnings("ignore")  # silence warnings


def change_sm():
    data = pd.read_csv('data/dataset/binary_set/X_change_numerical.csv')
    sm = from_pandas(data)
    viz = plot_structure(
        sm,
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK,
    )
    viz.draw("thresh_olded.png")


def diff_sm():
    data = pd.read_csv('data/dataset/binary_set/X_change_numerical.csv')
    data = data.replace(0, 1e-8)
    data = data.apply(np.log)
    data = data.fillna(method='bfill')
    data = data.fillna(method='ffill')
    sm = from_pandas(data)
    k = sm.edges

    viz = plot_structure(
        sm,
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.NORMAL,
    )
    viz.draw("time_series.png")

    causal_graph = """digraph {
    F_1;
    F_2;
    S_1;
    S_2;
    F_1 -> F_2;
    F_1 -> S_2;
    F_1 -> S_1;
    F_2 -> S_1;
    F_2 -> S_2;
    }"""
    model = dowhy.CausalModel(data=data,
                              graph=causal_graph.replace("\n", " "),
                              treatment="F_1",
                              outcome="S_1"
                              )

    identified_estimand = model.identify_effect()
    print(identified_estimand)

    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression",
                                     target_units="ate")
    print(estimate)


def diff_sign():
    data = pd.read_csv('data/dataset/binary_set/X_change_sign.csv')
    causal_graph = """digraph {
    F_1;
    F_2;
    S_1;
    S_2;
    F_2 -> F_1;
    F_1 -> S_1;
    F_2 -> S_2;
    F_2 -> S_1;
    S_2 -> S_1;
    }"""

    model = dowhy.CausalModel(data=data,
                              graph=causal_graph.replace("\n", " "),
                              treatment="F_2",
                              outcome="F_1"
                              )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    estimate = model.estimate_effect(identified_estimand,
                                     control_value=-1,
                                     treatment_value=1,
                                     method_name="backdoor.linear_regression",
                                     target_units='ate')
    print(estimate)

    refutation = model.refute_estimate(identified_estimand, estimate, method_name="bootstrap_refuter",
                                       placebo_type="permute", num_simulations=20)

    print(refutation)


diff_sign()
