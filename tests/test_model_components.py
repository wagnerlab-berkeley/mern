import torch

from mern._base_components import (
    RxnsToGenesLayer
)

def test_rxns_to_genes_layer():
    genes = ['gene1', 'gene2', 'gene3', 'gene4']
    rxns = {0: 'rxn1', 1: 'rxn2', 2: 'rxn3'}
    rxns_to_genes = {'rxn1': ['gene1', 'gene2'], 'rxn2': ['gene2', 'gene3'], 'rxn3': ['gene4']}
    layer = RxnsToGenesLayer(genes, rxns, rxns_to_genes)
    # make weights positive to start for test senstivity, they get clamped in forward
    layer.linear.weight = torch.nn.Parameter(torch.abs(layer.linear.weight))

    # 20 cells by 3 rxns
    x = torch.randn(20, 3)

    y = layer(x)
    assert y.shape == (20, 4)

    # check loss is correct
    weights = layer.linear.weight.cpu().detach()
    # shape genes by rxns
    expected_loss = weights[2, 0].abs() + weights[3, 0].abs() + weights[0,1].abs() + weights[3,1].abs() + weights[0,2].abs() + weights[1,2].abs() + weights[2,2].abs()
    print(layer.linear.weight)
    actual_loss = layer.reg_subset_loss(norm=1).cpu().detach()
    print(layer.linear.weight)

    print(f"Actual loss: {actual_loss}")
    print(f"Expected loss: {expected_loss}")
    print(f"Difference: {actual_loss - expected_loss}")
    print(f"Relative difference: {(actual_loss - expected_loss) / expected_loss}")
    assert torch.allclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-8)
