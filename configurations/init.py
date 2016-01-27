
def crps(cdf_predictions, cdf_targets):
    return T.mean((cdf_predictions - cdf_targets) ** 2, axis=[0, 1])