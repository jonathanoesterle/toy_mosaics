import numpy as np

from plot import plot_mosaics
from simulate import simulate_rgc_mosaics

# Example usage
if __name__ == "__main__":
    np.random.seed(300)

    # Parameters
    n_mosaics = 3
    center_noise = 0.  # Position noise
    diameter_noise = 0.

    mean_diameters = [10., 15.0, 20.0]  # Mean diameter for each mosaic
    n_missing_list = [2, 2, 0]  # Missing cells per mosaic
    overlap_factors = [1., 1.2, 1.5]  # No overlap, 15% overlap, 10% gaps

    # Generate mosaics
    mosaics = simulate_rgc_mosaics(
        n_mosaics=n_mosaics,
        mean_diameters=mean_diameters,
        center_noise=center_noise,
        diameter_noise=diameter_noise,
        n_missing_list=n_missing_list,
        overlap_factors=overlap_factors,
    )

    plot_mosaics(mosaics, mean_diameters)

