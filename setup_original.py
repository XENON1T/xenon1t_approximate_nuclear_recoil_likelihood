from setuptools import setup, find_packages
setup(
    name = "xe_likelihood",
    version = "0.1",
    packages = ["xe_likelihood"],
    package_dir = dict(
        xe_likelihood = "xe_likelihood"),
    package_data = dict(
        xe_likelihood = ["data/*",
                         "data/ancillary_data/*",
                         "data/xe1t_nr_1ty_result/*",
                         "data/xent_nr_20ty_projection/*"
        ]
    ),
    include_package_data = True,
    author = "XENON collaboration",
    description = "Approximate NR likelihood for XENON1T",
    )
