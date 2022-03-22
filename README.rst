xe-likelihood
-------------

Binwise approximations of the XENON1T likelihood and XENONnT projections for fast inference on arbitrary models.


Example XENON1T based inference

.. code-block:: python

    from xe_likelihood import BinwiseInference, Spectrum

    spectrum = Spectrum.from_wimp(mass=50)
    inference = BinwiseInference.from_xenon1t_sr(spectrum=spectrum)
    inference.plot_summary(show=True)


Will produce something like this:

.. image:: images/XENON1T_inference.png
  :width: 800
  :alt: Xenon1T Inference

========
Citation
========

If you use this package, please cite the following papers

