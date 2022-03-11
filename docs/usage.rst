=====
Usage
=====

To use xe-likelihood in a project::

    import xe_likelihood


For XENON1T based inference

.. code-block:: python

    from xe_likelihood import BinwiseInference

    inference = BinwiseInference.from_xenon1t_sr(spectrum=CALLABLE)
    inference.compute_ul(cl=0.1, asymptotic=False)

For XENONnT based inference

.. code-block:: python

    from xe_likelihood import BinwiseInference

    inference = BinwiseInference.from_xenonnt_mc(spectrum=CALLABLE)
    inference.compute_ul(cl=0.1, asymptotic=False)