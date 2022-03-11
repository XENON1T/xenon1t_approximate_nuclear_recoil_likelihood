import numpy as np
import matplotlib.pyplot as plt
import json
import logging

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


logger = logging.getLogger('xe_bin_logger')
logging.basicConfig(format='%(levelname)s:\t%(funcName)s\t | %(message)s')


class BinnedAxis:
    def __init__(self, edges, title='X', units=''):
        self.title = title
        self.edges = np.round(edges,6)
        self.units = units

    @classmethod
    def from_dict(cls, d):
        title = d['title']
        edges = np.linspace(d['low_edge'], d['high_edge'], d['N'])
        units = d['unit']
        return cls(edges, title=title, units=units)

    @property
    def bin_centers(self):
        return 0.5 * (self.edges[:-1] + self.edges[1:])

class Hist2DCollection:
    """
    Utility class to handle input of likelihood and migration matrices
    Matrices are stored as histrograms in JSON files.
    """

    def __init__(self, x: BinnedAxis, y: BinnedAxis, zs: dict,
                     title="values", units='', name='2D histogram collection'):
        """ 
        Read a pre-defined migration matrix or likelihood matrix.
            :x: BinnedAxis, x axis
            :y: BinnedAxis, y axis
            :values: ndarray, matrix with histrogream values
            :title: string, label of the histogram (optional)
            :units: string, units of histogram values (optional)
            :name: string, histogram name (optional)
        """

        self.x = x
        self.y = y
        self.zs = zs
        self.title = title
        self.units = units
        self.name = name

    @classmethod
    def from_file(cls, path, name=''):
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data, name=name)

    @classmethod
    def from_package(cls, module, fname, name=''):
        data = json.load(pkg_resources.open_text(module, fname))
        return cls.from_dict(data, name=name)

    @classmethod
    def from_dict(cls, data, name=''):
        x = BinnedAxis.from_dict(data['Xedges'])
        y = BinnedAxis.from_dict(data['Yedges'])
        zs = {k:np.array(v) for k,v in data['Z']['Data'].items()}
        title = data['Z']['title']
        units = data['Z']['unit']
        return cls(x,y,zs,title=title,units=units,name=name)

    @property
    def x_bins(self):
        return self.x.bin_centers

    @property
    def y_bins(self):
        return self.y.bin_centers

    @property
    def x_edges(self):
        return self.x.edges

    @property
    def y_edges(self):
        return self.y.edges
    
    def __len__(self):
        return len(self.zs)

    def __repr__(self) -> str:
        return self.name

    def __iter__(self):
        for idx in self.zs:
            yield self[idx]

    def __getitem__(self, idx):
        idx = str(idx)
        if idx not in self.zs:
            raise KeyError(f'No data exists for index {idx}')
        return self.zs[idx]

    def plot(self, idx=0, title="", show=True):
        """
        Display matrix
            :title: string, plot title
        """
        plt.title(f"{self.name} #{idx} {title}")
        plt.pcolormesh(self.x.edges, self.y.edges, self[idx].T, cmap="inferno_r")
        plt.xlabel(f"{self.x.title} [{self.x.units}]")
        plt.ylabel(f"{self.y.title} [{self.y.units}]")
        plt.colorbar(label=f"{self.title} [{self.units}]")
        if show:
            plt.show()

