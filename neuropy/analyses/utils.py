import numpy as np

def _san(var,as_np=False,wrap_none=False):
    """
    Sanitize array
    """
    if var is None: return [None] if wrap_none else None
    if not isinstance(var, list): var = [var]
    if as_np: var = np.array(var)
    return var


@dataclass(frozen=True)
class Config:
    def __str__(self):
        s=""
        for key, val in self.__dict__.items():
            s+=f"{key}: {val}\n"
        s+=f"config file: {self.filepath}\n"
        return s

    def save(self,filepath):
        with h5py.File(filepath, "w") as f:
            for k, v in self.__dict__.items():
                try:
                    f.create_dataset(k, data=v)
                except TypeError:
                    f.attrs[k] = str(v)  # fallback for non-array data

    @classmethod
    def load(cls, filepath):
        obj = cls.__new__(cls)  # bypass __init__
        with h5py.File(filepath, "r") as f:
            for k, v in f.items():
                obj.__dict__[k] = np.array(v)
            for k, v in f.attrs.items():
                obj.__dict__[k] = v
        return obj    


class AnalysisDataset:
    """
    Container for an analysis dataset.
    """
    def __init__(self, conf=None):
        self.data: Dict[Key, Any] = {}
        self._conf = conf

    def __len__(self):
        return len(self.data)
    
    def example(self,field=None) -> Any:
        """Get an example from data or another field"""
        if field:
            item = next(iter(self.__dict__[field].keys()))
            return self.__dict__[field].get(item,None)
        return self.data.get(next(iter(self.data.keys())),None)

    def filter(self, **criteria) -> Dict[Key, Any]:
        """
        Filter data by any combination of key attributes.
        
        Example:
            dataset.filter(session='s1', epoch='pre')
            dataset.filter(analysis_type='correlogram', neuron_type='pyramidal')
        """
        return {k: v for k, v in self.data.items() if k.matches(**criteria)}
    
    def filter_keys(self, **criteria) -> list[Key]:
        """Get all keys matching criteria"""
        return [k for k in self.data.keys() if k.matches(**criteria)]
    
    def groupby(self, *dimensions, source='data') -> Dict[Key, Dict[Key, Any]]:
        """
        Group data by specified dimensions.
        
        Example:
            dataset.groupby('session', 'epoch')
            # Returns: {('s1', 'pre'): {key: data, ...}, ('s1', 'post'): {...}}
        """
        items = getattr(self, source)
        groups = defaultdict(dict)
        for key, value in items.items():
            # group_key = tuple(getattr(key, dim, None) for dim in dimensions)
            groups[key.get(*dimensions)][key] = value
        return dict(groups)
            
    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self,conf):
        ans = input("Clear all datafields with the new config? [y/n]").lower()
        if ans=='y':
            self.data={}
            self.spurious={}
            self.auto={}
            self.connectivity={}
            print(f'{self.__class__.__name__}: all data fields are cleared')
        self._conf = conf
        print(f"{self.__class__.__name__}Config changed, which might create inconsistencies between existing data and config. Rerun if necessary.")


