import numpy as np
from dataclasses import dataclass, field, replace
from typing import Union, Optional, Dict, Any, Tuple, TypeVar, Type
from collections import defaultdict
import hickle as hkl

def _san(var,as_np=False,wrap_none=False):
    """
    Sanitize array
    """
    if var is None: return [None] if wrap_none else None
    if not isinstance(var, list): var = [var]
    if as_np: var = np.array(var)
    return var

class Savable():
    def save_path(self,**kwargs):
        return "./tmp.h5"
    
    def save(self, path: str=None):
        hkl.dump(self.__dict__, path or self.save_path())

    def load(self, path: str=None):
        try:
            self.__dict__.clear()
            self.__dict__.update(hkl.load(path or self.save_path()))
        except Exception as e:
            print(f"Failed to load {self.__class__} object: {e}")
    

@dataclass(frozen=True)
class Config(Savable):
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


K = TypeVar("K", bound="GenericKey")

@dataclass(frozen=True)
class GenericKey:
    """
    Indexing object.
    """

    def __str__(self):
        pass

    def __eq__(self,other):
        try:
            return all(getattr(self, f) == getattr(other, f) for f in self.__dataclass_fields__)
        except:
            return False # NOTE no type check, allows comparison with other 'key' classes

    def matches(self, **kwargs) -> bool:
        """Check if this key matches given criteria (for filtering)"""
        for k, v in kwargs.items():
            if isinstance(v, list):
                if getattr(self, k, None) not in v:
                    return False
            else:
                if v is not None and getattr(self, k, None) != v:
                    return False
        return True

    def _new(self: K, **kwargs) -> K:
        return replace(self, **kwargs)

    def get(self: K, *dimensions) -> K:
        return type(self)(**{d: getattr(self, d, None) for d in dimensions})

    def remove(self: K, *dimensions) -> K:
        return type(self)(**{
            f: getattr(self, f)
            for f in self.__dataclass_fields__
            if f not in dimensions
        })

    def add(self: K, **kwargs) -> K:
        for k in kwargs:
            assert getattr(self, k) is None
        return self._new(**kwargs)

    def change(self: K, **kwargs) -> K:
        return self._new(**kwargs)

    def nd(self) -> K:
        return self.get('session')
    
    @staticmethod
    def groupby(data, *dimensions) -> dict:
        """
        Group keys by specified dimensions.
        
        Example:
            dataset.groupby('session', 'epoch')
            # Returns: {('s1', 'pre'): {key: data, ...}, ('s1', 'post'): {...}}
        """
        groups = defaultdict(dict)
        for key, value in data.items():
            # group_key = tuple(getattr(key, dim, None) for dim in dimensions)
            groups[key.get(*dimensions)][key] = value
        return dict(groups)


class AnalysisDataset(Savable):
    """
    Container for an analysis dataset.
    """
    def __init__(self, conf=None):
        self.data: Dict[K, Any] = {}
        self._conf = conf

    def __len__(self):
        return len(self.data)
    
    def example(self,field=None) -> Any:
        """Get an example from data or another field"""
        if field:
            item = next(iter(self.__dict__[field].keys()))
            return self.__dict__[field].get(item,None)
        return self.data.get(next(iter(self.data.keys())),None)

    def filter(self, **criteria) -> Dict[K, Any]:
        """
        Filter data by any combination of key attributes.
        
        Example:
            dataset.filter(session='s1', epoch='pre')
            dataset.filter(analysis_type='correlogram', neuron_type='pyramidal')
        """
        return {k: v for k, v in self.data.items() if k.matches(**criteria)}
    
    def filter_keys(self, **criteria) -> list[K]:
        """Get all keys matching criteria"""
        return [k for k in self.data.keys() if k.matches(**criteria)]
    
    def groupby(self, *dimensions, source='data') -> Dict[K, Dict[K, Any]]:
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


