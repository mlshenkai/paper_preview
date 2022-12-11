# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/1 2:44 PM
# @File: load_config
# @Email: mlshenkai@163.com
import yaml
from detectron2.config import CfgNode


class Config:
    def __init__(self, config, namespace: str = None):
        if isinstance(config, str):
            config_dict = self._load(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError("config 必须是yaml文件或者dict")
        if namespace is not None:
            config_dict = {namespace: config_dict}
        self.__dict__.update(config_dict)
        self._dict = config_dict
        self._build_attr()

    def _build_attr(self):
        for key, value in self._dict.items():
            if isinstance(value, (list, tuple)):
                setattr(
                    self, key, [Config(x) if isinstance(x, dict) else x for x in value]
                )
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)
        self.dict_path = []
        self.get_dict_path(self._dict, [], self.dict_path)

    @staticmethod
    def _load(config_file_path):
        try:
            stream = open(config_file_path, "r")
            data = yaml.load_all(stream, yaml.FullLoader)
            yaml_config = dict()
            for doc in data:
                for k, v in doc.items():
                    yaml_config[k] = v
            return yaml_config
        except FileNotFoundError:
            raise FileNotFoundError("配置文件未找到...")

    def add(self, config_name, other_config):
        self._dict.update({config_name: other_config})
        self._build_attr()

    @staticmethod
    def get_dict_path(self_dict: dict, predecessors: list, path_results: list):
        for key, value in self_dict.items():
            if isinstance(value, dict):
                Config.get_dict_path(value, predecessors + [key], path_results)
            else:
                path_results.append({"path": predecessors + [key], "value": value})

    def update_detectron_config(self, cfg: CfgNode):
        for path_info in self.dict_path:
            path = path_info["path"]
            value = path_info["value"]
            _cfg = cfg
            for i in range(len(path) - 1):
                _cfg = _cfg.__getattr__(path[i])
            _cfg.__setattr__(path[-1], value)
        return cfg

    def __getattr__(self, item):
        return None

