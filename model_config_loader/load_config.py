# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/1 2:44 PM
# @File: load_config
# @Email: mlshenkai@163.com
import yaml


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

    def add(self, other_config):
        self._dict.update(other_config._dict)
        self._build_attr()


# if __name__ == "__main__":
#     config = Config("./config_sample.yaml","a")
#     b_config = Config("./config_sample.yaml","b")
#     config.add(b_config)
#     print(config)
