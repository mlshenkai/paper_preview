# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/11/1 2:44 PM
# @File: load_config
# @Email: mlshenkai@163.com
from typing import Any

import yaml


class Config:
    def __init__(self, param: dict):
        for k, v in param.items():
            setattr(self, k, v)

    def __getattr__(self, item):
        return None

    # def __add__(self, other):
    #     if isinstance(other, Config):
    #         for other.__dict__


class ConfigJson:
    def __init__(self, config_path: [str, dict]):
        if isinstance(config_path, str):
            config = self.load_config(config_path)
        else:
            config = config_path
        self.cfg_json = config
        self.cfg = self.to_obj()

    @staticmethod
    def load_config(fila_path):
        try:
            stream = open(fila_path, "r")
            data = yaml.load_all(stream, yaml.FullLoader)
            config = dict()
            for doc in data:
                for k, v in doc.items():
                    config[k] = v
            return config
        except Exception as e:
            raise RuntimeError("配置加载出错")

    def add_config(self, config: [str, dict]):
        if isinstance(config, str):
            new_config = self.load_config(config)
        else:
            new_config = config
        self.cfg_json.update(new_config)
        self.cfg = self.to_obj()

    def to_obj(self) -> Any:
        return Config(self.cfg_json)


if __name__ == "__main__":
    config = ConfigJson(
        "../document_intelligent/layoutmft/layoutlmv2/config/layoutlmv2_config.yaml"
    )
    cfg: Config = config.cfg
    for k, v in cfg.__dict__.items():
        print(f"{k}={v}")
