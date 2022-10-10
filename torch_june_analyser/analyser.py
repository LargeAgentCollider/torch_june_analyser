import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from torch_june import Runner

from torch_june_analyser.paths import default_config_path


class Analyser:
    def __init__(self, runner, save_path):
        self.runner = runner
        self.save_path = Path(save_path)
        self._set_parameters()
        self.results = None

    @classmethod
    def from_parameters(cls, params):
        runner = Runner.from_parameters(params)
        return cls(runner=runner, save_path=params["analyser"]["save_path"])

    @classmethod
    def from_file(cls, file=default_config_path):
        params = yaml.safe_load(open(file, "r"))
        return cls.from_parameters(params)

    def _get_network_names(self):
        return [n for n in self.runner.model.infection_networks.networks]

    def _set_parameters(self):
        # betas
        for network in self._get_network_names():
            self.runner.model.infection_networks.networks[
                network
            ].log_beta = torch.nn.Parameter(
                self.runner.model.infection_networks.networks[network].log_beta
            )

    def run(self):
        self.results, _ = self.runner()

    def get_gradient_cases_by_location(self):
        if self.results is None:
            raise ValueError("Need to run Runner first")
        total_cases = self.results["cases_per_timestep"].sum()
        total_cases.backward(retain_graph=True)
        ret = {}
        for name in self._get_network_names():
            ret[name] = self.runner.model.infection_networks.networks[
                name
            ].log_beta.grad.item()
        return ret

    def get_gradient_cases_by_location(self, date):
        if self.results is None:
            raise ValueError("Need to run Runner first")
        date = datetime.strptime(date, "%Y-%m-%d")
        date_idx = np.array(self.results["dates"]) == date
        cases = self.results["cases_per_timestep"][date_idx]
        cases.backward(retain_graph=True)
        ret = {}
        for name in self._get_network_names():
            ret[name] = self.runner.model.infection_networks.networks[
                name
            ].log_beta.grad.item()
        return ret

    def get_gradient_cases_by_age_location(self, date):
        if self.results is None:
            raise ValueError("Need to run Runner first")
        date = datetime.strptime(date, "%Y-%m-%d")
        date_idx = np.array(self.results["dates"]) == date
        age_bins = self.runner.age_bins
        ret = {}
        for age_bin in age_bins[1:]:
            age_bin = age_bin.item()
            ret[age_bin] = {}
            cases = self.results[f"cases_by_age_{age_bin:02d}"][date_idx]
            cases.backward(retain_graph=True)
            for name in self._get_network_names():
                ret[age_bin][name] = self.runner.model.infection_networks.networks[
                    name
                ].log_beta.grad.item()
        return ret
