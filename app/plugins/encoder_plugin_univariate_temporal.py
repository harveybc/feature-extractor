"""Univariate temporal encoder plugin (entry point `univariate_temporal`).

Thin wrapper over app.univariate_temporal. Its interface is NOT the window-to-vector
interface of the other encoder plugins: inputs are signal/observed_mask/delta_time
(B,T,1) and calendar (B,T,C); the output keeps the time axis, latent (B,T,D).
The target is never an input. `family` selects identity | random | ae | dae.
"""
from app import univariate_temporal as U


class Plugin:
    plugin_params = {"family": "ae", "window": 168, "calendar_dim": 6, "latent_dim": 8, "filters": 16,
                     "kernel_size": 3, "dilations": (1, 2, 4, 8, 16, 32), "seed": 0,
                     "learning_rate": 1e-3, "batch_size": 64}
    plugin_debug_vars = ["family", "window", "latent_dim", "filters", "dilations", "seed"]

    def __init__(self):
        self.params = dict(self.plugin_params)
        self.extractor = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def get_debug_info(self):
        return {k: self.params.get(k) for k in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def arch_config(self):
        p = self.params
        return U.ArchConfig(window=p["window"], calendar_dim=p["calendar_dim"], latent_dim=p["latent_dim"],
                            filters=p["filters"], kernel_size=p["kernel_size"], dilations=tuple(p["dilations"]),
                            decoder_filters=p["filters"])

    def configure_size(self, *args, **kwargs):
        p = self.params
        self.extractor = U.make_extractor(p["family"], self.arch_config(), p["seed"],
                                          learning_rate=p["learning_rate"], batch_size=p["batch_size"])
        return self.extractor

    def fit(self, train_batch, val_batch, early_stop=None):
        return self.extractor.fit(train_batch, val_batch, early_stop or U.EarlyStopConfig())

    def encode(self, batch):
        if isinstance(batch, dict):
            return self.extractor.encode_inputs(batch)
        return self.extractor.encode(batch)

    def save(self, out_dir, scope, feature_id, normalization=None):
        return self.extractor.export_donor(out_dir, scope, feature_id, normalization)

    def load(self, donor_dir, regime="R2", expected=None, seed=None):
        return U.load_donor(donor_dir, regime, seed=seed, expected=expected)
