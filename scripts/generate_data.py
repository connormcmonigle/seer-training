import config
import seer_train

cfg = config.Config('config.yaml')

gen = seer_train.DataGenerator(cfg.data_write_path, cfg.target_sample_count, cfg.tt_mb_size).set_concurrency(cfg.concurrency)
gen.generate_data()
