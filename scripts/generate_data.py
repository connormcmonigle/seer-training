import argparse
import config
import seer_train


def main():
    parser = argparse.ArgumentParser(description="Seer Data Generation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Specifies where to look for the configuration file. Default 'config.yaml'",
    )

    args = parser.parse_args()
    cfg = config.Config(args.config)

    gen = seer_train.DataGenerator(cfg.data_write_path, cfg.target_sample_count, cfg.tt_mb_size).set_fixed_depth(
        cfg.fixed_depth).set_fixed_nodes(cfg.fixed_nodes).set_concurrency(cfg.concurrency)
    gen.generate_data()


if __name__ == '__main__':
    main()