import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse

import config
import seer_train
import dataset
import model


def train_step(nnue, sample, opt, queue, max_queue_size, report=False):
    pov, white, black, score, result = sample
    pred = nnue(pov, white, black)
    loss = model.loss_fn(score, result, pred)
    loss.backward()
    opt.step()
    nnue.zero_grad()

    if report:
        print(loss.item())

    if(len(queue) >= max_queue_size):
        queue.pop(0)

    queue.append(loss.item())


def main():
    parser = argparse.ArgumentParser(description="Seer Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Specifies where to look for the configuration file. Default 'config.yaml'",
    )

    args = parser.parse_args()
    cfg = config.Config(args.config)

    def sample_to_device(x): return tuple(
        map(lambda t: t.to(cfg.device, non_blocking=True), dataset.post_process(x)))

    print(f"Fine Tuning: {cfg.fine_tune}")
    nnue = model.NNUE(fine_tune=cfg.fine_tune).to(cfg.device)

    if (os.path.exists(cfg.model_save_path)):
        print(f"Loading model from {cfg.model_save_path} ... ")
        nnue.load_state_dict(torch.load(cfg.model_save_path))

    writer = SummaryWriter(cfg.visual_directory)
    opt = optim.Adadelta(nnue.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, cfg.step_size, gamma=cfg.gamma)

    queue = []
    total_steps = 0

    assert len(cfg.data_read_paths) == len(cfg.data_read_lengths)

    reader = seer_train.StochasticMultiplexSampleReader(
        cfg.data_read_lengths,
        [seer_train.SampleReader(path) for path in cfg.data_read_paths],
    )

    reader = dataset.SubsetConfigurable(sum(cfg.data_read_lengths), reader)

    for _ in range(cfg.epochs):
        train_data = dataset.SeerData(reader, cfg)

        train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=cfg.batch_size,
                                                        num_workers=cfg.concurrency,
                                                        pin_memory=True,
                                                        worker_init_fn=dataset.worker_init_fn)

        for i, sample in enumerate(train_data_loader):
            # update visual data
            if (i % cfg.test_rate) == 0 and i != 0:
                step = total_steps * cfg.batch_size
                train_loss = sum(queue) / len(queue)
                writer.add_scalar('train_loss', train_loss, step)

            if (i % cfg.save_rate) == 0 and i != 0:
                print(
                    f'Saving model to {cfg.model_save_path}, {cfg.bin_model_save_path}')
                nnue.flattened_parameters().tofile(cfg.bin_model_save_path)
                torch.save(nnue.state_dict(), cfg.model_save_path)

            train_step(nnue, sample_to_device(sample), opt, queue,
                       max_queue_size=cfg.max_queue_size, report=(0 == i % cfg.report_rate))
            total_steps += 1

        scheduler.step()


if __name__ == '__main__':
    main()
