import argparse
import os
import sys
import torch.multiprocessing as mp

current_dir = os.path.dirname(os.path.abspath(__file__))
lightning_npu_path = os.path.join(current_dir, "lightning_npu")
sys.path.append(lightning_npu_path)

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonUtils import (
    MAX_LEN,
    TOKEN2MASK,
    IterableJSONData,
)

# 导入NPU支持组件
import torch_npu
from lightning_npu.accelerators.npu import NPUAccelerator
from lightning_npu.strategies.npu import SingleNPUStrategy

class MaskedTokenizerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        tokenized = self.tokenizer(
            [ex["codons"] for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        seq_len = tokenized["input_ids"].shape[-1]
        species_index = torch.tensor([[ex["organism"]] for ex in examples])
        tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

        inputs = tokenized["input_ids"]
        targets = tokenized["input_ids"].clone()

        prob_matrix = torch.full(inputs.shape, 0.15)
        prob_matrix[torch.where(inputs < 5)] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(
            list((map(TOKEN2MASK.__getitem__, inputs[replaced].numpy())))
        )

        # 10% of the time, we replace masked input tokens with random vector.
        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool()
            & selected
            & ~replaced
        )
        random_idx = torch.randint(26, 90, prob_matrix.shape, dtype=torch.long)
        inputs[randomized] = random_idx[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

#        npu_device = torch.device("npu:0")
#        tokenized = {k: v.to(npu_device) for k, v in tokenized.items()}

        return tokenized


class plTrainHarness(pl.LightningModule):
    def __init__(self, model, learning_rate, warmup_fraction, train_dataloader=None, samples_count=None, batch_size=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction
        self.train_dataloader = train_dataloader
        self.samples_count = samples_count
        self.batch_size = batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # 显式计算total_steps
        if self.samples_count and self.batch_size:
            steps_per_epoch = self.samples_count // self.batch_size
            print(f"手动计算steps_per_epoch: {steps_per_epoch} (样本总数: {self.samples_count}, 批次大小: {self.batch_size})")
            total_steps = steps_per_epoch * self.trainer.max_epochs

        elif self.train_dataloader and hasattr(self.train_dataloader.dataset, '__len__'):
#        if self.train_dataloader is not None:
            steps_per_epoch = len(self.train_dataloader)
            total_steps = steps_per_epoch * self.trainer.max_epochs
        else:
            total_steps = self.trainer.estimated_stepping_batches  # �~G�~T��~V��~H
        
        if total_steps <= 0:
            raise ValueError(f"Total steps must be positive, got {total_steps}")
            
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=self.warmup_fraction,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        self.model.bert.set_attention_type("block_sparse")
        outputs = self.model(**batch)
        self.log_dict(
            dictionary={
                "loss": outputs.loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return outputs.loss


class DumpStateDict(pl.callbacks.ModelCheckpoint):
    def __init__(self, checkpoint_dir, checkpoint_filename, every_n_train_steps):
        super().__init__(
            dirpath=checkpoint_dir, every_n_train_steps=every_n_train_steps
        )
        self.checkpoint_filename = checkpoint_filename

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        model = trainer.model.model
        torch.save(
            model.state_dict(), os.path.join(self.dirpath, self.checkpoint_filename)
        )


def main(args):
    """Finetune the CodonTransformer model on NPU."""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        print(f"发现NPU设备: {torch.npu.current_device()}")
        device = "npu"
    else:
        print("未发现NPU设备，将使用CPU")
        device = "cpu"

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # 确保NPU环境可用
    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU device not found. Please check your environment.")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer-base")
#    model = model.to(npu_device)
    
    # 显式将模型转移到NPU
    if torch.npu.is_available():
        model = model.to(torch.device("npu:0"))
    
#    harnessed_model = plTrainHarness(model, args.learning_rate, args.warmup_fraction)

    # Load the training data
    train_data = IterableJSONData(args.dataset_dir, dist_env="slurm")

    samples_count = None
    if args.count_samples:
        print("开始统计数据集样本数...")
        samples_count = 0
        for _ in train_data:
            samples_count += 1
            if samples_count % 1000 == 0:
                print(f"已统计 {samples_count} 个样本...")
            print(f"数据集样本总数: {samples_count}")

    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=args.batch_size,
        num_workers=0 if args.debug else args.num_workers,
        persistent_workers=False if args.debug else True,
    )

        # 将data_loader传递给plTrainHarness
    harnessed_model = plTrainHarness(
        model=model,
        learning_rate=args.learning_rate,
        warmup_fraction=args.warmup_fraction,
        train_dataloader=data_loader
    )

    # Setup trainer and callbacks with NPU configuration
    save_checkpoint = DumpStateDict(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        every_n_train_steps=args.save_every_n_steps,
    )
    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        strategy=SingleNPUStrategy(device_index=0),  # 指定0号NPU卡
        accelerator=NPUAccelerator(),  # 使用NPU加速器
        devices=1,  # 单卡训练
        precision="bf16",
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        callbacks=[save_checkpoint],
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10
    )

    # Finetune the model on NPU
    trainer.fit(harnessed_model, data_loader)


if __name__ == "__main__":
    mp.set_start_method('spawn')  # 设置多进程启动方法为spawn
    parser = argparse.ArgumentParser(description="Finetune the CodonTransformer model on NPU.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="finetune.ckpt",
        help="Filename for the saved checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=15, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps to use for warmup",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=512,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--count_samples",
        action="store_true",
        help="手动统计数据集样本数（可能需要较长时间）"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)
