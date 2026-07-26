from types import SimpleNamespace

import pytest

from lightning_yolo.callbacks import EpochLogger


def test_epoch_logger_report(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 12.0])
    monkeypatch.setattr(EpochLogger, "_now", staticmethod(lambda: next(times)))
    logger = EpochLogger()
    trainer = SimpleNamespace(current_epoch=1, callback_metrics={"val/map": 0.5, "val/map_50": 0.75})
    batch = ([object()] * 64, {})

    messages: list[str] = []
    monkeypatch.setattr("lightning_yolo.callbacks.rank_zero_info", messages.append)

    logger.on_train_epoch_start(trainer, SimpleNamespace())
    logger.on_train_batch_end(trainer, SimpleNamespace(), None, batch, 0)
    logger.on_train_epoch_end(trainer, SimpleNamespace())
    logger.on_validation_epoch_end(trainer, SimpleNamespace())

    assert len(messages) == 1
    msg = messages[0]
    assert msg.startswith("[epoch]")
    assert "epoch=1" in msg
    assert "img/s=32.0" in msg
    assert "val/map=0.500000" in msg
