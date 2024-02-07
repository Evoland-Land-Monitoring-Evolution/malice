from test.utils import create_fake_batch, default_lvicregmodule


def test_forward():
    C = 10
    pl_module = default_lvicregmodule(d_in=C)
    batch = create_fake_batch(b=2, t=4, c=C, h=64, w=64)
    repr_1, repr_2 = pl_module.forward(batch)
    assert repr_1.seg_map.shape[0] == repr_2.seg_map.shape[0]
    assert len(repr_1.seg_map.shape) == 4
    assert repr_1.feature_maps[0].shape[-1] == 8
    assert repr_1.seg_map.shape[-1] == pl_module.d_model
    print(repr_1.feature_maps[0].shape)


def test_training_step():
    C = 10
    pl_module = default_lvicregmodule(d_in=C)
    batch = create_fake_batch(b=2, t=4, c=C, h=64, w=64)
    pl_module.training_step(batch, batch_idx=1)
