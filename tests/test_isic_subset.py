"""max_train_samples selects a deterministic subset; smaller budgets are prefixes of larger ones."""

from src.data.isic import ISIC2018


def _ids(ds):
    return [stem for _, _, stem in ds.items]


def test_no_subset_by_default(fake_isic_root):
    assert len(ISIC2018(fake_isic_root, split="train", image_size=8)) == 20


def test_subset_size(fake_isic_root):
    ds = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=5)
    assert len(ds) == 5


def test_subset_is_prefix_of_larger_subset(fake_isic_root):
    small = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=5)
    large = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=12)
    assert _ids(small) == _ids(large)[:5]


def test_subset_is_deterministic_and_seed_dependent(fake_isic_root):
    a = ISIC2018(
        fake_isic_root, split="train", image_size=8, max_train_samples=10, subset_seed=0
    )
    b = ISIC2018(
        fake_isic_root, split="train", image_size=8, max_train_samples=10, subset_seed=0
    )
    c = ISIC2018(
        fake_isic_root, split="train", image_size=8, max_train_samples=10, subset_seed=1
    )
    assert _ids(a) == _ids(b)
    assert _ids(a) != _ids(c)


def test_subset_larger_than_dataset_keeps_all(fake_isic_root):
    ds = ISIC2018(fake_isic_root, split="train", image_size=8, max_train_samples=999)
    assert len(ds) == 20
