# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pytest
import requests

from recommenders.datasets.mind import download_mind, extract_mind


@pytest.mark.parametrize(
    "url, content_length, etag",
    [
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDdemo_train.zip",
            "17372879",
            '"66b4bd16ce4322e71ed7f2ec238cf0708d6b0b04651fe7e9bae7b7dd5e17bc40"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDdemo_dev.zip",
            "10080022",
            '"99def74b59c7e423854b79d4b68757e9e49813d8fc0cce6ee172bf5b3240f01d"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDdemo_utils.zip",
            "97292694",
            '"0b887006ffc2db8572366ae7d529c9eee24f0eb915c75ffc0963437655b10711"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip",
            "52994575",
            '"0d6d50cfb534b494a1153694f8a488170b28b0ac1c254363b565aa6b11c7761f"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip",
            "30948560",
            '"e3db712b7eb4339bc1729c5a0075c2e8204107690f9d5f0f974de4852622564f"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_utils.zip",
            "155178106",
            '"8047c5282f4ee819389e6ad757841e93705a7988af7d851184047c05337114ed"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDlarge_train.zip",
            "531360717",
            '"6a73f3b9ab3ba208895a18c1ac542dcf6bc9c7b817af433e60282d041cfec427"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDlarge_dev.zip",
            "103592887",
            '"526a6486eed6b4053b9460c42aac7239c1234cfecaf82a3c345b742d3521642f"',
        ),
        (
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDlarge_utils.zip",
            "150359301",
            '"8e05c4c768336dd8af8bd05d098a3572f657c984a5415e64fc85aaa315a32e15"',
        ),
    ],
)
def test_mind_url(url, content_length, etag):
    url_headers = requests.head(url, allow_redirects=True).headers
    assert url_headers["Content-Length"] == content_length
    assert url_headers["ETag"] == etag


def test_download_mind_demo(tmp):
    train_path, valid_path = download_mind(size="demo", dest_path=tmp)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 17372879
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 10080022


def test_extract_mind_demo(tmp):
    train_zip, valid_zip = download_mind(size="demo", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip, clean_zip_file=False)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 14707247
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 16077470
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 23120370
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 4434762
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 11591565
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 15624320
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588


def test_download_mind_small(tmp):
    train_path, valid_path = download_mind(size="small", dest_path=tmp)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 52994575
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 30948560


def test_extract_mind_small(tmp):
    train_zip, valid_zip = download_mind(size="small", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip, clean_zip_file=False)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 92019716
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 25811015
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 41202121
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 42838544
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 21960998
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 33519092
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588


def test_download_mind_large(tmp_path):
    train_path, valid_path = download_mind(size="large", dest_path=tmp_path)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 531360717
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 103592887


def test_extract_mind_large(tmp):
    train_zip, valid_zip = download_mind(size="large", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 1373844151
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 40305151
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 84881998
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588

    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 230662527
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 31958202
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 59055351
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
