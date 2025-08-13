# constants.py
from pathlib import Path

SPECIAL_TOKENS = ["<PAD>", "<BOS>"]

# читаем и валидируем словарь BIP39 (EN)
bip_path = Path("Data/bip39.txt")
if not bip_path.exists():
    raise FileNotFoundError("Data/bip39.txt not found (нужен официальный EN wordlist на 2048 слов)")

with bip_path.open("r", encoding="utf-8-sig") as f:  # utf-8-sig на случай BOM
    BIP39_WORDS = [line.strip().lower() for line in f if line.strip()]

# базовые проверки
if len(BIP39_WORDS) != 2048:
    raise ValueError(f"Expected 2048 words, got {len(BIP39_WORDS)}")
if len(set(BIP39_WORDS)) != len(BIP39_WORDS):
    raise ValueError("Duplicate words found in BIP39 list")
if any(" " in w or "\t" in w for w in BIP39_WORDS):
    raise ValueError("BIP39 words must not contain whitespace")

# полный вокаб: спец + BIP39
VOCAB = SPECIAL_TOKENS + BIP39_WORDS

WORD2IDX = {word: idx for idx, word in enumerate(VOCAB)}
IDX2WORD = {idx: word for word, idx in WORD2IDX.items()}
VOCAB_SIZE = len(VOCAB)

PAD_IDX = WORD2IDX["<PAD>"]   # 0
BOS_IDX = WORD2IDX["<BOS>"]   # 1
