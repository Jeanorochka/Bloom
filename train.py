import json
import hashlib
import torch.nn as nn
import torch.optim as optim
import os, random, numpy as np, torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity
from constants import WORD2IDX, IDX2WORD, VOCAB_SIZE, PAD_IDX, BOS_IDX
from statistics import mean
from bip_utils import Bip39MnemonicGenerator, Bip39MnemonicValidator, Bip39SeedGenerator, Bip39WordsNum, Bip44, Bip44Coins, Bip44Changes, Bip84, Bip84Coins
from eth_account import Account
from nacl import signing
import hmac, struct, base58, os, random
from datetime import datetime

os.makedirs("Models", exist_ok=True)
os.makedirs("Data", exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHRASE_LEN = 12
ADDR_LEN = 34

ANCHOR_PHRASE = "drop aerobic behind select yellow sentence they lemon weasel whale luxury stay".split()
ANCHOR_WEIGHT = 5.0

TARGET_PHRASE = "pool scout later trade scorpion matrix fire wrap huge online keep vacuum"
TARGET_ETH = "0x6A91d64daad9B6487Df7F46FBdC5257186cA4128"
TARGET_SOL = "9Qjq89iTHyHjsHjCcZ3YYigR98LFstaLSucXXAA2qwaN"
TARGET_BTC = "bc1q5fw5svgpdqswg49g4kcrykypxe3q4melv2kakr"

REFERENCE_PHRASES = {
    tuple(TARGET_PHRASE.split()): 5.0,
    tuple("drop aerobic behind select yellow sentence they lemon weasel whale luxury stay".split()): 5.0,
}

RECALL_SCORE_MAP = {
    1: 0.001, 2: 0.002, 3: 0.003, 4: 0.005, 5: 0.008, 6: 0.015,
    7: 0.025, 8: 0.030, 9: 0.035, 10: 0.040, 11: 0.045, 12: 0.05
}

# -----------------------
# Determinism utilities
# -----------------------
def seed_everything(seed: int = 1126, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def make_loader(dataset, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

# -----------------------
# Safe helpers
# -----------------------
def normalize_to_bytes(identifier) -> bytes:
    """Простая нормализация строки идентификатора в байты (детерминированно)."""
    if identifier is None:
        return b""
    if isinstance(identifier, bytes):
        b = identifier
    else:
        b = str(identifier).strip().lower().encode("utf-8", "ignore")
    return b

def features_from_identifier(identifier: str, max_len: int | None = None) -> bytes:
    """Строка → байты, с опциональным обрезом/паддингом до max_len."""
    b = normalize_to_bytes(identifier)
    if max_len is not None:
        b = b[:max_len]
        if len(b) < max_len:
            b = b + b"\x00" * (max_len - len(b))
    return b

# -----------------------
# (Left as-is; not used below)
# -----------------------
def derive_pubkey_bytes(addr, coin):
    try:
        if coin == "eth":
            seed = Bip39SeedGenerator(" ".join(ANCHOR_PHRASE)).Generate()
            w = Bip44.FromSeed(seed, Bip44Coins.ETHEREUM).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
            return bytes.fromhex(w.PublicKey().RawCompressed().ToHex())
        elif coin == "btc":
            seed = Bip39SeedGenerator(" ".join(ANCHOR_PHRASE)).Generate()
            w = Bip84.FromSeed(seed, Bip84Coins.BITCOIN).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
            return bytes.fromhex(w.PublicKey().RawCompressed().ToHex())
        elif coin == "sol":
            seed = Bip39SeedGenerator(" ".join(ANCHOR_PHRASE)).Generate()
            priv32 = derive_path_ed25519("m/44'/501'/0'/0'", seed)
            sk = signing.SigningKey(priv32)
            return sk.verify_key.encode()
    except Exception:
        return None

def derive_path_ed25519(path: str, seed: bytes) -> bytes:
    digest = hmac.new(b"ed25519 seed", seed, hashlib.sha512).digest()
    k, c = digest[:32], digest[32:]
    for level in path.strip().split("/")[1:]:
        hardened = level.endswith("'")
        idx = int(level.rstrip("'")) + (0x80000000 if hardened else 0)
        idx_bytes = struct.pack(">I", idx)
        data = (b"\x00" + k if hardened else signing.SigningKey(k).verify_key.encode()) + idx_bytes
        digest = hmac.new(c, data, hashlib.sha512).digest()
        k, c = digest[:32], digest[32:]
    return k

def is_valid_unique_phrase(phrase, validator):
    return validator.IsValid(" ".join(phrase)) and len(set(phrase)) == len(phrase)

def generate_valid_phrase(generator, validator):
    while True:
        mnemonic = generator.FromWordsNumber(Bip39WordsNum.WORDS_NUM_12)
        phrase = str(mnemonic).split()
        if is_valid_unique_phrase(phrase, validator):
            return phrase

# -----------------------
# Dataset
# -----------------------
ALLOWED_COINS = {"eth": 1, "btc": 2, "sol": 3}

class MnemonicDataset(Dataset):
    def __init__(self, path):
        self.data = []
        self.skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    phrase = item["mnemonic"].split()
                    # проверка токенов по словарю
                    if not all(word in WORD2IDX for word in phrase):
                        self.skipped += 1
                        continue

                    for coin_name in ["eth", "btc", "sol"]:
                        address = item.get(coin_name)
                        ci = ALLOWED_COINS.get(coin_name)
                        if ci is None or not address:
                            self.skipped += 1
                            continue

                        # фичи из идентификатора (строки адреса)
                        addr_core = features_from_identifier(address, max_len=64)  # bytes
                        addr_core_list = list(addr_core)

                        # ТВОЙ РАССИНХРОН ОСТАВЛЕН: всегда срезаем 33 байта,
                        # несмотря на expected_len = 32 для sol.
                        expected_len = 33 if coin_name in ("eth", "btc") else 32
                        pubkey_trunc = addr_core_list[:33]  # намеренно [:33]

                        # приклеиваем индекс монеты в конец
                        addr_with_coin = pubkey_trunc + [ci]

                        self.data.append((addr_with_coin, phrase, ci))
                except json.JSONDecodeError:
                    self.skipped += 1

        print(f"Loaded {len(self.data):,} samples | Skipped: {self.skipped:,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        addr_with_coin, phrase, ci = self.data[idx]

        addr_bytes = torch.tensor(addr_with_coin, dtype=torch.long)  # [L_addr(+1)]
        phrase_tensor = torch.tensor([WORD2IDX[w] for w in phrase], dtype=torch.long)

        # веса/класс энтропии
        if len(set(phrase)) < len(phrase):
            weight = 0.0
            entropy_class = 0.0
        else:
            weight = 1.0
            entropy_class = 1.0

        return (
            addr_bytes,
            phrase_tensor,
            torch.tensor(weight, dtype=torch.float),
            torch.tensor(entropy_class, dtype=torch.float),
        )

# -----------------------
# Model
# -----------------------
class FullTransformerModel(nn.Module):
    def __init__(self, max_addr_len=64, max_seq_len=32):
        super().__init__()
        self.embed_phrase = nn.Embedding(VOCAB_SIZE, 128, padding_idx=PAD_IDX)
        self.pos_tok = nn.Embedding(max_seq_len, 128)
        self.byte_embed = nn.Embedding(256, 64)
        self.pos_addr = nn.Embedding(max_addr_len, 64)

        self.addr_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=4
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=8
        )
        self.proj_addr = nn.Linear(64, 128)
        self.fc = nn.Linear(128, VOCAB_SIZE)
        self.dropout = nn.Dropout(0.2)
        self.coin_embed = nn.Embedding(4, 64)

    def forward(self, addr_bytes, phrase_input):
        # addr_bytes: [B, Laddr(+1 с coin)], phrase_input: [B, Lseq]
        coin_idx = addr_bytes[:, -1]
        addr_core = addr_bytes[:, :-1]            # [B, Laddr]
        B, Laddr = addr_core.size()
        _, Lseq   = phrase_input.size()

        # позиционные индексы «на лету»
        pos_addr = torch.arange(Laddr, device=addr_core.device).unsqueeze(0)  # [1, Laddr]
        pos_tok  = torch.arange(Lseq,  device=phrase_input.device).unsqueeze(0)  # [1, Lseq]

        addr_embed_core = self.byte_embed(addr_core) + self.pos_addr(pos_addr)
        coin_vec = self.coin_embed(coin_idx).unsqueeze(1)  # [B,1,64]
        # без in-place:
        addr_embed = torch.cat([addr_embed_core[:, :1, :] + coin_vec,
                                addr_embed_core[:, 1:, :]], dim=1)

        phrase_embed = self.embed_phrase(phrase_input) + self.pos_tok(pos_tok)
        phrase_embed = self.dropout(phrase_embed)

        # кодер адреса
        addr_encoded = self.addr_encoder(addr_embed)
        mem = self.proj_addr(addr_encoded)

        # маска причинности
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Lseq).to(phrase_input.device)

        # МАСКА ПАДДИНГОВ для tgt (True = паддинг)
        tgt_key_padding_mask = (phrase_input == PAD_IDX)    # [B, Lseq], bool

        out = self.decoder(
            tgt=phrase_embed,
            memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            # memory_key_padding_mask=None  # начнёшь паддить addr — добавишь
        )
        logits = self.fc(out)
        return logits, out.mean(dim=1)

# -----------------------
# Safe stubs (no-op)
# -----------------------
def inject_synthetic_batch(model, optimizer, criterion, num_samples=64):
    """Заглушка: ничего не делает, только сообщение (чтобы тренировочный цикл не падал)."""
    print("[inject_synthetic_batch] skipped (stub)")

def check_target_recovery(model, epoch):
    """Заглушка: всегда возвращает False, без сетевых вызовов."""
    print("[check_target_recovery] skipped (stub)")
    return False

# -----------------------
# Train
# -----------------------
def train(seed=1126):
    epoch_accs = []
    dataset = MnemonicDataset("Data/dataset.jsonl")
    dataloader = make_loader(dataset, batch_size=64, seed=seed)

    model = FullTransformerModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1, reduction="none")

    best_acc = 0.0
    checkpoint_path = "Models/transformer_checkpoint.pt"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"🔁 Resuming from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, 10000):
            model.train()
            total_loss, total_acc, count = 0.0, 0.0, 0

            for addr_bytes, phrase_tensor, weights, entropy_class in tqdm(dataloader, desc=f"Epoch {epoch}"):
                addr_bytes = addr_bytes.to(DEVICE, non_blocking=True)
                phrase_tensor = phrase_tensor.to(DEVICE, non_blocking=True)
                weights = weights.to(DEVICE)
                entropy_class = entropy_class.to(DEVICE)

                optimizer.zero_grad()
                recall_n = random.choice(list(RECALL_SCORE_MAP.keys()))
                decoder_input = torch.full_like(phrase_tensor[:, :recall_n], PAD_IDX)
                decoder_input[:, 0] = BOS_IDX
                decoder_input[:, 1:] = phrase_tensor[:, :recall_n - 1]
                target_output = phrase_tensor[:, :recall_n]

                logits, addr_summary = model(addr_bytes, decoder_input)

                # контрастная часть (как у тебя)
                output_tokens = logits.argmax(dim=-1)
                embedded_tokens = model.embed_phrase(output_tokens)
                phrase_repr = embedded_tokens.mean(dim=1)
                contrast_loss = 1 - cosine_similarity(phrase_repr, addr_summary, dim=1).mean()

                loss_all = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    target_output.reshape(-1)
                ).reshape(target_output.shape)

                pred = logits.argmax(dim=-1)
                if recall_n > 1:
                    correct_words = (pred[:, 1:] == target_output[:, 1:]).float().sum(dim=1)
                    denom = max(recall_n - 1, 1)
                    recall_fraction = correct_words / denom
                    epoch_accs.append(recall_fraction.mean().item())  # <-- добавляем float ПОСЛЕ вычисления
                    total_acc += (recall_fraction.mean().item() * RECALL_SCORE_MAP[recall_n])
                else:
                    epoch_accs.append(0.0)

                loss_ce = (loss_all.sum(dim=1) * weights * entropy_class).mean()
                loss = loss_ce + 0.3 * contrast_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(count, 1)
            avg_acc = total_acc / max(count, 1)

            epoch_metric = mean(epoch_accs) if epoch_accs else 0.0
            epoch_accs.clear()

            if epoch % 40 == 0:
                print("Attempting to recover TARGET_PHRASE...")
                success = check_target_recovery(model, epoch)
                if success:
                    avg_acc += 0.01
                    print("Target phrase recovered. Boosting accuracy.")
                else:
                    print("Target phrase NOT recovered.")

            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
            scheduler.step()

            if epoch % 5 == 0:
                print("Injecting synthetic mnemonics (stub)...")
                inject_synthetic_batch(model, optimizer, criterion)

            if avg_acc > best_acc:
                best_acc = avg_acc
                print("Saving best model...")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc
                }, checkpoint_path)

    except KeyboardInterrupt:
        print("Interrupted. Saving checkpoint...")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }, checkpoint_path)
        print("Checkpoint saved.")

if __name__ == "__main__":
    SEED = 1126
    seed_everything(SEED, deterministic=True)
    print("Starting training...")
    train(SEED)
