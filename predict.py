#!/usr/bin/env python3
# predikt.py — предсказатель с поддержкой BTC, ETH и Solana

import sys, os, time, hashlib, requests, hmac, struct, base58
import torch
import torch.nn as nn
import concurrent.futures as futures
from nacl import signing
from datetime import datetime
from train import (
    FullTransformerModel, IDX2WORD, WORD2IDX,
    PAD_IDX, BOS_IDX, PHRASE_LEN, features_from_identifier
)
from bip_utils import (
    Bip39MnemonicValidator, Bip39SeedGenerator,
    Bip44, Bip44Coins, Bip44Changes,
    Bip84, Bip84Coins,
)
from eth_account import Account

# ───────────────────────── CONFIG ─────────────────────────
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH   = "Models/transformer_checkpoint.pt"
NUM_VARIANTS      = 8
BATCH_PHRASES     = 7
BTC_MAX_WORKERS   = 3
BTC_RETRIES       = 3

ETH_RPC_URL = os.getenv("ETH_RPC_URL", "https://rpc.ankr.com/eth/84ca8aa47c21b1dbff3d1b5bfd531462a19a11cf9bf675cb084427c54507c625")
SOL_RPC_URL = os.getenv("SOL_RPC_URL", "https://api.mainnet-beta.solana.com")

WORDS_EN = list(WORD2IDX.keys())  # полный словарь BIP39
os.makedirs("Data", exist_ok=True)

# ───────────────────────── INFER UTILS (match training) ─────────────────────────

ALLOWED_COINS = {"eth": 1, "btc": 2, "sol": 3}

def guess_coin(addr: str) -> str:
    a = (addr or "").strip().lower()
    if a.startswith("0x"): return "eth"
    if a.startswith("bc1"): return "btc"
    return "sol"

def make_addr_tensor(addr: str, coin: str | None = None) -> torch.Tensor:
    """Адрес -> байты (как в датасете): [:33] + индекс монеты."""
    coin = coin or guess_coin(addr)
    ci = ALLOWED_COINS.get(coin)
    if ci is None:
        raise ValueError(f"Unsupported coin '{coin}'. Use one of {list(ALLOWED_COINS.keys())}")
    core = features_from_identifier(addr, max_len=64)  # bytes
    core_list = list(core)
    pubkey_trunc = core_list[:33]      # ← оставляю твой 33/32 рассинхрон
    with_coin = pubkey_trunc + [ci]
    return torch.tensor([with_coin], dtype=torch.long, device=DEVICE)  # [1, L]

# ───────────────────────── CHECKSUM FIX ─────────────────────────

def fix_checksum(first11: str):
    """Подбираем 12-е слово, чтобы checksum сошлась."""
    v = Bip39MnemonicValidator()
    for w in WORDS_EN:
        cand = f"{first11} {w}"
        if v.IsValid(cand):
            return cand
    return None

# ───────────────────────── SAMPLING ─────────────────────────

def _topk_sample(probs: torch.Tensor, k: int, forbidden: set[int] | None = None) -> int:
    if forbidden:
        probs = probs.clone()
        for idx in forbidden:
            if 0 <= idx < probs.numel():
                probs[idx] = 0.0
        s = probs.sum()
        probs = probs / s if s > 0 else torch.full_like(probs, 1.0 / probs.numel())
    k = min(k, probs.numel())
    top_p, top_i = torch.topk(probs, k)
    top_p = top_p / top_p.sum()
    choice = torch.multinomial(top_p, 1).item()
    return top_i[choice].item()

# ───────────────────────── GENERATION (autoregressive with BOS) ─────────────────────────

@torch.no_grad()
def generate_one(model: nn.Module,
                 addr_tensor: torch.Tensor,
                 max_len: int = 12,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 forbid_repeat: bool = False) -> list[int]:
    model.eval()
    seq = torch.full((1, max_len + 1), PAD_IDX, dtype=torch.long, device=DEVICE)  # [+1 для BOS]
    seq[0, 0] = BOS_IDX

    used = set()
    for pos in range(1, max_len + 1):
        logits, _ = model(addr_tensor, seq[:, :pos])      # logits: [1, pos, vocab]
        logits_step = logits[0, pos - 1] / max(1e-8, temperature)
        probs = torch.softmax(logits_step, dim=0)

        forbidden = {PAD_IDX, BOS_IDX}
        if forbid_repeat and pos > 1:
            forbidden |= used

        nxt = _topk_sample(probs, k=top_k, forbidden=forbidden)
        seq[0, pos] = nxt
        if forbid_repeat:
            used.add(nxt)

    return seq[0, 1:].tolist()  # без BOS

@torch.no_grad()
def generate_phrases(model: nn.Module,
                     addr_tensor: torch.Tensor,
                     num_candidates: int = 16,
                     temperature: float = 0.9,
                     top_k_first: int = 64,
                     top_k_next: int = 32,
                     forbid_repeat: bool = True,
                     validate_checksum: bool = True) -> list[str]:
    model.eval()
    out, seen = [], set()

    # распределение первого слова (ввод только BOS)
    bos = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=DEVICE)
    logits0, _ = model(addr_tensor, bos)
    probs0 = torch.softmax(logits0[0, 0] / max(1e-8, temperature), dim=0)

    k0 = min(top_k_first, probs0.numel())
    p0_vals, p0_idx = torch.topk(probs0, k0)
    p0_vals = p0_vals / p0_vals.sum()

    for _ in range(num_candidates * 2):  # с запасом
        first_tok = p0_idx[torch.multinomial(p0_vals, 1).item()].item()
        if first_tok in (PAD_IDX, BOS_IDX):
            continue

        seq = torch.full((1, PHRASE_LEN + 1), PAD_IDX, dtype=torch.long, device=DEVICE)
        seq[0, 0] = BOS_IDX
        seq[0, 1] = first_tok

        used = {first_tok} if forbid_repeat else set()

        for pos in range(2, PHRASE_LEN + 1):
            logits, _ = model(addr_tensor, seq[:, :pos])
            step = logits[0, pos - 1] / max(1e-8, temperature)
            probs = torch.softmax(step, dim=0)
            nxt = _topk_sample(
                probs, k=top_k_next,
                forbidden=({PAD_IDX, BOS_IDX} | used) if forbid_repeat else {PAD_IDX, BOS_IDX}
            )
            seq[0, pos] = nxt
            if forbid_repeat:
                used.add(nxt)

        tok_ids = seq[0, 1:].tolist()
        words = [IDX2WORD[t] for t in tok_ids]

        if validate_checksum:
            fixed = fix_checksum(" ".join(words[:-1]))
            phrase = fixed if fixed else " ".join(words)
        else:
            phrase = " ".join(words)

        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
        if len(out) >= num_candidates:
            break

    return out

# ───────────────────────── DERIVATIONS ─────────────────────────

def seed_from_mnemonic(m):
    return Bip39SeedGenerator(m).Generate()

def derive_eth(seed):
    w = Bip44.FromSeed(seed, Bip44Coins.ETHEREUM).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    key = w.PrivateKey().Raw().ToHex()
    addr = Account.from_key(key).address
    return addr, key

def derive_btc(seed):
    w = Bip84.FromSeed(seed, Bip84Coins.BITCOIN).Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    key = w.PrivateKey().Raw().ToHex()
    addr = w.PublicKey().ToAddress()
    return addr, key

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

def derive_sol(seed, path="m/44'/501'/0'/0'"):
    priv32 = derive_path_ed25519(path, seed)
    sk     = signing.SigningKey(priv32)
    pubkey = sk.verify_key.encode()
    secret64 = (sk.encode() + pubkey).hex()
    return base58.b58encode(pubkey).decode(), secret64

# ───────────────────────── BALANCE CHECKS ─────────────────────────

def eth_balance(addr):
    try:
        wei = int(
            requests.post(
                ETH_RPC_URL,
                json={"jsonrpc":"2.0","method":"eth_getBalance","params":[addr,"latest"],"id":1},
                timeout=8,
            ).json()["result"],
            16,
        )
        return wei / 1e18
    except:
        return None

def sol_balance(addr):
    try:
        resp = requests.post(
            SOL_RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [addr]},
            timeout=8,
        )
        lamports = resp.json()["result"]["value"]
        return lamports / 1e9
    except:
        return None

def btc_balance_single(addr: str, retries: int = BTC_RETRIES):
    url = f"https://blockstream.info/api/address/{addr}"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                funded = data["chain_stats"]["funded_txo_sum"]
                spent  = data["chain_stats"]["spent_txo_sum"]
                return (funded - spent) / 1e8
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return None

# ───────────────────────── MAIN ─────────────────────────

def main():
    if len(sys.argv) < 2:
        addr = input("Enter address (or list via space): ").strip()
        if not addr: return
        addresses = addr.split()
    else:
        addresses = [a for a in sys.argv[1:] if len(a.strip()) >= 5]

    if not addresses:
        print("No valid addresses."); return

    model = FullTransformerModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model"])
    model.eval(); print("✅ Model loaded.")

    for addr in addresses:
        print(f"\n🔍 {addr}")
        addr_t = make_addr_tensor(addr)  # coin autodetected
        phrases = generate_phrases(
            model, addr_t,
            num_candidates=BATCH_PHRASES,
            temperature=0.9,
            top_k_first=64,
            top_k_next=32,
            forbid_repeat=True,
            validate_checksum=True,
        )
        if not phrases:
            print("No valid mnemonics"); continue

        btc_jobs = []
        for ph in phrases:
            seed = seed_from_mnemonic(ph)
            eth_a, eth_k = derive_eth(seed)
            btc_a, btc_k = derive_btc(seed)
            sol_a, sol_k = derive_sol(seed)

            eb = eth_balance(eth_a)
            sb = sol_balance(sol_a)

            btc_jobs.append((btc_a, ph, eth_a, eb, sol_a, sb))

        with futures.ThreadPoolExecutor(max_workers=BTC_MAX_WORKERS) as ex:
            fut_to_data = {ex.submit(btc_balance_single, j[0]): j for j in btc_jobs}
            for fut in futures.as_completed(fut_to_data):
                bb = fut.result()
                btc_a, ph, eth_a, eb, sol_a, sb = fut_to_data[fut]

                print("\n🔑", ph)
                print(f"ETH: {eth_a}, Bal: {eb}")
                print(f"BTC: {btc_a}, Bal: {bb}")
                print(f"SOL: {sol_a}, Bal: {sb}")

                with open("Data/attempts.txt", "a", encoding="utf-8") as logf:
                    logf.write(f"[{datetime.now().isoformat()}] TRY | Phrase: {ph}\n")
                    logf.write(f"    ETH: {eth_a}, Bal: {eb}\n")
                    logf.write(f"    BTC: {btc_a}, Bal: {bb}\n")
                    logf.write(f"    SOL: {sol_a}, Bal: {sb}\n\n")

if __name__ == "__main__":
    main()
