import json
import random
import re
from pathlib import Path

# 🤗 Hugging-Face tokenizer setup
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"additional_special_tokens": ["<mstart>", "<mend>"]})

# --- your masking functions unchanged ---
def logical_tokenize_lines(code):
    lines = code.splitlines()
    logical_units = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        logical_units.append(stripped)
    return logical_units

def mask_logical_regions(code, min_regions=1, max_regions=5):
    logical_chunks = logical_tokenize_lines(code)
    n = len(logical_chunks)
    if n == 0:
        return code, []
    num_regions = min(max_regions, max(min_regions, n // 4))
    selected_indices = set()
    regions = []

    for _ in range(num_regions):
        for _ in range(10):
            idx = random.randint(0, n - 1)
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            original_line = logical_chunks[idx]
            logical_chunks[idx] = f"<mstart> {original_line} <mend>"
            regions.append(original_line)
            break

    masked_lines = []
    ptr = 0
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            masked_lines.append(line)
        else:
            leading_ws = re.match(r"^\s*", line).group()
            masked_lines.append(leading_ws + logical_chunks[ptr])
            ptr += 1

    return "\n".join(masked_lines), regions

# --- new: manually read your JSONL and process only the "code" field ---
def main():
    input_path  = Path("spp_train.jsonl")            # your raw dataset
    output_path = Path("masked_and_tokenized.jsonl") # where to write

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            record = json.loads(line)
            code_snippet = record.get("code", "").strip()
            if not code_snippet:
                continue

            # 1) mask
            stmt_masked, mask_info = mask_logical_regions(code_snippet)

            # 2) tokenize *just* that masked string
            enc = tokenizer(
                stmt_masked,
                add_special_tokens=False,
                return_offsets_mapping=False
            )
            tokens    = tokenizer.convert_ids_to_tokens(enc["input_ids"])
            token_ids = enc["input_ids"]

            # locate your <mstart>/<mend>
            mstart_id = tokenizer.convert_tokens_to_ids("<mstart>")
            mend_id   = tokenizer.convert_tokens_to_ids("<mend>")
            mask_positions = [
                i for i, tid in enumerate(token_ids) if tid in (mstart_id, mend_id)
            ]

            out = {
                "original_statement"   : code_snippet,
                "statement_with_mask"  : stmt_masked,
                "mask_info"            : mask_info,
                "tokens"               : tokens,
                "token_ids"            : token_ids,
                "mask_token_positions" : mask_positions,
            }
            f_out.write(json.dumps(out) + "\n")

    print(f"→ Finished. Wrote {output_path}")

if __name__ == "__main__":
    main()
